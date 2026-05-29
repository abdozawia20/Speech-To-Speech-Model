import sys
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor

# Enable TF32 on A100 for free ~10-20% speedup on matrix multiplications
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Monkey-patch PEFT for Phi-4-multimodal-instruct compatibility
# Phi-4's modeling_phi4mm.py was written against an older PEFT API that had:
#   1. prepare_inputs_for_generation on Phi4MMModel (needed by PeftModelForCausalLM.__init__)
#   2. active_adapter as a mutable list (now a read-only str property in peft>=0.8)
try:
    import peft
    from peft.tuners.tuners_utils import BaseTuner

    _orig_peft_causal_init = peft.peft_model.PeftModelForCausalLM.__init__
    def _patched_peft_causal_init(self, model, peft_config, adapter_name="default"):
        if not hasattr(model, "prepare_inputs_for_generation"):
            model.prepare_inputs_for_generation = lambda *args, **kwargs: None
        _orig_peft_causal_init(self, model, peft_config, adapter_name)
    peft.peft_model.PeftModelForCausalLM.__init__ = _patched_peft_causal_init

    class _AppendableStr(str):
        """str subclass with a no-op append() for legacy peft API compatibility."""
        def append(self, _):
            pass

    def get_active_adapter(self):
        val = getattr(self, "_active_adapter_val", "default")
        return _AppendableStr(val)

    def set_active_adapter(self, value):
        self._active_adapter_val = value

    BaseTuner.active_adapter = property(get_active_adapter, set_active_adapter)
except Exception:
    pass

# Monkey-patch Cache and DynamicCache to support get_usable_length in newer transformers versions
try:
    from transformers.cache_utils import Cache, DynamicCache
    for cache_cls in (Cache, DynamicCache):
        if not hasattr(cache_cls, "get_usable_length"):
            cache_cls.get_usable_length = lambda self, new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
except Exception:
    pass

# Monkey-patch Phi-4's forward() to guard against num_logits_to_keep=None.
#
# Root cause: transformers 4.57.x passes num_logits_to_keep=None through the
#   generate() → prepare_inputs_for_generation() → forward() pipeline.
# Phi-4's cached modeling_phi4mm.py then executes:
#   hidden_states[:, -num_logits_to_keep:, :]
# which crashes with "bad operand type for unary -: 'NoneType'" because Python
# cannot negate None.  The inner forward signature defaults to 0 (= "keep all
# logits"), so mapping None → 0 restores that safe default without changing
# any model outputs.
try:
    from transformers.utils import is_flash_attn_2_available  # noqa: F401 – confirms transformers is importable
    import transformers.dynamic_module_utils as _dmu

    # The class lives in the trust_remote_code cached module; find it by walking
    # all loaded modules that expose Phi4MMForCausalLM.
    import sys as _sys
    for _mod in list(_sys.modules.values()):
        _cls = getattr(_mod, "Phi4MMForCausalLM", None)
        if _cls is not None and hasattr(_cls, "forward"):
            _orig_phi4_forward = _cls.forward

            def _patched_phi4_forward(self, *args, num_logits_to_keep=0, **kwargs):
                # Coerce None → 0 so hidden_states[:, -0:, :] == hidden_states[:, :, :]
                if num_logits_to_keep is None:
                    num_logits_to_keep = 0
                return _orig_phi4_forward(self, *args, num_logits_to_keep=num_logits_to_keep, **kwargs)

            _cls.forward = _patched_phi4_forward
            print(f"[model.py] Patched Phi4MMForCausalLM.forward for num_logits_to_keep=None fix.")
            break
except Exception as _e:
    # Non-fatal: if the class isn't loaded yet the patch will be applied lazily
    # via the generate_speech wrapper below.
    pass


# Add project root to sys.path so we can import encoders from the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from encoders import VQGANEncoder

BANDWIDTH     = 1.5
TOKEN_OFFSET  = 100_000
NUM_CODEBOOKS = 2     # at 1.5 kbps
CODEBOOK_SIZE = 1024  # EnCodec codebook entries per codebook at 1.5 kbps


class OmniPhiS2ST(nn.Module):
    """
    End-to-End Speech-to-Speech Translation model.

    Source block:  Phi-4-multimodal-instruct AutoProcessor
                   (internal WavLM-style feature extraction + bridge projection)
    LLM core:      Phi-4-multimodal-instruct (fine-tuned via LoRA speech adapter)
    Target block:  VQGANEncoder (EnCodec, frozen) for tokenization and synthesis
    """

    PHI4_MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
    PHI4_HUB_ID   = "microsoft/Phi-4-multimodal-instruct"

    def __init__(self, phi4_model_id: str = PHI4_MODEL_ID, device: str = "cuda"):
        super().__init__()
        self.device = device

        # ── Source & LLM Block ──────────────────────────────────────────────
        # Processor: try the given path first; fall back to hub if broken.
        # Checkpoints saved by OmniPhiTrainer may omit preprocessor_config.json
        # due to a Phi4MMProcessor bug — loading from the hub gives the exact
        # same (never-fine-tuned) weights and config.
        print("[OmniPhiS2ST] Loading AutoProcessor...")
        try:
            self.processor = AutoProcessor.from_pretrained(
                phi4_model_id, trust_remote_code=True
            )
        except (TypeError, OSError) as e:
            print(f"[OmniPhiS2ST] Processor load failed ({e.__class__.__name__}). "
                  f"Falling back to hub: {self.PHI4_HUB_ID}")
            self.processor = AutoProcessor.from_pretrained(
                self.PHI4_HUB_ID, trust_remote_code=True
            )

        print("[OmniPhiS2ST] Loading Phi-4 model...")
        # Prefer FlashAttention-2 on A100 (much faster); fall back to SDPA if not installed
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            print("[OmniPhiS2ST] FlashAttention-2 detected — using flash_attention_2.")
        except ImportError:
            attn_impl = "sdpa"
            print("[OmniPhiS2ST] flash_attn not installed — falling back to sdpa. "
                  "Install with: pip install flash-attn --no-build-isolation")
        if device.startswith("cuda"):
            self.phi4 = AutoModelForCausalLM.from_pretrained(
                phi4_model_id,
                torch_dtype=torch.bfloat16,
                _attn_implementation=attn_impl,
                trust_remote_code=True,
                # device_map="auto", # for local
                device_map={"": 0}, # for colab
            )
        else:
            self.phi4 = AutoModelForCausalLM.from_pretrained(
                phi4_model_id,
                torch_dtype=torch.bfloat16,
                _attn_implementation="sdpa",   # use flash_attention_2 if available
                trust_remote_code=True,
            ).to(device)

        # Apply LoRA: freezes vision/text backbone; trains only speech adapter
        self.phi4.set_lora_adapter("speech")
        self.phi4.enable_input_require_grads()
        print("[OmniPhiS2ST] LoRA speech adapter applied. Non-audio layers frozen.")

        # ── Target Codec Block (frozen) ──────────────────────────────────────
        print("[OmniPhiS2ST] Loading VQGANEncoder (EnCodec)...")
        self.vqgan = VQGANEncoder(model_name="facebook/encodec_24khz")
        self.vqgan.model.eval()
        self.vqgan.model.to(device)
        for p in self.vqgan.model.parameters():
            p.requires_grad = False
        print(f"[OmniPhiS2ST] VQGANEncoder frozen on {device}.")

    @property
    def hf_device_map(self):
        val = getattr(self.phi4, "hf_device_map", None)
        if val is None:
            raise AttributeError("hf_device_map is not set on the base model")
        return val

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "_modules" in self.__dict__ and "phi4" in self._modules:
                return getattr(self.phi4, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def save_pretrained(self, save_directory, **kwargs):
        """
        Override save_pretrained to force safe_serialization=False.

        Why: Phi-4 ties embed_tokens.weight ↔ lm_head.weight to the same tensor.
        safetensors (the default) refuses to serialize shared-memory tensors and
        raises a RuntimeError. Falling back to pytorch_model.bin (pickle) handles
        tied weights correctly and loads identically with from_pretrained().

        This override is called by BOTH trainer.save_model() AND the internal
        _save_checkpoint, so it works regardless of which Trainer class is used.
        """
        kwargs["safe_serialization"] = False
        import os
        os.makedirs(save_directory, exist_ok=True)
        self.phi4.save_pretrained(save_directory, **kwargs)

    def forward(
        self,
        input_ids,
        labels,
        attention_mask,
        input_audio_embeds,
        audio_embed_sizes,
        audio_attention_mask=None,
        input_mode=2,           # 2 = speech mode (required by Phi-4)
        **kwargs,
    ):
        """
        Standard causal LM forward pass.

        The Cross-Entropy loss is computed only on positions where labels != -100,
        which corresponds exclusively to the target EnCodec token IDs.
        Phi-4 handles this internally because its LM head returns a standard
        CausalLMOutputWithPast with .loss already computed.

        NOTE: The HF Trainer does NOT auto-move custom keys (input_audio_embeds,
        audio_embed_sizes) to the GPU device — it only moves standard keys like
        input_ids. We do it here explicitly to avoid a blocking implicit copy
        inside Phi-4's forward pass that costs ~30-50ms per step.
        """
        # Determine target device and dtype from the first trainable parameter
        # (avoids hardcoding and works with device_map / multi-GPU layouts)
        _param  = next(self.phi4.parameters())
        _device = _param.device
        _dtype  = _param.dtype   # bfloat16

        # ── GPU placement + dtype cast (eliminates CPU→GPU copy inside Phi-4) ──
        input_audio_embeds = input_audio_embeds.to(device=_device, dtype=_dtype, non_blocking=True)
        audio_embed_sizes  = audio_embed_sizes.to(device=_device, non_blocking=True)

        # 🚨 THE CRITICAL AUTOGRAD FIX 🚨
        # Force the input audio embedding tensor to require gradients during training.
        # This registers the entire downstream computational graph to track grads,
        # resolving the "None of the inputs have requires_grad=True" warning and 
        # avoiding CheckpointError mismatches during gradient checkpointing recomputation.
        if self.training:
            input_audio_embeds.requires_grad_(True)

        if audio_attention_mask is not None:
            audio_attention_mask = audio_attention_mask.to(device=_device, non_blocking=True)

        return self.phi4(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            input_audio_embeds=input_audio_embeds,
            audio_embed_sizes=audio_embed_sizes,
            audio_attention_mask=audio_attention_mask,
            input_mode=input_mode,
        )

    @torch.no_grad()
    def generate_speech(
        self,
        source_audio: np.ndarray,
        source_sr: int = 16000,
        max_new_tokens: int = 400,   # 400 tokens ≈ 10s audio; 200 was too short for most sentences
    ) -> np.ndarray:
        """
        End-to-end inference: English audio → german audio waveform.

        Args:
            source_audio:   Raw float32 numpy array at source_sr Hz.
            source_sr:      Sampling rate of source_audio.
            max_new_tokens: Upper limit on generated token count.
                            At 1.5 kbps: 800 tokens ≈ 20 seconds of audio.

        Returns:
            numpy array: Synthesized target audio at 24kHz.
        """
        self.phi4.eval()
        # Keep VQGANEncoder on CPU during LLM generation to maximise free VRAM.
        # It will be moved to device only for the final decode step.
        self.vqgan.model.eval().cpu()

        # ── Lazy forward-patch: ensure num_logits_to_keep=None is handled ────
        # The trust_remote_code class may not be in sys.modules at import time,
        # so we re-check here after the model is definitely loaded.
        _phi4_cls = type(self.phi4)
        if not getattr(_phi4_cls, "_num_logits_patched", False):
            _orig_fwd = _phi4_cls.forward
            def _safe_fwd(self_inner, *args, num_logits_to_keep=0, **kwargs):
                if num_logits_to_keep is None:
                    num_logits_to_keep = 0
                return _orig_fwd(self_inner, *args, num_logits_to_keep=num_logits_to_keep, **kwargs)
            _phi4_cls.forward = _safe_fwd
            _phi4_cls._num_logits_patched = True
            print(f"[generate_speech] Patched {_phi4_cls.__name__}.forward for num_logits_to_keep=None.")

        # ── Step 1: Build prompt and process source audio ───────────────────
        user_message = {
            "role": "user",
            "content": "<|audio_1|>\nTranslate this to spoken german:",
        }
        prompt_text = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=prompt_text,
            audios=[(source_audio, source_sr)],
            return_tensors="pt",
        ).to(self.device)

        # Remove input_mode if it's already in inputs to prevent duplicate argument error in generate()
        inputs.pop("input_mode", None)

        # ── Step 2: Auto-regressively generate target token IDs ─────────────
        eos_id = self.processor.tokenizer.eos_token_id
        torch.cuda.empty_cache()  # flush any leftover allocations before generation
        print(f"[generate_speech] Generating up to {max_new_tokens} tokens on {self.device}...")


        generated_ids = self.phi4.generate(
            **inputs,
            input_mode=2,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            do_sample=True,
            temperature=0.6,
            top_p=0.90,
            repetition_penalty=1.3,
            # LogitsProcessorList removed — causes num_logits_to_keep=None crash
            # in transformers 4.57.3. The token filter at Step 3 below discards
            # any stray text tokens the model might generate.
        )
        print(f"[generate_speech] Generation done. Total ids shape: {generated_ids.shape}")

        # Strip the prompt prefix; keep only newly generated tokens
        prompt_len    = inputs["input_ids"].shape[1]
        new_token_ids = generated_ids[0, prompt_len:].tolist()

        # ── Step 3: Remove EOS / suffix tokens ──────────────────────────────
        #    The model will generate <|end|><|endoftext|> when done.
        #    These have IDs < TOKEN_OFFSET so they are easy to filter out.
        audio_token_ids = [t for t in new_token_ids if t >= TOKEN_OFFSET]

        if len(audio_token_ids) == 0:
            print("[generate_speech] Warning: no audio tokens generated.")
            return np.zeros(16000, dtype=np.float32)

        # ── Step 4: Reverse offset & validate range ──────────────────────────
        raw_codes = [t - TOKEN_OFFSET for t in audio_token_ids]

        # Clamp to valid codebook range to prevent CUDA out-of-bounds index
        # assertions in EncodecVectorQuantization.decode() when the model
        # generates tokens outside [TOKEN_OFFSET, TOKEN_OFFSET+CODEBOOK_SIZE-1].
        oob = sum(1 for c in raw_codes if not (0 <= c < CODEBOOK_SIZE))
        if oob > 0:
            print(f"[generate_speech] Warning: {oob}/{len(raw_codes)} codes out-of-range — "
                  f"model may not be sufficiently fine-tuned. Clamping to [0, {CODEBOOK_SIZE-1}].")
        raw_codes = [max(0, min(c, CODEBOOK_SIZE - 1)) for c in raw_codes]

        # ── Step 5: Unflatten 1D → [1, Codebooks, Frames] ───────────────────
        #    We interleaved as [CB0_F0, CB1_F0, CB0_F1, CB1_F1, ...]
        #    Trim to a multiple of NUM_CODEBOOKS to avoid shape errors
        n = (len(raw_codes) // NUM_CODEBOOKS) * NUM_CODEBOOKS
        raw_codes = raw_codes[:n]

        codes_array = np.array(raw_codes, dtype=np.int64).reshape(-1, NUM_CODEBOOKS)
        # codes_array shape: [Frames, Codebooks]
        # EnCodec API expects audio_codes of shape [nb_frames, batch, codebooks, frames].
        # With chunk_length=None (as in encodec_24khz) it does frame = audio_codes[0],
        # so we wrap with two unsqueeze(0) calls: [Frames, Codebooks] → [1, 1, 2, Frames].
        # Move VQGAN back to the compute device only for this decode step
        self.vqgan.model.to(self.device)
        codes_tensor = (
            torch.tensor(codes_array, dtype=torch.long)
            .T                    # [Frames, CB] → [CB, Frames]
            .unsqueeze(0)         # → [1, CB, Frames]  (batch dim)
            .unsqueeze(0)         # → [1, 1, CB, Frames]  (nb_frames dim)
            .to(self.device)
        )  # → [nb_frames=1, batch=1, codebooks=2, frames]

        # ── Step 6: Decode with VQGANEncoder → audio waveform ───────────────
        #    EncodecModel.decode() expects (audio_codes, audio_scales)
        #    audio_scales must be a list of length nb_frames; None means no scale.
        decoded = self.vqgan.model.decode(codes_tensor, [None])
        # decoded.audio_values shape: [1, 1, T]  at 24kHz
        waveform = decoded.audio_values.squeeze().cpu().numpy()

        return waveform
