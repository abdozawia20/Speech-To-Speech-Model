import sys
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor

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

# Add project root to sys.path so we can import encoders from the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from encoders import VQGANEncoder

BANDWIDTH    = 1.5
TOKEN_OFFSET = 100_000
NUM_CODEBOOKS = 2  # at 1.5 kbps

class OmniPhiS2ST(nn.Module):
    """
    End-to-End Speech-to-Speech Translation model.

    Source block:  Phi-4-multimodal-instruct AutoProcessor
                   (internal WavLM-style feature extraction + bridge projection)
    LLM core:      Phi-4-multimodal-instruct (fine-tuned via LoRA speech adapter)
    Target block:  VQGANEncoder (EnCodec, frozen) for tokenization and synthesis
    """

    PHI4_MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

    def __init__(self, phi4_model_id: str = PHI4_MODEL_ID, device: str = "cuda"):
        super().__init__()
        self.device = device

        # ── Source & LLM Block ──────────────────────────────────────────────
        print("[OmniPhiS2ST] Loading Phi-4 AutoProcessor...")
        self.processor = AutoProcessor.from_pretrained(
            phi4_model_id, trust_remote_code=True
        )

        print("[OmniPhiS2ST] Loading Phi-4 model...")
        if device.startswith("cuda"):
            self.phi4 = AutoModelForCausalLM.from_pretrained(
                phi4_model_id,
                torch_dtype=torch.bfloat16,
                _attn_implementation="sdpa",   # use flash_attention_2 if available
                trust_remote_code=True,
                device_map="auto",
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
        print("[OmniPhiS2ST] LoRA speech adapter applied. Non-audio layers frozen.")

        # ── Target Codec Block (frozen) ──────────────────────────────────────
        print("[OmniPhiS2ST] Loading VQGANEncoder (EnCodec)...")
        self.vqgan = VQGANEncoder(model_name="facebook/encodec_24khz")
        self.vqgan.model.eval()
        # Keep vqgan on CPU during training to save GPU VRAM
        self.vqgan.model.to("cpu")
        for p in self.vqgan.model.parameters():
            p.requires_grad = False
        print("[OmniPhiS2ST] VQGANEncoder frozen on CPU.")

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
        """
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
        max_new_tokens: int = 800,
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
        self.vqgan.model.eval().to(self.device)

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

        # ── Step 2: Auto-regressively generate target token IDs ─────────────
        eos_id = self.processor.tokenizer.eos_token_id
        generated_ids = self.phi4.generate(
            **inputs,
            input_mode=2,                   # speech mode
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            do_sample=False,                # greedy for determinism
        )

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

        # ── Step 4: Reverse offset ───────────────────────────────────────────
        raw_codes = [t - TOKEN_OFFSET for t in audio_token_ids]

        # ── Step 5: Unflatten 1D → [1, Codebooks, Frames] ───────────────────
        #    We interleaved as [CB0_F0, CB1_F0, CB0_F1, CB1_F1, ...]
        #    Trim to a multiple of NUM_CODEBOOKS to avoid shape errors
        n = (len(raw_codes) // NUM_CODEBOOKS) * NUM_CODEBOOKS
        raw_codes = raw_codes[:n]

        codes_array = np.array(raw_codes, dtype=np.int64).reshape(-1, NUM_CODEBOOKS)
        # codes_array shape: [Frames, Codebooks]
        # EnCodec expects:   [Batch=1, Codebooks, Frames]
        codes_tensor = (
            torch.tensor(codes_array, dtype=torch.long)
            .T.unsqueeze(0)
            .to(self.device)
        )  # → [1, 2, Frames]

        # ── Step 6: Decode with VQGANEncoder → audio waveform ───────────────
        #    EncodecModel.decode() expects (audio_codes, audio_scales)
        #    audio_scales can be None for simple reconstruction.
        decoded = self.vqgan.model.decode(codes_tensor, [None])
        # decoded[0] shape: [1, 1, T]  at 24kHz
        waveform = decoded.audio_values.squeeze().cpu().numpy()

        return waveform
