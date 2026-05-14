import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
import numpy as np
from contextlib import contextmanager
from transformers import (
    SpeechT5ForSpeechToSpeech, SpeechT5Processor, SpeechT5HifiGan,
    WavLMModel, Wav2Vec2FeatureExtractor
)
from transformers.modeling_outputs import BaseModelOutput
from datasets import load_from_disk, load_dataset
import dataset_loader
import sys
import json
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import gc
from functools import partial
from speechbrain.inference.speaker import EncoderClassifier


class Conv1DBridge(nn.Module):
    """
    Revised Bridge using LayerNorm for BATCH_SIZE=1 stability.
    """
    def __init__(self, dim=768, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=padding)
        self.act1 = nn.GELU()
        self.norm1 = nn.LayerNorm(dim)
        
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=padding)
        self.act2 = nn.GELU()
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x shape: (Batch, Seq, Dim)
        residual = x
        
        # Conv1d expects (Batch, Dim, Seq)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.norm1(self.act1(x))
        
        x = x.transpose(1, 2)
        x = self.conv2(x)
        x = x.transpose(1, 2)
        x = self.norm2(self.act2(x))

        return x + residual


class MockEncoder(torch.nn.Module):
    """
    Bypasses the SpeechT5 encoder by returning pre-computed hidden states.
    This allows us to use WavLM + Bridge as the encoder while still
    leveraging the auto-regressive generate_speech method.

    Set debug=True to receive a RuntimeWarning if forward() is called more
    than once during a single generation pass (indicates multi-pass behaviour).
    """
    def __init__(self, encoder_out, debug=False):
        super().__init__()
        self.encoder_out = encoder_out
        self.debug = debug
        self._call_count = 0

    def forward(self, input_values=None, attention_mask=None, return_dict=True, **kwargs):
        self._call_count += 1
        if self.debug and self._call_count > 1:
            import warnings
            warnings.warn(
                f"MockEncoder called {self._call_count} times — generation may be multi-pass.",
                RuntimeWarning,
                stacklevel=2,
            )
        return self.encoder_out

    @property
    def main_input_name(self):
        return "input_values"


class SpeechT5WavLMDataset(Dataset):
    """
    PyTorch Dataset for the hybrid WavLM → SpeechT5 architecture.

    Reads from a UNIFIED, pre-processed Arrow dataset (one row per aligned pair)
    with two columns:
        'input_values' : WavLM hidden states,  shape (Seq_Len, 768)  — source (EN)
        'labels'       : 80-bin mel-spectrogram, shape (T, 80)       — target (DE)

    This avoids maintaining two separate language directories and ensures that
    every (source, target) pair is always correctly aligned.
    """
    def __init__(self, paired_ds, processor, speaker_embeddings):
        """
        Args:
            paired_ds         : HuggingFace Dataset with columns ['input_values', 'labels']
            processor         : SpeechT5Processor (kept for potential fallback / tokenisation)
            speaker_embeddings: 1-D tensor of shape (512,) — fallback target-language x-vector
        """
        self.paired_ds = paired_ds
        self.processor = processor
        self.speaker_embeddings = speaker_embeddings

    def __len__(self):
        return len(self.paired_ds)

    def __getitem__(self, idx):
        row = self.paired_ds[int(idx)]

        # ------------------------------------------------------------------
        # SOURCE: WavLM hidden states  (Seq_Len, 768)
        # ------------------------------------------------------------------
        src_val = np.array(row["input_values"], dtype=np.float32)
        if src_val.ndim == 1:
            if src_val.size % 768 == 0:
                src_val = src_val.reshape(-1, 768)
            else:
                raise ValueError(f"[Dataset] WavLM source has unexpected size {src_val.size}")
        source_features = torch.tensor(src_val, dtype=torch.float32)

        # ------------------------------------------------------------------
        # TARGET: 80-bin log-mel spectrogram  (T, 80)
        # ------------------------------------------------------------------
        tgt_val = np.array(row["labels"], dtype=np.float32)
        if tgt_val.ndim == 1:
            if tgt_val.size % 80 == 0:
                tgt_val = tgt_val.reshape(-1, 80)
            else:
                raise ValueError(f"[Dataset] Target mel has unexpected size {tgt_val.size}")
        elif tgt_val.ndim == 3:
            tgt_val = tgt_val.squeeze(0)

        target_features = torch.tensor(tgt_val, dtype=torch.float32)

        if target_features.shape[0] == 80 and target_features.shape[1] != 80:
            target_features = target_features.transpose(0, 1)

        # ------------------------------------------------------------------
        # SPEAKER: 512-dim x-vector
        # ------------------------------------------------------------------
        if "speaker_embeddings" in row:
            spk_emb = torch.tensor(row["speaker_embeddings"], dtype=torch.float32)
        else:
            spk_emb = self.speaker_embeddings

        return {
            "input_values": source_features,
            "labels": target_features,
            "speaker_embeddings": spk_emb,
        }

    @staticmethod
    def validate_dataset(paired_ds, num_samples=50):
        """
        Spot-check up to `num_samples` rows from the dataset for shape correctness.
        Raises ValueError on the first malformed row found.

        Checks performed per row:
          - Source (input_values) is reshapeable to (Seq, 768) with no remainder.
          - Target (labels) has 80 mel bins (T may be any positive integer;
            trimming to be divisible by reduction_factor is the collate function's job).
          - Neither source nor target contain NaN or Inf values.
        """
        import random
        indices = random.sample(range(len(paired_ds)), min(num_samples, len(paired_ds)))

        for idx in indices:
            row = paired_ds[int(idx)]

            src = np.array(row["input_values"], dtype=np.float32)
            tgt = np.array(row["labels"],       dtype=np.float32)

            # Source must be reshapeable to (Seq, 768)
            if src.ndim == 1 and src.size % 768 != 0:
                raise ValueError(
                    f"Row {idx}: source size {src.size} is not divisible by 768."
                )
            if src.ndim == 2 and src.shape[1] != 768:
                raise ValueError(
                    f"Row {idx}: source shape {src.shape} — expected dim 1 == 768."
                )

            # Target must have 80 mel bins
            tgt_2d = tgt.reshape(-1, 80) if tgt.ndim == 1 else tgt
            if tgt_2d.ndim == 2 and tgt_2d.shape[1] != 80:
                raise ValueError(
                    f"Row {idx}: target shape {tgt_2d.shape} — expected 80 mel bins."
                )

            # No NaN or Inf in source or target
            if not np.isfinite(src).all():
                raise ValueError(f"Row {idx}: source contains NaN or Inf.")
            if not np.isfinite(tgt).all():
                raise ValueError(f"Row {idx}: target contains NaN or Inf.")

        print(f"[DatasetValidation] {len(indices)} rows checked — all OK.")


def wavlm_speecht5_collate_fn(batch, reduction_factor=2):
    input_values = [item["input_values"] for item in batch]
    # Trim labels to be divisible by reduction_factor
    labels = [
        item["labels"][:item["labels"].shape[0] - item["labels"].shape[0] % reduction_factor]
        if item["labels"].shape[0] % reduction_factor != 0
        else item["labels"]
        for item in batch
    ]
    speaker_embeddings = [item["speaker_embeddings"] for item in batch]

    input_values_padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100.0)
    speaker_embeddings_stacked = torch.stack(speaker_embeddings)

    # Attention mask for 2D WavLM hidden states: Time steps where entire hidden vector is 0 are padding.
    attention_mask = (input_values_padded.abs().sum(dim=-1) != 0).long()

    return input_values_padded, attention_mask, labels_padded, speaker_embeddings_stacked


class SpeechT5WavLM(torch.nn.Module):
    def __init__(self, wavlm_model_name="microsoft/wavlm-base-plus", speecht5_model_name="microsoft/speecht5_vc"):
        super().__init__()
        
        self.wavlm_model_name = wavlm_model_name
        self.GRAD_ACCUM_STEPS = 1
        print(f"Loading SpeechT5WavLM components (WavLM={wavlm_model_name})...")
        
        self.processor = SpeechT5Processor.from_pretrained(speecht5_model_name)
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained(speecht5_model_name)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model.to(self.device)
        self.vocoder.to(self.device)

        # CRITICAL OVERRIDE: High Resolution Setup (v6)
        self.model.config.num_mel_bins = 80
        self.vocoder.config.num_mel_bins = 80
        self.vocoder.config.hop_length = 256
        self.vocoder.config.win_length = 1024

        self.target_embeddings = None

        # Trainable Adapter/Projection layer to align WavLM space with SpeechT5
        self.wavlm_proj = Conv1DBridge(dim=768, kernel_size=5).to(self.device)
        
        # Speaker projection for injection into the encoder pre-net (Paper v2)
        self.encoder_spk_proj = nn.Linear(512, 768).to(self.device)

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _log_trainable_params(self):
        """Print a trainable-parameter audit across all major sub-modules."""
        sections = {
            "SpeechT5 encoder prenet":      self.model.speecht5.encoder.prenet
                if hasattr(self.model.speecht5.encoder, "prenet")
                else self.model.speecht5.encoder,
            "SpeechT5 transformer encoder": self.model.speecht5.encoder.wrapped_encoder
                if hasattr(self.model.speecht5.encoder, "wrapped_encoder")
                else self.model.speecht5.encoder,
            "SpeechT5 decoder":             self.model.speecht5.decoder,
            "SpeechT5 speech postnet":      self.model.speech_decoder_postnet,
            "Conv1DBridge (wavlm_proj)":     self.wavlm_proj,
            "encoder_spk_proj":             self.encoder_spk_proj,
        }
        print("\n--- Trainable Parameter Audit ---")
        for name, module in sections.items():
            total   = sum(p.numel() for p in module.parameters())
            trained = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"  {name:<42} {trained:>10,} / {total:>10,} params trainable")
        print("---------------------------------\n")

    @contextmanager
    def _mock_encoder_ctx(self, encoder_out, debug=False):
        """
        Context manager that temporarily replaces the SpeechT5 encoder with a
        MockEncoder wrapping pre-computed hidden states, then unconditionally
        restores the original encoder on exit (even under KeyboardInterrupt).

        Pass debug=True to receive a RuntimeWarning if the encoder is called
        more than once during generation (indicates unexpected multi-pass use).
        """
        original_encoder = self.model.speecht5.encoder
        self.model.speecht5.encoder = MockEncoder(encoder_out, debug=debug)
        try:
            yield
        finally:
            self.model.speecht5.encoder = original_encoder

    def get_speaker_embedding(self, target_lang):
        print("Initializing X-Vector classifier for embedding extraction...")
        spk_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb", 
            savedir="tmp_spkrec", 
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        
        print("Extracting target speaker embedding...")
        try:
            config_name = dataset_loader._get_fleurs_config(target_lang)
            ds_stream = load_dataset(
                "google/fleurs", 
                config_name,
                streaming=True, 
                trust_remote_code=True,
                split="train"
            )
            
            tgt_waveform = None
            for sample in ds_stream:
                if 'audio' in sample:
                    arr = sample['audio']['array']
                    if len(arr) > 0:
                        tgt_waveform = torch.tensor(arr).unsqueeze(0).to(spk_classifier.device)
                        break
            
            if tgt_waveform is None:
                 raise ValueError("Could not find target audio in stream.")

            with torch.no_grad():
                embeddings = spk_classifier.encode_batch(tgt_waveform)
                self.target_embeddings = torch.nn.functional.normalize(embeddings, dim=2).squeeze().cpu()
            print("Embedding extracted successfully.")

        except Exception as e:
            print(f"Warning: Could not stream raw audio for embedding ({e}). Using random embedding.")
            self.target_embeddings = torch.randn(512)

        del spk_classifier
        torch.cuda.empty_cache()
        gc.collect()


    def _encode_wavlm_states(self, hidden_states, speaker_embeddings, attention_mask=None):
        # 1. Pass through the Conv1D Bridge
        projected_states = self.wavlm_proj(hidden_states)

        # 2. Inject Speaker Embeddings (Paper Design)
        # speaker_embeddings: (Batch, 512) -> (Batch, 1, 768)
        spk_proj = self.encoder_spk_proj(speaker_embeddings).unsqueeze(1)
        projected_states = projected_states + spk_proj
        
        # 3. Access encoder components
        encoder = self.model.speecht5.encoder
        prenet = encoder.prenet if hasattr(encoder, "prenet") else encoder
        transformer_enc = encoder.wrapped_encoder if hasattr(encoder, "wrapped_encoder") else encoder
        
        # 4. Apply Positional Convolutional Embedding
        # projected_states shape: (Batch, Seq, Dim)
        projected_states = prenet.pos_conv_embed(projected_states)
        
        # 5. Apply LayerNorm
        projected_states = transformer_enc.layer_norm(projected_states)
        
        # 6. Pass through the Transformer stack
        encoder_outputs = transformer_enc(
            hidden_states=projected_states,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return encoder_outputs

    def run_inference(self, audio_array, sampling_rate, speaker_embedding=None,
                      threshold=0.5, minlenratio=0.0, maxlenratio=1.2):
        """
        Legacy wrapper for raw audio inference. 
        Extracts WavLM features and delegates to self.infer().
        """
        self.model.eval()
        self.wavlm_proj.eval()
        self.encoder_spk_proj.eval()

        # 1. Prepare speaker embedding
        if speaker_embedding is not None:
            emb = torch.tensor(speaker_embedding).to(self.device).unsqueeze(0)
        elif self.target_embeddings is not None:
            emb = self.target_embeddings.to(self.device).unsqueeze(0)
        else:
            emb = torch.randn((1, 512)).to(self.device)

        # 2. Extract WavLM features (Lazy load model if needed)
        if not hasattr(self, '_wavlm_proc'):
            print(f"Loading WavLM ({self.wavlm_model_name}) extractor for inference...")
            self._wavlm_proc  = Wav2Vec2FeatureExtractor.from_pretrained(self.wavlm_model_name)
            self._wavlm_model = WavLMModel.from_pretrained(self.wavlm_model_name).to(self.device).eval()
            for p in self._wavlm_model.parameters():
                p.requires_grad_(False)

        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        inputs = self._wavlm_proc(
            audio_array, sampling_rate=sampling_rate,
            return_tensors="pt", padding=False,
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            wavlm_out = self._wavlm_model(input_values)
            hidden_states = wavlm_out.last_hidden_state  # (1, Seq, 768)

        # 3. Delegate to infer()
        speech = self.infer(
            wavlm_hidden_states=hidden_states,
            speaker_embeddings=emb,
            threshold=threshold,
            minlenratio=minlenratio,
            maxlenratio=maxlenratio,
            return_spectrogram=False
        )

        return {'audio': {'array': speech, 'sampling_rate': 16000}}

    def infer(self, wavlm_hidden_states, speaker_embeddings, threshold=0.5, minlenratio=0.0, maxlenratio=1.2, return_spectrogram=False):
        """
        Perform autoregressive inference using pre-computed WavLM features.
        
        Args:
            wavlm_hidden_states: Tensor of shape (Seq, 768) or (1, Seq, 768)
            speaker_embeddings: Tensor of shape (512,) or (1, 512)
            threshold: Stop threshold for generation
            minlenratio: Minimum length ratio
            maxlenratio: Maximum length ratio
            return_spectrogram: Whether to return both audio and mel-spectrogram
        """
        self.model.eval()
        self.vocoder.eval()
        self.wavlm_proj.eval()
        self.encoder_spk_proj.eval()

        # Handle shapes
        if wavlm_hidden_states.ndim == 2:
            wavlm_hidden_states = wavlm_hidden_states.unsqueeze(0)
        if speaker_embeddings.ndim == 1:
            speaker_embeddings = speaker_embeddings.unsqueeze(0)

        wavlm_hidden_states = wavlm_hidden_states.to(self.device)
        speaker_embeddings = speaker_embeddings.to(self.device)

        # Attention mask: all 1s
        attention_mask = torch.ones(wavlm_hidden_states.shape[:2], dtype=torch.long, device=self.device)

        with torch.no_grad():
            encoder_out = self._encode_wavlm_states(wavlm_hidden_states, speaker_embeddings, attention_mask)

            # Dummy input matching the encoder sequence length
            dummy_input = torch.ones(
                (1, encoder_out.last_hidden_state.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )

            with self._mock_encoder_ctx(encoder_out):
                if return_spectrogram:
                    spectrogram = self.model.generate_speech(
                        dummy_input,
                        speaker_embeddings=speaker_embeddings,
                        attention_mask=attention_mask,
                        threshold=threshold,
                        minlenratio=minlenratio,
                        maxlenratio=maxlenratio,
                        vocoder=None,
                    )
                    speech = self.vocoder(spectrogram)
                else:
                    speech = self.model.generate_speech(
                        dummy_input,
                        speaker_embeddings=speaker_embeddings,
                        attention_mask=attention_mask,
                        threshold=threshold,
                        minlenratio=minlenratio,
                        maxlenratio=maxlenratio,
                        vocoder=self.vocoder,
                    )

            speech = speech.squeeze()

        if return_spectrogram:
            return speech.cpu().numpy(), spectrogram.squeeze().cpu().numpy()
        return speech.cpu().numpy()

    def _train_phase(self, paired_ds, epochs, learning_rate, batch_size, saving_checkpoint,
                     optimizer, scaler, scheduler, coarse_mode=False, global_step=0,
                     start_epoch=0, total_epochs_label=None, step_callback=None):
        """
        Internal method to handle a single training phase (Coarse or Fine).

        Coarse supervision is achieved by striding mel targets 2× along the
        time axis (labels[:, ::2, :]) — model.config.reduction_factor is never
        changed from its initialised value of 2, because the decoder prenet's
        Linear weight is frozen to shape (256, 80*2=160) at load time.
        """
        if epochs <= 0:
            return global_step

        phase_name = "COARSE" if coarse_mode else "FINE"
        print(f"\n>>> Starting {phase_name} Training Phase ({epochs} epochs)")
        if coarse_mode:
            print("[COARSE] Temporal striding active: mel targets downsampled 2x. config.reduction_factor stays at 2.")

        # 0. Ensure memory is clean and checkpointing is off
        torch.cuda.empty_cache()
        gc.collect()
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        
        # 1. Setup Dataset and DataLoader
        train_dataset = SpeechT5WavLMDataset(
            paired_ds,
            self.processor,
            self.target_embeddings
        )

        # Collate always uses reduction_factor=2 — the model config is never changed.
        collate_fn = partial(wavlm_speecht5_collate_fn, reduction_factor=2)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        # 2. Training state
        self.model.train()
        self.wavlm_proj.train()

        # 3. Epoch Loop
        try:
            for epoch_idx in range(epochs):
                actual_epoch = start_epoch + epoch_idx + 1
                display_total = total_epochs_label if total_epochs_label else epochs
                
                epoch_loss = 0.0
                num_batches = 0
                optimizer.zero_grad()

                pbar = tqdm(train_loader, desc=f"[{phase_name}] Epoch {actual_epoch}/{display_total}")

                for step, (input_values, attention_mask, labels, speaker_embeddings) in enumerate(pbar):
                    if step == 0 and epoch_idx == 0:
                        assert labels.shape[1] % 2 == 0, f"Labels length {labels.shape[1]} not divisible by 2"

                    global_step += 1
                    input_values       = input_values.to(self.device)
                    attention_mask     = attention_mask.to(self.device).long()
                    labels             = labels.to(self.device)
                    speaker_embeddings = speaker_embeddings.to(self.device)

                    # Coarse-phase temporal downsampling: rather than setting
                    # reduction_factor=4 in the model config (which breaks the
                    # decoder prenet's fixed Linear(160, ...) weight shape),
                    # we stride the mel targets 2x along the time axis to give
                    # a coarser supervision signal while keeping config.reduction_factor=2.
                    # The post-stride length must also be even so that
                    # shift_spectrograms_right (which uses reduction_factor=2) never
                    # receives an odd-length sequence and produces a 240-dim instead
                    # of 160-dim input to the prenet Linear.
                    if phase_name == "COARSE":
                        labels = labels[:, ::2, :]        # (B, T//2, 80)
                        if labels.shape[1] % 2 != 0:     # ensure even length
                            labels = labels[:, :-1, :]   # drop one frame if odd

                    # Mixed Precision Forward Pass
                    with torch.amp.autocast('cuda'):
                        # 1. Encode WavLM states
                        encoder_out = self._encode_wavlm_states(input_values, speaker_embeddings, attention_mask)

                        # 2. Native HF Forward Pass
                        # By passing `labels`, HF automatically handles shift_spectrograms_right, 
                        # generates the correct decoder_attention_mask, and calculates the exact Triple Loss.
                        outputs = self.model(
                            encoder_outputs=(encoder_out.last_hidden_state,),
                            attention_mask=attention_mask, # HF uses this correctly for cross-attention
                            speaker_embeddings=speaker_embeddings,
                            labels=labels,
                            use_cache=False,
                            return_dict=True,
                        )
                        
                        loss = outputs.loss
                        mel_after = outputs.spectrogram  # Extract the predicted mel for visualization callback

                    # Fallback: SpeechT5ForSpeechToSpeech may return None for loss
                    # (e.g. with non-default reduction_factor). Compute L1 mel reconstruction loss manually.
                    if loss is None:
                        if mel_after is not None:
                            T = min(mel_after.shape[1], labels.shape[1])
                            loss = torch.nn.functional.l1_loss(mel_after[:, :T, :], labels[:, :T, :])
                        else:
                            print(f"[Warning] Both outputs.loss and outputs.spectrogram are None at step {global_step}. Skipping batch.")
                            optimizer.zero_grad()
                            continue

                    # NaN/Inf loss guard — skip corrupted batches rather than crash
                    if not torch.isfinite(loss):
                        print(f"[Warning] Non-finite loss at step {global_step}: {loss.item()}. Skipping batch.")
                        optimizer.zero_grad()
                        continue

                    # Scaled Backward Pass
                    scaler.scale(loss / self.GRAD_ACCUM_STEPS).backward()

                    if (step + 1) % self.GRAD_ACCUM_STEPS == 0:
                        # Unscale gradients for clipping
                        scaler.unscale_(optimizer)

                        # Grad norm diagnostic — fires at step 1 and every ~2 epochs thereafter
                        steps_per_epoch_approx = max(1, len(train_loader))
                        if global_step == 1 or global_step % (steps_per_epoch_approx * 2) == 0:
                            for name, param in [("wavlm_proj", self.wavlm_proj), ("encoder_spk_proj", self.encoder_spk_proj)]:
                                grad_norm = sum(
                                    p.grad.norm().item() ** 2 for p in param.parameters() if p.grad is not None
                                ) ** 0.5
                                print(f"[GradCheck] {name} grad norm at step {global_step}: {grad_norm:.6f}")

                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.wavlm_proj.parameters(), max_norm=1.0)
                        torch.nn.utils.clip_grad_norm_(self.encoder_spk_proj.parameters(), max_norm=1.0)
                        
                        # Scaled Optimizer Step
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()

                    epoch_loss += loss.item()
                    num_batches += 1
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    
                    if step_callback is not None:
                        # Ensure we don't hold references to GPU tensors in the callback
                        step_callback(
                            step=global_step,
                            loss=loss.item(),
                            target_mel=labels.detach().cpu(),
                            pred_mel=mel_after.detach().cpu()
                        )
                    
                    # Aggressive Cleanup
                    del encoder_out, outputs, loss, mel_after
                    if (step + 1) % 10 == 0:
                        torch.cuda.empty_cache()

                avg_loss = epoch_loss / max(num_batches, 1)
                print(f"[{phase_name}] Epoch {actual_epoch} Avg Loss: {avg_loss:.4f}")
                
                # End of epoch cleanup
                torch.cuda.empty_cache()
                gc.collect()

                if actual_epoch % saving_checkpoint == 0:
                    self.save(f"checkpoint_epoch_{actual_epoch}")

            return global_step

        except KeyboardInterrupt:
            print(f"\nTraining interrupted during {phase_name} phase!")
            raise

    def fine_tune(self, preprocessed_path, epochs, learning_rate, batch_size, saving_checkpoint=5, coarse_epochs=0, step_callback=None):
        """
        Fine-tune the hybrid WavLM → SpeechT5 pipeline using a two-phase Coarse-to-Fine approach.
        """
        print("Starting WavLM+SpeechT5 fine-tuning (Hybrid Architecture).")

        # 1. Load the unified paired dataset
        print(f"Loading preprocessed data from {preprocessed_path}...")
        if os.path.exists(os.path.join(preprocessed_path, "dataset_info.json")):
            paired_ds = load_from_disk(preprocessed_path)
            target_lang = self._target_lang if hasattr(self, '_target_lang') else "de"
        else:
            print("Detected legacy v1 format — merging language directories...")
            sub_dirs = sorted([
                d for d in os.listdir(preprocessed_path)
                if os.path.isdir(os.path.join(preprocessed_path, d))
            ])
            if len(sub_dirs) < 2:
                raise ValueError("Expected at least two language directories in preprocessed path")
            source_lang, target_lang = sub_dirs[0], sub_dirs[1]
            source_ds = load_from_disk(os.path.join(preprocessed_path, source_lang))
            target_ds = load_from_disk(os.path.join(preprocessed_path, target_lang))
            labels_list = target_ds["audio"] if "audio" in target_ds.column_names else target_ds["labels"]
            paired_ds = source_ds.rename_column("audio", "input_values")
            paired_ds = paired_ds.add_column("labels", labels_list)

        print(f"Loaded {len(paired_ds)} aligned (source, target) pairs.")

        # 2. Speaker embedding
        if self.target_embeddings is None:
            self.get_speaker_embedding(target_lang)

        # 3. Freeze the encoder prenet (bypassed by WavLM injection anyway) and
        #    log the full trainable-parameter audit so the split is explicit.
        print("Freezing SpeechT5 encoder prenet (bypassed by WavLM injection).")
        for p in self.model.speecht5.encoder.prenet.parameters():
            p.requires_grad_(False)
        self._log_trainable_params()
        
        # 4. Initialize Projection (Residual Bypass Hack)
        # Ensure the bridge starts close to an identity mapping
        nn.init.dirac_(self.wavlm_proj.conv1.weight)
        nn.init.zeros_(self.wavlm_proj.conv1.bias)
        nn.init.zeros_(self.wavlm_proj.conv2.weight)
        nn.init.zeros_(self.wavlm_proj.conv2.bias)

        total_epochs = coarse_epochs + epochs

        # 5. Setup Optimizer and Scheduler (Hoisted to persist across phases)
        trainable_params = (
            list(filter(lambda p: p.requires_grad, self.model.parameters())) +
            list(self.wavlm_proj.parameters()) +
            list(self.encoder_spk_proj.parameters())
        )
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)
        scaler = torch.amp.GradScaler('cuda')  # Updated from deprecated torch.cuda.amp.GradScaler

        # Calculate total steps for Inverse Square Root Decay scheduler
        steps_per_epoch = (len(paired_ds) + batch_size - 1) // batch_size
        effective_steps_per_epoch = steps_per_epoch // self.GRAD_ACCUM_STEPS
        total_training_steps = total_epochs * effective_steps_per_epoch
        warmup_steps = int(total_training_steps * 0.10)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Inverse Square Root Decay after warmup
            return (warmup_steps ** 0.5) / (max(1, current_step) ** 0.5)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # 6. Validate dataset shapes before any training begins
        SpeechT5WavLMDataset.validate_dataset(paired_ds, num_samples=100)

        global_step = 0
        
        try:
            # PHASE A: Coarse Training
            if coarse_epochs > 0:
                global_step = self._train_phase(
                    paired_ds, coarse_epochs, learning_rate, batch_size, saving_checkpoint,
                    optimizer=optimizer, scaler=scaler, scheduler=scheduler, coarse_mode=True,
                    global_step=global_step, start_epoch=0, total_epochs_label=total_epochs,
                    step_callback=step_callback
                )
                print("\n>>> Phase A (Coarse) complete. Transitioning to Phase B (Fine)...")

            # PHASE B: Fine-grained Training
            self._train_phase(
                paired_ds, epochs, learning_rate, batch_size, saving_checkpoint,
                optimizer=optimizer, scaler=scaler, scheduler=scheduler, coarse_mode=False,
                global_step=global_step, start_epoch=coarse_epochs, total_epochs_label=total_epochs,
                step_callback=step_callback
            )

            print("\nTraining completed successfully.")


        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving current progress...")
            self.save("speecht5_wavlm_interrupted")
            print("Saved to 'speecht5_wavlm_interrupted'. Exiting safely.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nError during training: {e}")
            self.save("speecht5_wavlm_error_mid_train")
            print("Saved to 'speecht5_wavlm_error_mid_train'. Exiting safely.")

    def fine_tune_vocoder(self, preprocessed_path, epochs, learning_rate, batch_size):
        """
        Fine-tune ONLY the HiFi-GAN vocoder using adversarial training.
        The SpeechT5 transformer and WavLM backbone are explicitly frozen.
        """
        from models.SpeechT5WavLM.vocoder_trainer import VocoderTrainer
        
        print("Starting Vocoder Fine-Tuning...")

        # 1. Freeze transformer and WavLM
        print("Freezing SpeechT5 transformer and WavLM backbone...")
        for p in self.model.parameters():
            p.requires_grad_(False)
        
        if hasattr(self, '_wavlm_model'):
            for p in self._wavlm_model.parameters():
                p.requires_grad_(False)
        
        # 2. Delegate to Trainer
        trainer = VocoderTrainer(
            model=self.model,
            vocoder=self.vocoder,
            device=self.device,
            target_embeddings=self.target_embeddings
        )
        
        try:
            trainer.train(
                preprocessed_path, 
                epochs, 
                learning_rate, 
                batch_size,
                save_callback=lambda epoch: self.save(f"vocoder_checkpoint_epoch_{epoch}")
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving current progress...")
            self.save("speecht5_wavlm_vocoder_interrupted")
            print("Saved to 'speecht5_wavlm_vocoder_interrupted'. Exiting safely.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error during vocoder training: {e}")
            self.save("speecht5_wavlm_vocoder_error")
            print("Saved to 'speecht5_wavlm_vocoder_error'. Exiting safely.")

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        # Save model and config
        self.model.save_pretrained(os.path.join(path, "model"))
        self.processor.save_pretrained(os.path.join(path, "processor"))
        self.vocoder.save_pretrained(os.path.join(path, "vocoder"))

        # Save custom projection weights
        torch.save(self.wavlm_proj.state_dict(),        os.path.join(path, "wavlm_proj.pth"))
        torch.save(self.encoder_spk_proj.state_dict(),  os.path.join(path, "encoder_spk_proj.pth"))

        # Save speaker embeddings
        if self.target_embeddings is not None:
            np.save(os.path.join(path, "speaker_embedding.npy"), self.target_embeddings.numpy())

        # Save metadata
        metadata = {
            "wavlm_model_name": self.wavlm_model_name,
            "architecture": "Conv1DBridge_Residual_V1"
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)

    def load(self, path):
        model_path = os.path.join(path, "model")
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print("Loading PEFT/Lora adapter...")
            self.model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            print("Loading full model...")
            self.model = SpeechT5ForSpeechToSpeech.from_pretrained(model_path)
        
        proj_path = os.path.join(path, "wavlm_proj.pth")
        if os.path.exists(proj_path):
            print("Loading custom Conv1DBridge weights...")
            self.wavlm_proj.load_state_dict(torch.load(proj_path, map_location=self.device))

        spk_proj_path = os.path.join(path, "encoder_spk_proj.pth")
        if os.path.exists(spk_proj_path):
            print("Loading encoder_spk_proj weights...")
            self.encoder_spk_proj.load_state_dict(torch.load(spk_proj_path, map_location=self.device))

        self.model.to(self.device)
        self.wavlm_proj.to(self.device)
        self.encoder_spk_proj.to(self.device)
        self.model.eval()
        self.wavlm_proj.eval()
        self.encoder_spk_proj.eval()

        emb_path = os.path.join(path, "speaker_embedding.npy")
        if os.path.exists(emb_path):
            self.target_embeddings = torch.tensor(np.load(emb_path))
