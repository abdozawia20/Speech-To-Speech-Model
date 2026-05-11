import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
import numpy as np
from transformers import get_linear_schedule_with_warmup
from transformers import (
    SpeechT5ForSpeechToSpeech, SpeechT5Processor, SpeechT5HifiGan,
    WavLMModel, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
)
from transformers.modeling_outputs import BaseModelOutput
from datasets import load_from_disk, load_dataset
import dataset_loader
import librosa
import sys
import json
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import gc
from speechbrain.inference.speaker import EncoderClassifier


class Conv1DBridge(nn.Module):
    """
    Temporally aware bridge between WavLM (50Hz) and SpeechT5.
    Uses a sliding window (kernel_size=5) to give the attention mechanism
    surrounding acoustic context, smoothing out the alignment matrix.
    Incorporates a Residual Bypass to ensure 100% feature passthrough on Epoch 1.
    """
    def __init__(self, dim=768, kernel_size=5):
        super().__init__()
        # Padding ensures the sequence length remains exactly the same
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=padding)
        self.act1 = nn.GELU()
        self.norm1 = nn.BatchNorm1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=padding)
        self.act2 = nn.GELU()
        self.norm2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        # Conv1d expects shape (Batch, Channels, Sequence Length)
        # WavLM outputs shape (Batch, Sequence Length, Channels)
        x = x.transpose(1, 2)
        residual = x

        # Apply layers
        x = self.norm1(self.act1(self.conv1(x)))
        x = self.norm2(self.act2(self.conv2(x)))

        # Residual connection ensures features pass through even if layers are zeroed
        x = x + residual

        # Transpose back to SpeechT5 format
        return x.transpose(1, 2)


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

        if target_features.shape[0] % 2 != 0:
            target_features = target_features[:-1, :]

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


def wavlm_speecht5_collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]
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

    def _get_speecht5_transformer_encoder(self):
        encoder_obj = self.model.speecht5.encoder
        if hasattr(encoder_obj, "wrapped_encoder"):
            return encoder_obj.wrapped_encoder
        return encoder_obj

    def _encode_wavlm_states(self, hidden_states, attention_mask=None):
        # 1. Pass through the Conv1D Bridge
        projected_states = self.wavlm_proj(hidden_states)

        # ---------------------------------------------------------
        # 2. THE POSITIONAL ENCODING FIX
        # We must add SpeechT5's native temporal awareness back into the 
        # features so the cross-attention matrix knows left from right.
        # ---------------------------------------------------------
        encoder = self.model.speecht5.encoder
        
        # pos_conv_embed expects shape: (Batch, Channels, SeqLen)
        hidden_for_pos = projected_states.transpose(1, 2)
        pos_embeds = encoder.pos_conv_embed(hidden_for_pos)
        pos_embeds = pos_embeds.transpose(1, 2)
        
        # Add the temporal positional information
        projected_states = projected_states + pos_embeds
        
        # Apply the native LayerNorm to stabilize the variance for the Decoder
        projected_states = encoder.layer_norm(projected_states)
        # ---------------------------------------------------------

        # 3. Mock the BaseModelOutput for SpeechT5
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=projected_states,
            hidden_states=None,
            attentions=None
        )

    def run_inference(self, audio_array, sampling_rate, speaker_embedding=None,
                      threshold=0.5, minlenratio=0.0, maxlenratio=2.0):

        self.model.eval()
        self.vocoder.eval()
        self.wavlm_proj.eval()

        if speaker_embedding is not None:
            emb = torch.tensor(speaker_embedding).to(self.device).unsqueeze(0)
        elif self.target_embeddings is not None:
            emb = self.target_embeddings.to(self.device).unsqueeze(0)
        else:
            emb = torch.randn((1, 512)).to(self.device)

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

        # Attention mask: all 1s — single sample, no padding
        attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.long, device=self.device)

        with torch.no_grad():
            encoder_out = self._encode_wavlm_states(hidden_states, attention_mask)
            encoder_states = encoder_out.last_hidden_state

            # 5. THE FIX: Let Hugging Face handle the complex auto-regressive generation
            # Note: SpeechT5's generate_speech hardcodes the CNN encoder pass. 
            # We temporarily mock the CNN encoder to bypass it and return your WavLM states!
            original_encoder = self.model.speecht5.encoder
            
            class MockEncoder(torch.nn.Module):
                def forward(self, input_values, attention_mask, return_dict):
                    return encoder_out
                @property
                def main_input_name(self):
                    return "input_values"
                    
            self.model.speecht5.encoder = MockEncoder()
            
            # generate_speech requires an input_values tensor to determine the sequence length
            # for the attention mask. 
            # 1. THE BLIND MASK FIX: Use ones, not zeros! 
            # (pad_token_id is 0, so zeros would make the attention mask blind the model).
            dummy_input = torch.ones((1, encoder_states.shape[1]), dtype=torch.float32, device=self.device)
            
            # 2. THE DROPOUT RE-INJECTION
            # Force the prenet dropout to remain ACTIVE to break the deterministic loop
            for module in self.model.speecht5.decoder.prenet.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()

            try:
                speech = self.model.generate_speech(
                    dummy_input,
                    speaker_embeddings=emb,
                    threshold=threshold,
                    minlenratio=minlenratio,
                    maxlenratio=maxlenratio,
                    vocoder=self.vocoder
                )
            finally:
                self.model.speecht5.encoder = original_encoder
                
            speech = speech.squeeze()

        return {'audio': {'array': speech.cpu().numpy(), 'sampling_rate': 16000}}

    def fine_tune(self, preprocessed_path, epochs, learning_rate, batch_size, saving_checkpoint=5):
        """
        Fine-tune the hybrid WavLM → SpeechT5 pipeline.

        Architecture overview
        ---------------------
        1. INPUT  : WavLM hidden states  (B, Seq_Len, 768)  — pre-computed by preprocessor
        2. ENCODER: SpeechT5 transformer encoder receives the WavLM states directly,
                    bypassing its native CNN feature extractor entirely.
        3. DECODER: SpeechT5 autoregressive spectrogram decoder.
        4. LOSS   : L1 + MSE between predicted mel-spectrogram and target mel-spectrogram
                    (80-bin, computed by SpeechT5Processor during preprocessing).

        Why bypass the CNN?
        -------------------
        SpeechT5's CNN front-end was designed to process raw 16-kHz waveforms into
        a 768-dim frame sequence.  WavLM already does this (and does it better for
        cross-lingual tasks), so feeding WavLM states directly into the transformer
        encoder avoids a redundant and potentially lossy second convolution stage.

        Why L1 + MSE?
        -------------
        L1 encourages sharp spectral detail (robust to outliers), while MSE penalises
        large deviations strongly.  Their sum has been shown empirically to yield
        cleaner spectrograms than either alone for TTS / VC tasks.
        """
        from transformers.models.speecht5.modeling_speecht5 import shift_spectrograms_right

        print("Starting WavLM+SpeechT5 fine-tuning (Hybrid Architecture).")
        GRAD_ACCUM_STEPS = 1

        # ------------------------------------------------------------------
        # 1. Load the unified paired dataset
        # ------------------------------------------------------------------
        print(f"Loading preprocessed data from {preprocessed_path}...")
        # The new preprocessor saves a SINGLE dataset with columns
        # ['input_values', 'labels'] — no language sub-directories.
        if os.path.exists(os.path.join(preprocessed_path, "dataset_info.json")):
            # Unified paired dataset (v2 format)
            paired_ds = load_from_disk(preprocessed_path)
            target_lang = self._target_lang if hasattr(self, '_target_lang') else "de"
        else:
            # Legacy v1 format: two language sub-directories — convert on the fly
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
        print(f"Dataset columns: {paired_ds.column_names}")

        # ------------------------------------------------------------------
        # 2. Speaker embedding
        # ------------------------------------------------------------------
        if self.target_embeddings is None:
            self.get_speaker_embedding(target_lang)

        # ------------------------------------------------------------------
        # 3. Freeze the SpeechT5 CNN feature encoder
        #    (irrelevant for our pipeline but prevents accidental gradient
        #    flow if the model is ever called with raw waveforms elsewhere)
        # ------------------------------------------------------------------
        print("Freezing SpeechT5 CNN feature encoder (not used in hybrid path).")
        self.model.freeze_feature_encoder()

        # ------------------------------------------------------------------
        # 4. Build DataLoader
        # ------------------------------------------------------------------
        train_dataset = SpeechT5WavLMDataset(
            paired_ds,
            self.processor,
            self.target_embeddings,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=wavlm_speecht5_collate_fn,
            num_workers=2,     # keep low; dataset reads are cheap (pre-encoded)
            pin_memory=True,
        )

        # ------------------------------------------------------------------
        # 5. Optimiser
        # ------------------------------------------------------------------
        # 5. Optimiser
        self.model.train()
        self.wavlm_proj.train()

        # --- THE RESIDUAL BYPASS HACK ---
        # 1. Initialize Conv1 as Dirac (Identity)
        nn.init.dirac_(self.wavlm_proj.conv1.weight)
        nn.init.zeros_(self.wavlm_proj.conv1.bias)

        # 2. Initialize Conv2 as ZERO
        # This makes the entire bridge output exactly 0 + residual on Epoch 1.
        # This bypasses the GELU/BatchNorm distortion completely!
        nn.init.zeros_(self.wavlm_proj.conv2.weight)
        nn.init.zeros_(self.wavlm_proj.conv2.bias)

        # Disable dropout during the 1-sample overfit to establish base alignment
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

        # Ensure the Conv1D Bridge parameters are included in the optimizer
        trainable_params = (
            list(filter(lambda p: p.requires_grad, self.model.parameters())) +
            list(self.wavlm_proj.parameters())
        )

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
        )

        # Calculate total optimization steps (accounting for gradient accumulation)
        total_steps = len(train_loader) * epochs // GRAD_ACCUM_STEPS

        # Set warmup to 0 for brute-force overfit testing
        warmup_steps = 0

        # Initialize the new scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )

        # L1 + MSE combined loss (element-wise, ignores padding frames)
        l1_criterion  = torch.nn.L1Loss(reduction="none")
        mse_criterion = torch.nn.MSELoss(reduction="none")

        # ------------------------------------------------------------------
        # 6. Training loop
        # ------------------------------------------------------------------
        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                optimizer.zero_grad()

                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

                for step, (input_values, attention_mask, labels, speaker_embeddings) in enumerate(pbar):
                    # Move all tensors to device
                    input_values       = input_values.to(self.device)        # (B, Seq_Len, 768)
                    attention_mask     = attention_mask.to(self.device).long() # (B, Seq_Len)
                    labels             = labels.to(self.device)              # (B, T, 80)
                    speaker_embeddings = speaker_embeddings.to(self.device)  # (B, 512)

                    # -------------------------------------------------------
                    # MODALITY BRIDGE: WavLM states → SpeechT5 transformer
                    #
                    # _encode_wavlm_states() feeds input_values directly into
                    # the SpeechT5 *transformer* encoder (self-attention stack),
                    # completely bypassing the CNN feature extractor that would
                    # normally process raw waveforms.  The output is a
                    # BaseModelOutput with last_hidden_state shape (B, Seq, 768)
                    # that the SpeechT5 decoder cross-attends to.
                    # -------------------------------------------------------
                    encoder_out = self._encode_wavlm_states(input_values, attention_mask)

                    # Teacher-forced decoder input: shift target mel right by one frame
                    decoder_input_values, decoder_attention_mask = shift_spectrograms_right(
                        labels, self.model.config.reduction_factor, None
                    )

                    # Full SpeechT5 model forward pass with pre-computed encoder states
                    # 'encoder_outputs' tells SpeechT5 to skip its own encoder and use
                    # our WavLM-derived encoder_out directly.
                    outputs = self.model.speecht5(
                        encoder_outputs=encoder_out,          # skip CNN + transformer encoder
                        attention_mask=attention_mask,
                        decoder_input_values=decoder_input_values,
                        decoder_attention_mask=decoder_attention_mask,
                        speaker_embeddings=speaker_embeddings,
                        use_cache=False,
                        return_dict=True,
                    )

                    # Postnet: projects decoder hidden states → mel-spectrogram frames
                    # Returns (pre_postnet_mel, post_postnet_mel, stop_token_logits)
                    outputs_before_postnet, outputs_after_postnet, stop_logits = (
                        self.model.speech_decoder_postnet(outputs.last_hidden_state)
                    )

                    # Build a valid-frame mask from the mel padding sentinel (-100)
                    # so padding frames don't contribute to the loss.
                    valid_mask = (labels != -100.0).any(dim=-1)              # (B, T)
                    valid_mask_expanded = valid_mask.unsqueeze(-1).float()   # (B, T, 1)

                    # Trim predicted sequences to the length of the target
                    T_tgt = labels.shape[1]
                    pred_before = outputs_before_postnet[:, :T_tgt, :]      # (B, T, 80)
                    pred_after  = outputs_after_postnet[:, :T_tgt, :]       # (B, T, 80)

                    # 1. Calculate un-reduced, element-wise losses (masking out padding)
                    # We compute the raw loss first, then multiply by the valid_mask_expanded (0 for padding)
                    l1_pre_raw  = l1_criterion(pred_before, labels) * valid_mask_expanded
                    mse_pre_raw = mse_criterion(pred_before, labels) * valid_mask_expanded
                    l1_post_raw = l1_criterion(pred_after, labels) * valid_mask_expanded
                    mse_post_raw= mse_criterion(pred_after, labels) * valid_mask_expanded

                    # 2. Count exact number of valid scalar predictions
                    # valid_mask_expanded is (B, T, 1). Summing it gives total valid timeframes.
                    # Multiply by 80 (num_mel_bins) to get the true mathematical denominator.
                    num_valid_elements = valid_mask_expanded.sum() * 80.0
                    
                    # Prevent division by zero in edge cases
                    num_valid_elements = torch.clamp(num_valid_elements, min=1.0)

                    # 3. Sum the loss and calculate the TRUE mean over valid speech
                    loss_pre  = (l1_pre_raw.sum() + mse_pre_raw.sum()) / num_valid_elements
                    loss_post = (l1_post_raw.sum() + mse_post_raw.sum()) / num_valid_elements

                    # Stop-token binary cross-entropy
                    stop_targets = (~valid_mask).float()                     # 1 at padding = stop
                    loss_stop = torch.nn.functional.binary_cross_entropy_with_logits(
                        stop_logits[:, :T_tgt].squeeze(-1),
                        stop_targets,
                    )

                    loss = loss_pre + loss_post + 0.5 * loss_stop

                    (loss / GRAD_ACCUM_STEPS).backward()

                    if (step + 1) % GRAD_ACCUM_STEPS == 0:
                        # 1. Clip the main model
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        
                        # 2. CRITICAL: Clip your custom projection layer too!
                        if hasattr(self, 'wavlm_proj'):
                            torch.nn.utils.clip_grad_norm_(self.wavlm_proj.parameters(), max_norm=1.0)
                        optimizer.step()
                        scheduler.step() # <--- ADDED HERE
                        optimizer.zero_grad()

                    current_loss = loss.item()
                    epoch_loss  += current_loss
                    num_batches += 1
                    pbar.set_postfix({"loss": f"{current_loss:.4f}",
                                      "l1_pre": f"{loss_pre.item():.4f}",
                                      "l1_post": f"{loss_post.item():.4f}"})

                    # Free intermediate tensors explicitly each step
                    del encoder_out, outputs, pred_before, pred_after, loss

                avg_loss = epoch_loss / max(num_batches, 1)
                print(f"Epoch {epoch+1}/{epochs}  Avg Loss: {avg_loss:.4f}")
                if (epoch + 1) % saving_checkpoint == 0:
                    self.save(f"checkpoint_epoch_{epoch+1}")

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
        self.model.save_pretrained(os.path.join(path, "model"))
        self.processor.save_pretrained(os.path.join(path, "processor"))
        self.vocoder.save_pretrained(os.path.join(path, "vocoder"))
        torch.save(self.wavlm_proj.state_dict(), os.path.join(path, "wavlm_proj.pth"))
        if self.target_embeddings is not None:
            np.save(os.path.join(path, "speaker_embedding.npy"), self.target_embeddings.numpy())

    def load(self, path):
        model_path = os.path.join(path, "model")
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            self.model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            self.model = SpeechT5ForSpeechToSpeech.from_pretrained(model_path)
        
        proj_path = os.path.join(path, "wavlm_proj.pth")
        if os.path.exists(proj_path):
            print("Loading WavLM projection layer...")
            self.wavlm_proj.load_state_dict(torch.load(proj_path, map_location=self.device))

        self.model.to(self.device)
        self.wavlm_proj.to(self.device)
        self.model.eval()
        self.wavlm_proj.eval()
        
        emb_path = os.path.join(path, "speaker_embedding.npy")
        if os.path.exists(emb_path):
            self.target_embeddings = torch.tensor(np.load(emb_path))
