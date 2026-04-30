import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
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
            speaker_embeddings: 1-D tensor of shape (512,) — target-language x-vector
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
        # The preprocessor saved numpy arrays of shape (Seq_Len, 768).
        # They may arrive as a flat list or already as a 2-D numpy array.
        src_val = np.array(row["input_values"], dtype=np.float32)
        if src_val.ndim == 1:
            # Flattened during serialisation — restore (Seq_Len, 768)
            if src_val.size % 768 == 0:
                src_val = src_val.reshape(-1, 768)
            else:
                raise ValueError(
                    f"[Dataset] WavLM source has unexpected flat size {src_val.size}; "
                    "not divisible by 768."
                )
        source_features = torch.tensor(src_val, dtype=torch.float32)  # (Seq_Len, 768)

        # ------------------------------------------------------------------
        # TARGET: 80-bin log-mel spectrogram  (T, 80)
        # ------------------------------------------------------------------
        # SpeechT5Processor saved shape (T, 80); deserialisation may produce
        # (80, T) or a flat array — normalise to (T, 80) unconditionally.
        tgt_val = np.array(row["labels"], dtype=np.float32)
        if tgt_val.ndim == 1:
            if tgt_val.size % 80 == 0:
                tgt_val = tgt_val.reshape(-1, 80)  # (T, 80)
            else:
                raise ValueError(
                    f"[Dataset] Target mel has unexpected flat size {tgt_val.size}; "
                    "not divisible by 80."
                )
        elif tgt_val.ndim == 3:
            tgt_val = tgt_val.squeeze(0)           # (1, T, 80) → (T, 80)

        target_features = torch.tensor(tgt_val, dtype=torch.float32)  # (T, 80)

        # Ensure time axis is first (SpeechT5 convention)
        if target_features.shape[0] == 80 and target_features.shape[1] != 80:
            target_features = target_features.transpose(0, 1)         # (80, T) → (T, 80)

        # SpeechT5 requires T to be divisible by reduction_factor (default=2)
        if target_features.shape[0] % 2 != 0:
            target_features = target_features[:-1, :]                 # trim last frame

        return {
            "input_values": source_features,      # (Seq_Len, 768)
            "labels": target_features,            # (T, 80)
            "speaker_embeddings": self.speaker_embeddings,
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
        self.target_embeddings = None

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

    def _encode_wavlm_states(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> BaseModelOutput:
        # Normalize WavLM features (SpeechT5 expects normalized features here)
        hidden_states = torch.nn.functional.layer_norm(hidden_states, [hidden_states.shape[-1]])

        prenet = self.model.speecht5.encoder.prenet

        # Add positional convolutional embedding
        positional_conv_embedding = prenet.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + positional_conv_embedding

        # Add positional sinusoidal embedding
        if attention_mask is not None:
            padding_mask = attention_mask.ne(1).long()
        else:
            padding_mask = torch.zeros(hidden_states.shape[:2], dtype=torch.long, device=hidden_states.device)

        positional_sinusoidal_embeddings = prenet.pos_sinusoidal_embed(padding_mask)
        hidden_states = hidden_states + positional_sinusoidal_embeddings

        transformer_enc = self._get_speecht5_transformer_encoder()
        return transformer_enc(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            return_dict=True,
        )

    def run_inference(self, audio_array, sampling_rate, speaker_embedding=None,
                      threshold=0.5, minlenratio=0.0, maxlenratio=2.0):

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

            config = self.model.config
            num_mel = config.num_mel_bins
            rf = config.reduction_factor
            max_steps = int(maxlenratio * encoder_states.shape[1] / rf)
            min_steps = int(minlenratio * encoder_states.shape[1] / rf)

            output_seq = torch.zeros(1, 1, num_mel, device=self.device)
            past_key_values = None
            spectrogram = []

            for step in range(max_steps):
                dec_hidden = self.model.speecht5.decoder.prenet(output_seq, emb)
                dec_out = self.model.speecht5.decoder.wrapped_decoder(
                    hidden_states=dec_hidden[:, -1:],
                    attention_mask=None,
                    encoder_hidden_states=encoder_states,
                    encoder_attention_mask=None,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                last_out = dec_out.last_hidden_state.squeeze(1)
                past_key_values = dec_out.past_key_values

                spectrum = self.model.speech_decoder_postnet.feat_out(last_out)
                spectrum = spectrum.view(1, rf, num_mel)
                spectrogram.append(spectrum)

                new_frame = spectrum[:, -1:, :]
                output_seq = torch.cat((output_seq, new_frame), dim=1)

                prob = torch.sigmoid(self.model.speech_decoder_postnet.prob_out(last_out))
                if step >= min_steps and prob.max() > threshold:
                    break

            if not spectrogram:
                return {'audio': {'array': np.zeros(16000, dtype=np.float32), 'sampling_rate': 16000}}

            mel = torch.stack(spectrogram).transpose(0, 1).flatten(1, 2)
            mel = mel + self.model.speech_decoder_postnet.postnet(mel)
            speech = self.vocoder(mel).squeeze()

        return {'audio': {'array': speech.cpu().numpy(), 'sampling_rate': 16000}}

    def fine_tune(self, preprocessed_path, epochs, learning_rate, batch_size):
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
        GRAD_ACCUM_STEPS = 4

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
        self.model.train()
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
        )
        
        # Calculate total optimization steps (accounting for gradient accumulation)
        total_steps = len(train_loader) * epochs // GRAD_ACCUM_STEPS

        # Set warmup to 10% of total training steps
        warmup_steps = int(total_steps * 0.1)

        # Initialize the new scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )

        # L1 + MSE combined loss (element-wise, ignores padding frames)
        l1_criterion  = torch.nn.L1Loss(reduction="mean")
        mse_criterion = torch.nn.MSELoss(reduction="mean")

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

                    # Mask out padded target frames
                    pred_before_m = pred_before * valid_mask_expanded
                    pred_after_m  = pred_after  * valid_mask_expanded
                    labels_m      = labels.clone()
                    labels_m[labels == -100.0] = 0.0                        # zero-out padding

                    # L1 + MSE on both pre- and post-postnet predictions
                    loss_pre  = l1_criterion(pred_before_m, labels_m) + mse_criterion(pred_before_m, labels_m)
                    loss_post = l1_criterion(pred_after_m,  labels_m) + mse_criterion(pred_after_m,  labels_m)

                    # Stop-token binary cross-entropy
                    stop_targets = (~valid_mask).float()                     # 1 at padding = stop
                    loss_stop = torch.nn.functional.binary_cross_entropy_with_logits(
                        stop_logits[:, :T_tgt].squeeze(-1),
                        stop_targets,
                    )

                    loss = loss_pre + loss_post + 0.5 * loss_stop

                    (loss / GRAD_ACCUM_STEPS).backward()

                    if (step + 1) % GRAD_ACCUM_STEPS == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
                    if step % 50 == 0:
                        torch.cuda.empty_cache()

                avg_loss = epoch_loss / max(num_batches, 1)
                print(f"Epoch {epoch+1}/{epochs}  Avg Loss: {avg_loss:.4f}")
                if (epoch + 1) % 5 == 0:
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

    def save(self, path):
        self.model.save_pretrained(os.path.join(path, "model"))
        self.processor.save_pretrained(os.path.join(path, "processor"))
        self.vocoder.save_pretrained(os.path.join(path, "vocoder"))
        if self.target_embeddings is not None:
            np.save(os.path.join(path, "speaker_embedding.npy"), self.target_embeddings.numpy())

    def load(self, path):
        model_path = os.path.join(path, "model")
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            self.model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            self.model = SpeechT5ForSpeechToSpeech.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        emb_path = os.path.join(path, "speaker_embedding.npy")
        if os.path.exists(emb_path):
            self.target_embeddings = torch.tensor(np.load(emb_path))
