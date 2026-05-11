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
    def __init__(self, paired_ds, processor, speaker_embeddings, coarse_mode=False):
        """
        Args:
            paired_ds         : HuggingFace Dataset with columns ['input_values', 'labels']
            processor         : SpeechT5Processor (kept for potential fallback / tokenisation)
            speaker_embeddings: 1-D tensor of shape (512,) — fallback target-language x-vector
            coarse_mode       : bool, whether to apply temporal average pooling (coarse-to-fine training)
        """
        self.paired_ds = paired_ds
        self.processor = processor
        self.speaker_embeddings = speaker_embeddings
        self.coarse_mode = coarse_mode

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
        # COARSE MODE: Temporal Average Pooling (Downsample by 2)
        # ------------------------------------------------------------------
        if self.coarse_mode:
            # For WavLM (Seq, 768) -> pooling along Seq dimension
            # AvgPool1d expects (Batch, Channels, Seq)
            source_features = source_features.transpose(0, 1).unsqueeze(0) # (1, 768, Seq)
            source_features = torch.nn.functional.avg_pool1d(source_features, kernel_size=2, stride=2)
            source_features = source_features.squeeze(0).transpose(0, 1) # (Seq/2, 768)

            # For Mel-spectrogram (T, 80) -> pooling along T dimension
            target_features = target_features.transpose(0, 1).unsqueeze(0) # (1, 80, T)
            target_features = torch.nn.functional.avg_pool1d(target_features, kernel_size=2, stride=2)
            target_features = target_features.squeeze(0).transpose(0, 1) # (T/2, 80)

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
        
        # 2. Access encoder components
        encoder = self.model.speecht5.encoder
        prenet = encoder.prenet if hasattr(encoder, "prenet") else encoder
        transformer_enc = encoder.wrapped_encoder if hasattr(encoder, "wrapped_encoder") else encoder
        
        # 3. Apply Positional Convolutional Embedding
        # projected_states shape: (Batch, Seq, Dim)
        projected_states = prenet.pos_conv_embed(projected_states)
        
        # 4. Apply LayerNorm
        projected_states = transformer_enc.layer_norm(projected_states)
        
        # 5. THE FIX: Pass through the Transformer stack
        # We need to call the actual transformer layers
        encoder_outputs = transformer_enc(
            hidden_states=projected_states,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return encoder_outputs

    def run_inference(self, audio_array, sampling_rate, speaker_embedding=None,
                      threshold=0.5, minlenratio=0.0, maxlenratio=1.2):

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
            original_encoder = self.model.speecht5.encoder
            
            class MockEncoder(torch.nn.Module):
                def __init__(self, encoder_out):
                    super().__init__()
                    self.encoder_out = encoder_out
                def forward(self, input_values, attention_mask=None, return_dict=True):
                    return self.encoder_out
                @property
                def main_input_name(self):
                    return "input_values"
                    
            self.model.speecht5.encoder = MockEncoder(encoder_out)
            
            # Use dummy_input matching the encoder sequence length directly.
            # Since we use a MockEncoder, the internal CNN subsampling is bypassed.
            dummy_input = torch.ones((1, encoder_states.shape[1]), dtype=torch.float32, device=self.device)
            
            try:
                speech = self.model.generate_speech(
                    dummy_input,
                    speaker_embeddings=emb,
                    attention_mask=attention_mask,
                    threshold=threshold,
                    minlenratio=minlenratio,
                    maxlenratio=maxlenratio,
                    vocoder=self.vocoder
                )
            finally:
                self.model.speecht5.encoder = original_encoder
                
            speech = speech.squeeze()

        return {'audio': {'array': speech.cpu().numpy(), 'sampling_rate': 16000}}

    def infer(self, wavlm_hidden_states, speaker_embeddings, threshold=0.5, minlenratio=0.0, maxlenratio=1.2, return_spectrogram=False):
        """
        Perform autoregressive inference using pre-computed WavLM features.
        
        Args:
            wavlm_hidden_states: Tensor of shape (Seq, 768) or (1, Seq, 768)
            speaker_embeddings: Tensor of shape (512,) or (1, 512)
        """
        self.model.eval()
        self.vocoder.eval()
        self.wavlm_proj.eval()

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
            encoder_out = self._encode_wavlm_states(wavlm_hidden_states, attention_mask)
            
            original_encoder = self.model.speecht5.encoder
            
            class MockEncoder(torch.nn.Module):
                def __init__(self, encoder_out):
                    super().__init__()
                    self.encoder_out = encoder_out
                def forward(self, input_values, attention_mask=None, return_dict=True):
                    return self.encoder_out
                @property
                def main_input_name(self):
                    return "input_values"
                    
            self.model.speecht5.encoder = MockEncoder(encoder_out)
            
            # Dummy input matching the encoder sequence length
            dummy_input = torch.ones((1, wavlm_hidden_states.shape[1]), dtype=torch.float32, device=self.device)
            
            try:
                if return_spectrogram:
                    spectrogram = self.model.generate_speech(
                        dummy_input,
                        speaker_embeddings=speaker_embeddings,
                        attention_mask=attention_mask,
                        threshold=threshold,
                        minlenratio=minlenratio,
                        maxlenratio=maxlenratio,
                        vocoder=None
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
                        vocoder=self.vocoder
                    )
            finally:
                self.model.speecht5.encoder = original_encoder
                
            speech = speech.squeeze()

        if return_spectrogram:
            return speech.cpu().numpy(), spectrogram.squeeze().cpu().numpy()
        return speech.cpu().numpy()

    def _train_phase(self, paired_ds, epochs, learning_rate, batch_size, saving_checkpoint, coarse_mode, start_epoch=0, total_epochs_label=None):
        """
        Internal method to handle a single training phase (Coarse or Fine).
        """
        if epochs <= 0:
            return

        phase_name = "COARSE" if coarse_mode else "FINE"
        print(f"\n>>> Starting {phase_name} Training Phase ({epochs} epochs)")

        # 1. Setup Dataset and DataLoader
        train_dataset = SpeechT5WavLMDataset(
            paired_ds,
            self.processor,
            self.target_embeddings,
            coarse_mode=coarse_mode
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=wavlm_speecht5_collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        # 2. Setup Optimizer and training state
        self.model.train()
        self.wavlm_proj.train()

        # --- THE OVERFIT HACK: DISABLE STOCHASTICITY ---
        for m in self.model.modules():
            if "Dropout" in m.__class__.__name__:
                m.eval()

        trainable_params = (
            list(filter(lambda p: p.requires_grad, self.model.parameters())) +
            list(self.wavlm_proj.parameters())
        )

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
        )

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
                    input_values       = input_values.to(self.device)
                    attention_mask     = attention_mask.to(self.device).long()
                    labels             = labels.to(self.device)
                    speaker_embeddings = speaker_embeddings.to(self.device)

                    # MODALITY BRIDGE: WavLM states → SpeechT5 transformer
                    encoder_out = self._encode_wavlm_states(input_values, attention_mask)

                    # NATIVE LOSS: Use SpeechT5's built-in loss calculation (includes stop token)
                    # NOTE: labels padding (-100) is handled correctly by MSELoss if we mask it,
                    # but HF's native loss expects it to be handled or ignored.
                    # For safety with MSE, we ensure no -100 remains in labels.
                    labels[labels == -100.0] = 0.0

                    outputs = self.model(
                        input_values=None,
                        speaker_embeddings=speaker_embeddings,
                        attention_mask=attention_mask,
                        labels=labels,
                        encoder_outputs=encoder_out,
                        return_dict=True,
                    )
                    
                    loss = outputs.loss
                    (loss / self.GRAD_ACCUM_STEPS).backward()

                    if (step + 1) % self.GRAD_ACCUM_STEPS == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        if hasattr(self, "wavlm_proj"):
                            torch.nn.utils.clip_grad_norm_(self.wavlm_proj.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                    epoch_loss += loss.item()
                    num_batches += 1
                    pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    
                    del encoder_out, outputs, loss

                avg_loss = epoch_loss / max(num_batches, 1)
                print(f"[{phase_name}] Epoch {actual_epoch} Avg Loss: {avg_loss:.4f}")
                
                if actual_epoch % saving_checkpoint == 0:
                    self.save(f"checkpoint_epoch_{actual_epoch}")

        except KeyboardInterrupt:
            print(f"\nTraining interrupted during {phase_name} phase!")
            raise

    def fine_tune(self, preprocessed_path, epochs, learning_rate, batch_size, saving_checkpoint=5, coarse_epochs=0):
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

        # 3. Freeze the SpeechT5 CNN feature encoder
        print("Freezing SpeechT5 CNN feature encoder.")
        self.model.freeze_feature_encoder()

        # 4. Initialize Projection (Residual Bypass Hack)
        # Ensure the bridge starts close to an identity mapping
        nn.init.dirac_(self.wavlm_proj.conv1.weight)
        nn.init.zeros_(self.wavlm_proj.conv1.bias)
        nn.init.zeros_(self.wavlm_proj.conv2.weight)
        nn.init.zeros_(self.wavlm_proj.conv2.bias)

        total_epochs = coarse_epochs + epochs
        
        try:
            # PHASE A: Coarse Training
            if coarse_epochs > 0:
                self._train_phase(paired_ds, coarse_epochs, learning_rate, batch_size, saving_checkpoint, 
                                  coarse_mode=True, start_epoch=0, total_epochs_label=total_epochs)
                print("\n>>> Phase A (Coarse) complete. Transitioning to Phase B (Fine)...")
            
            # PHASE B: Fine-grained Training
            self._train_phase(paired_ds, epochs, learning_rate, batch_size, saving_checkpoint, 
                              coarse_mode=False, start_epoch=coarse_epochs, total_epochs_label=total_epochs)

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
        
        # Save the custom Conv1DBridge state
        torch.save(self.wavlm_proj.state_dict(), os.path.join(path, "wavlm_proj.pth"))
        
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

        self.model.to(self.device)
        self.wavlm_proj.to(self.device)
        self.model.eval()
        self.wavlm_proj.eval()
        
        emb_path = os.path.join(path, "speaker_embedding.npy")
        if os.path.exists(emb_path):
            self.target_embeddings = torch.tensor(np.load(emb_path))
