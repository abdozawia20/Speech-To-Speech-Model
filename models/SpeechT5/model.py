import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torchaudio
import numpy as np
from transformers import SpeechT5ForSpeechToSpeech, SpeechT5Processor, SpeechT5HifiGan
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

# Supported encoder types.
# 'default'  : source data is a raw normalised waveform  (1-D, float32)
#              → fed directly to SpeechT5's CNN feature extractor.
# 'wav2vec'  : source data is Wav2Vec2 hidden states     (Seq_Len, 768)
#              → bypasses the CNN prenet and is injected into the
#                SpeechT5 transformer encoder layers directly.
SUPPORTED_ENCODER_TYPES = ('default', 'wav2vec')

# Monkey-patch torchaudio for SpeechBrain compatibility
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from speechbrain.inference.speaker import EncoderClassifier

class SpeechT5Dataset(Dataset):
    def __init__(self, source_ds, target_ds, processor, speaker_embeddings,
                 is_preprocessed=True, encoder_type='default'):
        if encoder_type not in SUPPORTED_ENCODER_TYPES:
            raise ValueError(f"encoder_type must be one of {SUPPORTED_ENCODER_TYPES}, got '{encoder_type}'")
        self.source_ds = source_ds
        self.target_ds = target_ds
        self.processor = processor
        self.speaker_embeddings = speaker_embeddings
        self.is_preprocessed = is_preprocessed
        self.encoder_type = encoder_type

    def __len__(self):
        return len(self.source_ds)

    def __getitem__(self, idx):
        src_item = self.source_ds[int(idx)]
        tgt_item = self.target_ds[int(idx)]

        # Helper to extract array from dataset item
        def get_val(item):
            val = item['audio']
            if isinstance(val, dict) and 'array' in val:
                return val['array']
            return val

        # 1. Load Data
        src_val = np.array(get_val(src_item), dtype=np.float32)
        tgt_val = np.array(get_val(tgt_item), dtype=np.float32)

        # 2. Fast Path: Preprocessed Data
        # We need to be careful. Even if is_preprocessed=True, the disk data might be raw audio 
        # if the preprocessing script saved it that way or if it's mixed.
        target_features = None
        source_features = None

        if self.is_preprocessed:
            # -------------------------------------------------------------- #
            # Source encoding                                                 #
            # -------------------------------------------------------------- #
            if self.encoder_type == 'wav2vec':
                # src_val is Wav2Vec2 hidden states: (Seq_Len, 768).
                # DO NOT flatten — the 2-D shape is intentional.
                if src_val.ndim == 1:
                    # Defensive: reshape if saved as flat (Seq_Len * 768,)
                    if src_val.size % 768 == 0:
                        src_val = src_val.reshape(-1, 768)
                    else:
                        raise ValueError(
                            f"wav2vec source array has unexpected shape: {src_val.shape}. "
                            "Expected (Seq_Len, 768) or flat multiple of 768."
                        )
                source_features = torch.tensor(src_val, dtype=torch.float32)  # (Seq_Len, 768)
            else:
                # Default: raw normalised waveform — keep 1-D.
                if src_val.ndim > 1:
                    src_val = src_val.flatten()
                source_features = torch.tensor(src_val, dtype=torch.float32)

            # Target: Should be Spectrogram (Time, 80)
            # CHECK: Is it actually a spectrogram?
            # Case A: Already 2D
            if tgt_val.ndim == 2:
                target_features = torch.tensor(tgt_val)
            # Case B: Flattened Spectrogram (Divisible by 80)
            elif tgt_val.ndim == 1 and tgt_val.size > 80 and tgt_val.size % 80 == 0:
                # Heuristic: If it divides by 80, it's likely a flattened spec.
                target_features = torch.tensor(tgt_val).view(-1, 80)
            # Case C: 3D (1, Time, 80)
            elif tgt_val.ndim == 3:
                target_features = torch.tensor(tgt_val).squeeze()
            
            # If target_features is set, check if it makes sense (Time > 0)
            if target_features is not None:
                 if target_features.shape[-1] != 80:
                      # If last dim is not 80, our assumption was wrong (e.g. it was raw audio divisible by 80)
                      # Discard and fallback
                      target_features = None

        # 3. Fallback: Raw Audio Processing (Slow Path)
        if target_features is None:
            if tgt_val.ndim > 1: tgt_val = tgt_val.flatten()

        if source_features is None:
            if self.encoder_type == 'wav2vec':
                raise ValueError("cannot use raw fallback for wav2vec source")
            if src_val.ndim > 1: src_val = src_val.flatten()
            source_features = torch.tensor(src_val, dtype=torch.float32)

        if target_features is None:
            # Generate Target Spectrogram
            try:
                # Force feature extractor
                features = self.processor.feature_extractor(
                    tgt_val, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).input_values[0]
                
                # Validation: Did it return raw audio?
                if features.dim() == 1 and len(features) == len(tgt_val):
                     raise ValueError("Processor returned raw audio")
                
                target_features = features
            except:
                # Librosa fallback
                mel = librosa.feature.melspectrogram(y=tgt_val, sr=16000, n_fft=1024, hop_length=256, n_mels=80)
                log_mel = librosa.power_to_db(mel, ref=np.max)
                norm_mel = (log_mel + 40.0) / 20.0 
                target_features = torch.tensor(norm_mel.T, dtype=torch.float32)

        # 4. Final Shape Validations
        # Ensure (Time, 80)
        if target_features.dim() == 2:
             if target_features.shape[0] == 80 and target_features.shape[1] != 80:
                  target_features = target_features.transpose(0, 1)
        elif target_features.dim() == 1:
             # Last ditch effort for 1D tensors that slipped through
             if target_features.shape[0] % 80 == 0:
                  target_features = target_features.view(-1, 80)
             else:
                  # Recalculate using librosa if we have the original val
                  mel = librosa.feature.melspectrogram(y=tgt_val, sr=16000, n_fft=1024, hop_length=256, n_mels=80)
                  log_mel = librosa.power_to_db(mel, ref=np.max)
                  norm_mel = (log_mel + 40.0) / 20.0 
                  target_features = torch.tensor(norm_mel.T, dtype=torch.float32)

        return {
            "input_values": source_features,
            "labels": target_features,
            "speaker_embeddings": self.speaker_embeddings
        }

def speecht5_collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [item["labels"] for item in batch]
    speaker_embeddings = [item["speaker_embeddings"] for item in batch]

    # pad_sequence works for both 1-D (raw audio) and 2-D (Wav2Vec hidden states)
    # because it always pads along dim-0 (the time axis).
    input_values_padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    speaker_embeddings_stacked = torch.stack(speaker_embeddings)

    # Attention mask:
    #   • 1-D padded  → shape (Batch, T)         → non-zero positions
    #   • 2-D padded  → shape (Batch, T, Hidden)  → time steps where the
    #                                               entire hidden vector is zero
    #                                               are padding.
    if input_values_padded.dim() == 3:          # Wav2Vec hidden states
        attention_mask = (input_values_padded.abs().sum(dim=-1) != 0).long()  # (Batch, T)
    else:                                        # Raw audio waveform
        attention_mask = (input_values_padded != 0).long()                    # (Batch, T)

    return input_values_padded, attention_mask, labels_padded, speaker_embeddings_stacked

class SpeechT5(torch.nn.Module):
    def __init__(self, encoder_type: str = 'default'):
        """
        Args:
            encoder_type: Controls how *source* audio is encoded during fine-tuning.
                'default'  – raw normalised waveform fed through SpeechT5's own
                             CNN feature extractor (original behaviour).
                'wav2vec'  – pre-computed Wav2Vec2 hidden states (Seq_Len, 768)
                             loaded from disk; the CNN prenet is bypassed and the
                             states are injected directly into the transformer
                             encoder layers of SpeechT5.
        """
        super().__init__()
        if encoder_type not in SUPPORTED_ENCODER_TYPES:
            raise ValueError(f"encoder_type must be one of {SUPPORTED_ENCODER_TYPES}, got '{encoder_type}'")
        self.encoder_type = encoder_type

        print(f"Loading SpeechT5 components (encoder_type='{encoder_type}')...")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model.to(self.device)
        self.vocoder.to(self.device)
        self.target_embeddings = None

    def get_speaker_embedding(self, target_lang):
        """
        Smart embedding extractor. 
        If data is preprocessed, it streams ONE raw sample from HF to get the voice.
        """
        print("Initializing X-Vector classifier for embedding extraction...")
        # Load locally to avoid constant HF fetching if possible
        spk_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb", 
            savedir="tmp_spkrec", 
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        
        print("Extracting target speaker embedding...")
        try:
            print(f"Streaming 1 sample from google/fleurs for {target_lang}...")
            
            # Use helper to get config (e.g. 'de' -> 'de_de')
            config_name = dataset_loader._get_fleurs_config(target_lang)
            
            ds_stream = load_dataset(
                "google/fleurs", 
                config_name,
                streaming=True, 
                trust_remote_code=True,
                split="train"
            )
            
            tgt_waveform = None
            
            # Iterate a few samples to find valid audio
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

        # VRAM CLEANUP
        print("Cleaning up speaker classifier to free VRAM...")
        del spk_classifier
        torch.cuda.empty_cache()
        gc.collect()

    def run_inference(self, audio_array, sampling_rate, speaker_embedding=None, threshold=0.5, minlenratio=0.0, maxlenratio=2.0):

        inputs = self.processor(audio=audio_array, sampling_rate=sampling_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if speaker_embedding is not None:
            emb = torch.tensor(speaker_embedding).to(self.device).unsqueeze(0)
        elif self.target_embeddings is not None:
            emb = self.target_embeddings.to(self.device).unsqueeze(0)
        else:
            emb = torch.randn((1, 512)).to(self.device)

        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_values"], 
                emb, 
                vocoder=self.vocoder, 
                threshold=threshold,
                minlenratio=minlenratio,
                maxlenratio=maxlenratio
            )

        return {'audio': {'array': speech.cpu().numpy(), 'sampling_rate': 16000}}

    # ---------------------------------------------------------------------- #
    # Internal helpers for the wav2vec encoder bypass                        #
    # ---------------------------------------------------------------------- #

    def _get_speecht5_transformer_encoder(self):
        """Return the transformer (non-CNN) part of the SpeechT5 encoder."""
        encoder_obj = self.model.speecht5.encoder
        if hasattr(encoder_obj, "wrapped_encoder"):
            return encoder_obj.wrapped_encoder
        return encoder_obj

    def _encode_wav2vec_states(self, hidden_states: torch.Tensor,
                               attention_mask: torch.Tensor) -> BaseModelOutput:
        """
        Run Wav2Vec2 hidden states through the SpeechT5 transformer encoder
        layers (skipping the CNN / prenet), and return a BaseModelOutput
        that can be passed directly as ``encoder_outputs`` to
        ``SpeechT5ForSpeechToSpeech.forward()``.

        Args:
            hidden_states:  (Batch, Seq_Len, 768) — Wav2Vec2 last hidden states.
            attention_mask: (Batch, Seq_Len)       — 1 for real tokens, 0 for pad.

        Returns:
            BaseModelOutput whose ``last_hidden_state`` is (Batch, Seq_Len, 768).
        """
        transformer_enc = self._get_speecht5_transformer_encoder()

        # Convert the 0/1 attention mask to the extended float mask that
        # transformer layers expect: 0.0 for real tokens, -inf for padding.
        extended_mask = None
        if attention_mask is not None:
            bsz, seq_len = attention_mask.shape
            expanded_mask = attention_mask[:, None, None, :].expand(bsz, 1, seq_len, seq_len).float()
            extended_mask = (1.0 - expanded_mask) * torch.finfo(hidden_states.dtype).min

        states = hidden_states
        for layer in transformer_enc.layers:
            layer_out = layer(
                states,
                attention_mask=extended_mask,
                output_attentions=False,
            )
            states = layer_out[0]

        if hasattr(transformer_enc, "layer_norm") and transformer_enc.layer_norm is not None:
            states = transformer_enc.layer_norm(states)

        return BaseModelOutput(last_hidden_state=states)

    # ---------------------------------------------------------------------- #

    def fine_tune(self, source_lang, target_lang, batch_size, epochs, learning_rate):
        print(f"Starting fine-tuning: {source_lang} -> {target_lang}  "
              f"[encoder_type='{self.encoder_type}']")

        # --- CONFIGURATION ---
        GRAD_ACCUM_STEPS = 8  # Simulate larger batch size

        # 1. Load Preprocessed Datasets
        # Each encoder_type has its own preprocessed dataset directory.
        if self.encoder_type == 'wav2vec':
            candidate_paths = [
                os.path.join(dataset_loader.DATASETS_DIR,
                             f"processed_speecht5_wav2vec_{source_lang}_{target_lang}_v1"),
            ]
        else:  # 'default' aka MelSpectrogram
            candidate_paths = [
                os.path.join(dataset_loader.DATASETS_DIR,
                             f"processed_speecht5_{source_lang}_{target_lang}_v2_cleaned"),
            ]

        preprocessed_path = next((p for p in candidate_paths if os.path.exists(p)), None)

        if preprocessed_path:
            print(f"Loading preprocessed data from {preprocessed_path}...")
            source_ds = load_from_disk(os.path.join(preprocessed_path, source_lang))
            target_ds = load_from_disk(os.path.join(preprocessed_path, target_lang))
            is_preprocessed_flag = True
        else:
            print("Preprocessed dataset not found. Proceeding with raw load (High VRAM warning).")
            datasets = dataset_loader.load_data(
                lang=[source_lang, target_lang], split="train", dataset=['seamless_align']
            )
            source_ds = datasets.get(source_lang)
            target_ds = datasets.get(target_lang)
            is_preprocessed_flag = False

        # 2. Get Speaker Embedding
        if self.target_embeddings is None:
            self.get_speaker_embedding(target_lang)

        # 3. Freeze the CNN feature encoder (only relevant for 'default' path,
        #    but harmless for 'wav2vec' since that path is bypassed entirely).
        print("Freezing Feature Encoder.")
        self.model.freeze_feature_encoder()

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_dataset = SpeechT5Dataset(
            source_ds,
            target_ds,
            self.processor,
            self.target_embeddings,
            is_preprocessed=is_preprocessed_flag,
            encoder_type=self.encoder_type,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=speecht5_collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                num_batches = 0
                optimizer.zero_grad()

                pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

                for step, (input_values, attention_mask, labels, speaker_embeddings) in enumerate(pbar):
                    input_values      = input_values.to(self.device)
                    attention_mask    = attention_mask.to(self.device)
                    labels            = labels.to(self.device)
                    speaker_embeddings = speaker_embeddings.to(self.device)

                    # -------------------------------------------------------- #
                    # Forward pass — behaviour differs by encoder_type          #
                    # -------------------------------------------------------- #
                    if self.encoder_type == 'wav2vec':
                        # input_values is (Batch, Seq_Len, 768) — Wav2Vec2 hidden
                        # states.  We bypass SpeechT5's CNN feature encoder and
                        # inject the states directly into its transformer layers.
                        encoder_out = self._encode_wav2vec_states(
                            input_values, attention_mask
                        )
                        outputs = self.model(
                            encoder_outputs=encoder_out,
                            attention_mask=attention_mask, # Needed for decoder cross-attention
                            speaker_embeddings=speaker_embeddings,
                            labels=labels,
                            use_cache=False,  # KV-cache is inference-only; new Cache API causes unpack errors during training
                        )
                    else:
                        # 'default': raw normalised waveform (Batch, T).
                        # SpeechT5's own CNN encoder processes it.
                        outputs = self.model(
                            input_values=input_values,
                            attention_mask=attention_mask,
                            speaker_embeddings=speaker_embeddings,
                            labels=labels,
                            use_cache=False,  # KV-cache is inference-only
                        )

                    loss = outputs.loss
                    if loss is None:
                        pred    = outputs.spectrogram
                        min_len = min(pred.size(1), labels.size(1))
                        loss    = torch.nn.functional.l1_loss(
                            pred[:, :min_len, :], labels[:, :min_len, :]
                        )

                    (loss / GRAD_ACCUM_STEPS).backward()

                    if (step + 1) % GRAD_ACCUM_STEPS == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                    current_loss = loss.item()
                    epoch_loss  += current_loss
                    num_batches += 1
                    pbar.set_postfix({"loss": f"{current_loss:.4f}"})

                scheduler.step()
                print(f"Epoch {epoch+1} Avg Loss: {epoch_loss / num_batches:.4f}")
                if (epoch + 1) % 5 == 0:
                    self.save(f"checkpoint_epoch_{epoch+1}")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving current progress to 'speecht5_interrupted'...")
            self.save("speecht5_interrupted")
            print("Progress saved. Exiting safely.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nAn error occurred: {e}")
            print("Saving current progress to 'speecht5_error_mid_train'...")
            self.save("speecht5_error_mid_train")
            print("Progress saved. Exiting safely.")

    def save(self, path):
        self.model.save_pretrained(os.path.join(path, "model"))
        self.processor.save_pretrained(os.path.join(path, "processor"))
        self.vocoder.save_pretrained(os.path.join(path, "vocoder"))
        if self.target_embeddings is not None:
            np.save(os.path.join(path, "speaker_embedding.npy"), self.target_embeddings.numpy())

    def load(self, path):
        model_path = os.path.join(path, "model")
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            print("Loading LoRA model...")
            self.model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
            self.model = PeftModel.from_pretrained(self.model, model_path)
        else:
            print("Loading base model...")
            self.model = SpeechT5ForSpeechToSpeech.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Try load embedding
        emb_path = os.path.join(path, "speaker_embedding.npy")
        if os.path.exists(emb_path):
            self.target_embeddings = torch.tensor(np.load(emb_path))