import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torchaudio

# Monkey-patch torchaudio & os.symlink for SpeechBrain compatibility
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

import shutil
# Fix WinError 1314 when downloading SpeechBrain models on Windows without Admin
if hasattr(os, "symlink"):
    os.symlink = lambda src, dst, *args, **kwargs: shutil.copy(src, dst)

import numpy as np
from transformers import (
    SpeechT5ForSpeechToSpeech, SpeechT5Processor,
    WavLMModel, Wav2Vec2FeatureExtractor
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
from speechbrain.inference.speaker import EncoderClassifier
import gc

class SpeechT5WavLMDataset(Dataset):
    def __init__(self, source_ds, target_ds, processor, speaker_embeddings, is_preprocessed=True):
        self.source_ds = source_ds
        self.target_ds = target_ds
        self.processor = processor
        self.speaker_embeddings = speaker_embeddings
        self.is_preprocessed = is_preprocessed

    def __len__(self):
        return len(self.source_ds)

    def __getitem__(self, idx):
        src_item = self.source_ds[int(idx)]
        tgt_item = self.target_ds[int(idx)]

        def get_val(item):
            val = item['audio']
            if isinstance(val, dict) and 'array' in val:
                return val['array']
            return val

        src_val = np.array(get_val(src_item), dtype=np.float32)
        tgt_val = np.array(get_val(tgt_item), dtype=np.float32)

        target_features = None
        source_features = None

        if self.is_preprocessed:
            # Source encoding - WavLM hidden states: (Seq_Len, 768)
            if src_val.ndim == 1:
                if src_val.size % 768 == 0:
                    src_val = src_val.reshape(-1, 768)
                else:
                    raise ValueError(
                        f"WavLM source array has unexpected shape: {src_val.shape}. "
                        "Expected (Seq_Len, 768) or flat multiple of 768."
                    )
            source_features = torch.tensor(src_val, dtype=torch.float32)

            # Target encoding - Expected Spectrogram (Time, 80)
            if tgt_val.ndim == 2:
                target_features = torch.tensor(tgt_val)
            elif tgt_val.ndim == 1 and tgt_val.size > 80 and tgt_val.size % 80 == 0:
                target_features = torch.tensor(tgt_val).view(-1, 80)
            elif tgt_val.ndim == 3:
                target_features = torch.tensor(tgt_val).squeeze()
            
            if target_features is not None:
                 if target_features.shape[-1] != 80:
                      target_features = None

        # Fallback (Slow Path)
        if target_features is None:
            if tgt_val.ndim > 1: tgt_val = tgt_val.flatten()

            try:
                features = self.processor.feature_extractor(
                    tgt_val, sampling_rate=16000, return_tensors="pt"
                ).input_values[0]
                if features.dim() == 1 and len(features) == len(tgt_val):
                     raise ValueError("Processor returned raw audio")
                target_features = features
            except:
                mel = librosa.feature.melspectrogram(y=tgt_val, sr=16000, n_fft=1024, hop_length=256, n_mels=80)
                log_mel = librosa.power_to_db(mel, ref=np.max)
                norm_mel = (log_mel + 40.0) / 20.0 
                target_features = torch.tensor(norm_mel.T, dtype=torch.float32)

        if source_features is None:
            raise ValueError("Raw fallback for WavLM source is not natively handled in Dataset. Please preprocess audio to WavLM vectors first.")

        if target_features.dim() == 2:
             if target_features.shape[0] == 80 and target_features.shape[1] != 80:
                  target_features = target_features.transpose(0, 1)

        # Enforce reduction_factor=2 dimension requirement
        if target_features.shape[0] % 2 != 0:
            target_features = target_features[:-1, :]

        return {
            "input_values": source_features,
            "labels": target_features,
            "speaker_embeddings": self.speaker_embeddings
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
        
        try:
            self.processor = SpeechT5Processor.from_pretrained(speecht5_model_name)
        except TypeError:
            print("Warning: SpeechT5Processor could not load the tokenizer. Bypassing as it is not needed for inference!")
            self.processor = None
            
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained(speecht5_model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model.to(self.device)
        
        from speechbrain.inference.vocoders import HIFIGAN
        print("Loading pre-trained German SpeechBrain Vocoder (padmalcom/tts-hifigan-german)...")
        # Load the German SpeechBrain vocoder. We don't call .to(device) because run_opts handles it.
        self.vocoder = HIFIGAN.from_hparams(
            source="padmalcom/tts-hifigan-german", 
            savedir="tmp_hifigan_de",
            run_opts={"device": str(self.device)}
        )

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
            print(f"Loading WavLM ({self.wavlm_model_name}) for inference...")
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
            mel = self.model.speech_decoder_postnet.postnet(mel)
            # Use SpeechBrain's decode_batch (mel shape: 1, Total_Frames, 80 -> 1, 80, Total_Frames)
            mel_transposed = mel.transpose(1, 2)
            speech = self.vocoder.decode_batch(mel_transposed).squeeze()

        return {'audio': {'array': speech.cpu().numpy(), 'sampling_rate': 16000}}

    def fine_tune(self, preprocessed_path, epochs, learning_rate, batch_size):
        from transformers.models.speecht5.modeling_speecht5 import shift_spectrograms_right, SpeechT5SpectrogramLoss

        print(f"Starting WavLM+SpeechT5 fine-tuning.")

        GRAD_ACCUM_STEPS = 8

        print(f"Loading preprocessed data from {preprocessed_path}...")
        # Since friend provides preprocessed data, we'll map them
        source_dirs = [d for d in os.listdir(preprocessed_path) if os.path.isdir(os.path.join(preprocessed_path, d))]
        if len(source_dirs) < 2:
             raise ValueError("Expected at least two language directories in preprocessed path")
        source_lang = source_dirs[0]
        target_lang = source_dirs[1]
        
        source_ds = load_from_disk(os.path.join(preprocessed_path, source_lang))
        target_ds = load_from_disk(os.path.join(preprocessed_path, target_lang))

        if self.target_embeddings is None:
            self.get_speaker_embedding(target_lang)

        # Freeze SpeechT5 feature encoder
        print("Freezing CNN Feature Encoder.")
        self.model.freeze_feature_encoder()

        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        train_dataset = SpeechT5WavLMDataset(
            source_ds,
            target_ds,
            self.processor,
            self.target_embeddings,
            is_preprocessed=True,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=wavlm_speecht5_collate_fn,
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
                    attention_mask    = attention_mask.to(self.device).long()
                    labels            = labels.to(self.device)
                    speaker_embeddings = speaker_embeddings.to(self.device)

                    decoder_input_values, decoder_attention_mask = shift_spectrograms_right(
                        labels, self.model.config.reduction_factor, None
                    )

                    encoder_out = self._encode_wavlm_states(input_values, attention_mask)
                    
                    outputs = self.model.speecht5(
                        encoder_outputs=encoder_out,
                        attention_mask=attention_mask,
                        decoder_input_values=decoder_input_values,
                        decoder_attention_mask=decoder_attention_mask,
                        speaker_embeddings=speaker_embeddings,
                        use_cache=False,
                        output_attentions=True,
                    )

                    outputs_before_postnet, outputs_after_postnet, logits = self.model.speech_decoder_postnet(outputs[0])

                    criterion = SpeechT5SpectrogramLoss(self.model.config)
                    loss = criterion(
                        attention_mask,
                        outputs_before_postnet,
                        outputs_after_postnet,
                        logits,
                        labels,
                        cross_attentions=outputs.cross_attentions
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
            print("\nTraining interrupted! Saving current progress to 'speecht5_wavlm_interrupted'...")
            self.save("speecht5_wavlm_interrupted")
            print("Progress saved. Exiting safely.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nAn error occurred: {e}")
            print("Saving current progress to 'speecht5_wavlm_error_mid_train'...")
            self.save("speecht5_wavlm_error_mid_train")
            print("Progress saved. Exiting safely.")

    def save(self, path):
        self.model.save_pretrained(os.path.join(path, "model"))
        self.processor.save_pretrained(os.path.join(path, "processor"))
        # self.vocoder.save_pretrained(os.path.join(path, "vocoder")) # Skipped: SpeechBrain vocoder is pre-trained and fixed
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
