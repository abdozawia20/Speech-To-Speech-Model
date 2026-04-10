import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import SpeechT5HifiGan
from datasets import load_dataset
import torchaudio
import numpy as np
from tqdm import tqdm

class VocoderDataset(Dataset):
    def __init__(self, dataset, max_length=16000 * 2):
        # max_length of 16000 * 2 = 2 seconds of audio at 16kHz
        self.dataset = dataset
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def extract_target_mel(self, waveform):
        """
        Extracts the mel-spectrogram numpy using librosa logic for the *input* features.
        But mathematically, it's faster to do this in the collate_fn using torchaudio.
        So we just return the raw waveform here.
        """
        return waveform

    def __getitem__(self, idx):
        item = self.dataset[idx]
        y = np.array(item['audio']['array'], dtype=np.float32)
        
        # Trim or pad to max_length for uniform batching
        if len(y) > self.max_length:
            start = np.random.randint(0, len(y) - self.max_length)
            y = y[start:start + self.max_length]
        else:
            y = np.pad(y, (0, self.max_length - len(y)))

        return {
            "waveform": torch.tensor(y, dtype=torch.float32)
        }

def collate_fn(batch):
    waveforms = torch.stack([item['waveform'] for item in batch])
    return waveforms

def main():
    print("1. Initializing SpeechT5 HiFi-GAN Vocoder Fine-Tuner...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load baseline model from Hugging Face
    model = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    model.train()

    # Load your dataset (Switch this to your actual pre-processed dataset later)
    # Using German FLEURS as an example here since it's for German speech
    print("2. Loading Dataset (google/fleurs - de_de)...")
    ds = load_dataset("google/fleurs", "de_de", split="train")

    dataset = VocoderDataset(ds)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # -------------------------------------------------------------
    # THE LOSS FUNCTION: Mel-Spectrogram Reconstruction Loss
    # Because Hugging Face HifiGan doesn't provide a Discriminator
    # network, we fine-tune by ensuring the *predicted audio* 
    # produces the exact same Mel-Spectrogram as the *real audio*.
    # -------------------------------------------------------------
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        power=2.0 # Power spectrum
    ).to(device)

    # Convert power spectrogram to Decibels (Log-Mel) safely within PyTorch
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    epochs = 10
    print("\n3. Starting Vocoder Fine-Tuning Sequence...")
    try:
        for epoch in range(epochs):
            epoch_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
            
            for target_waveforms in pbar:
                target_waveforms = target_waveforms.to(device)

                optimizer.zero_grad()

                # Step A: Get the ground truth Mel-Spectrograms (Input)
                with torch.no_grad():
                    real_mels = mel_transform(target_waveforms)
                    real_mels_db = amplitude_to_db(real_mels)
                    
                    # Normalize to [-1, 1] rough estimation for HifiGan input
                    real_mels_db = (real_mels_db + 40.0) / 20.0
                    
                    # Transpose to (Batch, Time, Features(80)) for HifiGan
                    input_mels = real_mels_db.transpose(1, 2)

                # Step B: Generate Fake Waveforms
                # forward() returns BaseModelOutput, we grab the waveform
                pred_waveforms = model(input_mels).waveform

                # Step C: Align the sequence lengths 
                # (Vocoder upsampling can sometimes be off by an exact few frames)
                min_len = min(pred_waveforms.shape[-1], target_waveforms.shape[-1])
                pred_wav_cropped = pred_waveforms[:, :min_len]
                target_wav_cropped = target_waveforms[:, :min_len]

                # Step D: Apply PyTorch Auto-Differentiation Loss
                # Instead of raw waveform loss, we compare their acoustic signatures
                pred_mels = mel_transform(pred_wav_cropped)
                target_mels = mel_transform(target_wav_cropped)

                pred_mels_db = amplitude_to_db(pred_mels)
                target_mels_db = amplitude_to_db(target_mels)

                # Use L1 Loss to minimize the difference in the spectrograms
                loss = nn.functional.l1_loss(pred_mels_db, target_mels_db)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"mel_loss": f"{loss.item():.4f}"})
            
            print(f"Epoch {epoch + 1} completed. Avg Loss: {epoch_loss / len(loader):.4f}")

            # Save checkpoint periodically
            save_path = f"finetuned_vocoder_epoch_{epoch+1}"
            model.save_pretrained(save_path)
            print(f"Checkpoint successfully saved to {save_path}")

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user. Saving fail-safe checkpoint...")
        model.save_pretrained("finetuned_vocoder_interrupted")
        print("Data saved safely.")

    print("\nVocoder Fine-Tuning Script Completed!")

if __name__ == "__main__":
    main()