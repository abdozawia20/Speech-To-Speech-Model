import torch
import torchaudio
import numpy as np
import os
from transformers import SpeechT5HifiGan
from datasets import load_from_disk
import soundfile as sf

def test_diagnostic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load a sample from your processed dataset (v4)
    ds_path = "datasets/processed_wavlm_en_de_v7"
    if not os.path.exists(ds_path):
        print(f"Dataset {ds_path} not found.")
        return
        
    ds = load_from_disk(ds_path)
    sample = ds[0]
    
    # Normalized mel from dataset
    mel_norm = torch.tensor(sample["labels"]).unsqueeze(0).to(device) # (1, T, 80)
    
    # 2. Denormalize using current logic
    # We used: (log_mel - (-5.0)) / 2.0
    mel_raw = mel_norm * 2.0 - 5.0
    
    # 3. Load ORIGINAL pre-trained vocoder
    print("Loading pre-trained SpeechT5 HiFi-GAN...")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    vocoder.eval()
    
    # 4. Try to generate audio
    print(f"Generating audio from sample 0 (Mel shape: {mel_raw.shape})...")
    with torch.no_grad():
        # HiFi-GAN expects (B, T, 80)
        audio = vocoder(mel_raw).cpu().squeeze().numpy()
    
    # 5. Save and check values
    print(f"Audio max: {np.abs(audio).max()}")
    print(f"Audio mean: {audio.mean()}")
    
    out_file = "diagnostic_pretrained_vocoder_2.wav"
    sf.write(out_file, audio, 16000)
    print(f"Saved to {out_file}. Please check if this audio is noise or speech.")

    # 6. Check Vocoder Config
    print("\nVocoder Config:")
    print(f"Mean: {vocoder.config.mean}")
    print(f"Scale: {vocoder.config.scale}")

if __name__ == "__main__":
    test_diagnostic()
