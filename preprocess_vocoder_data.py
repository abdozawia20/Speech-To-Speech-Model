import os
import sys
import numpy as np
import torch
import torchaudio
from datasets import Audio
from transformers import SpeechT5FeatureExtractor

# Add project root to path so we can import dataset_loader
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
if project_root not in sys.path:
    sys.path.append(project_root)

import dataset_loader

# ── Mel-spectrogram settings (must exactly match train_vocoder.py) ──────────
# Loaded from the SpeechT5 feature extractor config to guarantee consistency.
FE          = SpeechT5FeatureExtractor.from_pretrained("microsoft/speecht5_vc")
N_MELS      = FE.num_mel_bins   # 80
HOP_LENGTH  = FE.hop_length     # 256
WIN_LENGTH  = FE.win_length     # 1024
N_FFT       = FE.n_fft          # 1024
SAMPLE_RATE = FE.sampling_rate  # 16000
FMIN        = FE.fmin           # 80
FMAX        = FE.fmax           # 7600
LOG_FLOOR   = 1e-5

print(f"Mel config → n_mels={N_MELS}, hop={HOP_LENGTH}, n_fft={N_FFT}, "
      f"fmin={FMIN}, fmax={FMAX}, sr={SAMPLE_RATE}")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP_LENGTH,
    f_min=FMIN,
    f_max=FMAX,
    n_mels=N_MELS,
    power=1.0,          # amplitude — consistent with SpeechT5
    norm="slaney",
    mel_scale="slaney",
)

def compute_log_mel(audio_array: np.ndarray) -> np.ndarray:
    """
    Convert a raw waveform (np.float32, 16 kHz) into a log-mel spectrogram.
    Returns an array of shape (frames, 80) — Time-first, ready for HiFi-GAN.
    """
    y = np.nan_to_num(audio_array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    wav = torch.tensor(y).unsqueeze(0)              # (1, T)
    mel = mel_transform(wav)                        # (1, 80, F)
    mel = torch.clamp(mel, min=LOG_FLOOR)
    log_mel = torch.log(mel)                        # natural log
    return log_mel.squeeze(0).T.numpy()             # (F, 80)  — Time-first


def precompute_mel_batch(batch):
    """
    HuggingFace datasets .map() function.
    Adds an 'input_mels' column (list of np.ndarray of shape (F, 80)).
    """
    mels = []
    for audio_item in batch["audio"]:
        audio_array = np.array(audio_item["array"], dtype=np.float32)
        mels.append(compute_log_mel(audio_array))
    return {"input_mels": mels}


def main():
    print("1. Loading seamless_align dataset (English-German pair)...")
    num_samples = 15000
    datasets = dataset_loader.load_data(
        dataset=['seamless_align'],
        lang=['en', 'de'],
        num_samples=num_samples
    )

    de_dataset = datasets['de']
    print(f"\nLoaded {len(de_dataset)} German audio samples.")

    print("2. Standardizing sampling rate to 16 kHz...")
    de_dataset = de_dataset.cast_column("audio", Audio(sampling_rate=16000))
    print("Features updated to 16 kHz:", de_dataset.features)

    print("3. Pre-computing log-mel spectrograms (shape: frames × 80)...")
    print("   This is the bottleneck that would otherwise run on every training batch.")
    de_dataset = de_dataset.map(
        precompute_mel_batch,
        batched=True,
        batch_size=64,
        desc="Computing log-mel spectrograms",
    )
    print(f"   Done. New features: {de_dataset.features}")

    print("4. Saving preprocessed dataset to disk...")
    output_dir = "preprocessed_vocoder_de"
    de_dataset.save_to_disk(output_dir)

    print(f"\nVocoder dataset saved to '{output_dir}'!")
    print("The dataset now contains a pre-computed 'input_mels' column.")
    print("Training will load these directly, skipping on-the-fly mel computation.")


if __name__ == "__main__":
    main()
