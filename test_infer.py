import torch
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Add the model directory to path
sys.path.append('models/SpeechT5WavLM')
from model import SpeechT5WavLM, SpeechT5WavLMDataset, wavlm_speecht5_collate_fn
from torch.utils.data import DataLoader
from datasets import load_from_disk

def test():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Initialize the model wrapper
    model = SpeechT5WavLM()
    model.device = device
    model.model.to(device)
    model.vocoder.to(device)
    model.wavlm_proj.to(device)
    
    # Try to load the best available checkpoint
    checkpoint_path = "models/SpeechT5WavLM/checkpoint_epoch_5"
    if not os.path.exists(checkpoint_path):
        checkpoints = sorted([d for d in os.listdir('models/SpeechT5WavLM') if d.startswith('checkpoint_')], 
                             key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0,
                             reverse=True)
        if checkpoints:
            checkpoint_path = os.path.join('models/SpeechT5WavLM', checkpoints[0])
        else:
            checkpoint_path = None

    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load(checkpoint_path)
    
    model.eval()

    # Load data from the overfit sample
    preprocessed_path = "models/SpeechT5WavLM/temp_overfit_dataset"
    if not os.path.exists(preprocessed_path):
         print(f"CRITICAL: {preprocessed_path} not found. Please ensure the dataset exists.")
         return

    paired_ds = load_from_disk(preprocessed_path)
    dataset = SpeechT5WavLMDataset(paired_ds, model.processor, torch.randn(512))
    loader = DataLoader(dataset, batch_size=1, collate_fn=wavlm_speecht5_collate_fn)
    
    # Get a single batch
    batch = next(iter(loader))
    input_values, _, labels, speaker_embeddings = batch
    
    wavlm_features = input_values[0]
    speaker_emb = speaker_embeddings[0]
    gt_mel = labels[0].cpu().numpy() # Shape: (T, 80)
    
    # Remove padding from ground truth (-100 is the padding value)
    mask = (gt_mel != -100).any(axis=1)
    gt_mel = gt_mel[mask]

    print(f"Input WavLM features shape: {wavlm_features.shape}")
    print(f"Ground Truth Mel shape: {gt_mel.shape}")
    
    print("Running autoregressive inference...")
    with torch.no_grad():
        audio = model.infer(wavlm_features, speaker_emb)
    
    print(f"Generated audio shape: {audio.shape}")
    
    # Compute mel-spectrogram for the generated audio to compare with ground truth
    # We use parameters matching SpeechT5's mel extraction
    gen_mel = librosa.feature.melspectrogram(
        y=audio, 
        sr=16000, 
        n_fft=1024, 
        hop_length=256, 
        win_length=1024, 
        n_mels=80,
        fmin=0,
        fmax=8000
    )
    gen_mel_db = librosa.power_to_db(gen_mel, ref=np.max)
    
    # Create the visualization
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    
    # Subplot 1: Ground Truth
    # Note: gt_mel in our dataset is already log-scaled mel (from SpeechT5Processor)
    img1 = librosa.display.specshow(gt_mel.T, ax=ax[0], x_axis='time', y_axis='mel', sr=16000, fmax=8000)
    ax[0].set_title('Ground Truth Mel Spectrogram (from dataset)')
    fig.colorbar(img1, ax=ax[0], format='%+2.0f dB')
    
    # Subplot 2: Generated
    img2 = librosa.display.specshow(gen_mel_db, ax=ax[1], x_axis='time', y_axis='mel', sr=16000, fmax=8000)
    ax[1].set_title('Generated Audio Mel Spectrogram (Reconstructed)')
    fig.colorbar(img2, ax=ax[1], format='%+2.0f dB')
    
    plt.tight_layout()
    plt.savefig('spectrogram_comparison.png')
    print("\nSUCCESS: Spectrogram comparison saved to 'spectrogram_comparison.png'")

    # Signal check
    audio_max = np.abs(audio).max()
    audio_std = audio.std()
    print(f"Audio Stats -> Max: {audio_max:.4f}, Std: {audio_std:.4f}")
    
    if audio_max < 1e-5:
        print("RESULT: CRITICAL - Audio is silent.")
    elif audio_std < 1e-4:
        print("RESULT: WARNING - Audio is likely static or a continuous drone.")
    else:
        print("RESULT: PASS - Audio contains a dynamic signal.")

if __name__ == "__main__":
    test()
