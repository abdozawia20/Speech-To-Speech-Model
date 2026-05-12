import os
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
from datasets import load_from_disk
from dataset_loader import load_data
from encoders import Wav2VecSpeechT5Encoder
from transformers import SpeechT5HifiGan

def main():
    # 0. Setup
    output_dir = "validation_results"
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Data
    print("Loading preprocessed dataset...")
    preprocessed_ds = load_from_disk("preprocessed_speecht5_wav2vec")
    
    print("Loading original raw audio for comparison...")
    # Using 100 samples to match the preprocessing run, then we pick the first one
    original_datasets = load_data(dataset=['fleurs'], lang=['en', 'de'], num_samples=100)
    en_orig_ds = original_datasets['en']
    de_orig_ds = original_datasets['de']

    # 2. Select 1 target sample pair
    # Use the first sample
    prep_sample = preprocessed_ds[0]
    sample_id = prep_sample['id']
    
    # Find the corresponding original samples
    # They should be at index 0 because load_data sorts by id, and prep was done in order
    en_orig = en_orig_ds[0]
    de_orig = de_orig_ds[0]
    
    assert str(en_orig['id']) == str(sample_id), f"ID mismatch: {en_orig['id']} != {sample_id}"
    assert str(de_orig['id']) == str(sample_id), f"ID mismatch: {de_orig['id']} != {sample_id}"
    
    print(f"Processing sample ID: {sample_id}")

    # 3. Audio Extraction/Reconstruction
    print("Generating audio files...")
    
    # original_en.wav
    orig_en_audio = en_orig['audio']['array']
    sf.write(os.path.join(output_dir, "original_en.wav"), orig_en_audio, 16000)
    
    # original_de.wav
    orig_de_audio = de_orig['audio']['array']
    sf.write(os.path.join(output_dir, "original_de.wav"), orig_de_audio, 16000)
    
    # reconstructed_en_w2v.wav
    print("Reconstructing English audio from Wav2Vec hidden states...")
    w2v_encoder = Wav2VecSpeechT5Encoder(load_decoder=True)
    # Convert list to numpy array if necessary and ensure float32
    en_hidden_states = np.array(prep_sample['en_hidden_states'], dtype=np.float32)
    # Reshape to (Seq_Len, 768)
    if en_hidden_states.ndim == 1:
        en_hidden_states = en_hidden_states.reshape(-1, 768)
    en_hidden_states = torch.from_numpy(en_hidden_states)
    # speaker_embedding: zero tensor of shape (1, 512)
    speaker_embedding = torch.zeros(1, 512, dtype=torch.float32)
    
    recon_en = w2v_encoder.decode(en_hidden_states, speaker_embedding)
    sf.write(os.path.join(output_dir, "reconstructed_en_w2v.wav"), recon_en, 16000)
    
    # reconstructed_de_mel.wav
    print("Reconstructing German audio from Mel Spectrogram...")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
    # Convert list to numpy array if necessary and ensure float32
    de_mel = np.array(prep_sample['de_mel_spectrogram'], dtype=np.float32)
    # Reshape to (Seq_Len, 80)
    if de_mel.ndim == 1:
        de_mel = de_mel.reshape(-1, 80)
    de_mel = torch.from_numpy(de_mel).to(device)
    
    # HiFi-GAN expects (1, T, 80)
    if de_mel.ndim == 2:
        de_mel = de_mel.unsqueeze(0)
    
    print(f"de_mel shape: {de_mel.shape}")
    if hasattr(vocoder, "mean"):
        print(f"vocoder.mean shape: {vocoder.mean.shape}")
    if hasattr(vocoder, "scale"):
        print(f"vocoder.scale shape: {vocoder.scale.shape}")

    with torch.no_grad():
        try:
            recon_de = vocoder(de_mel).cpu().numpy()
        except Exception as e:
            print(f"HiFi-GAN decoding failed: {e}")
            print("Trying transpose...")
            recon_de = vocoder(de_mel.transpose(1, 2)).cpu().numpy()
            print("Transpose successful!")
    
    sf.write(os.path.join(output_dir, "reconstructed_de_mel.wav"), recon_de, 16000)

    # 4. Spectrogram Visualization
    print("Generating spectrogram visualizations...")
    
    # Calculate original German Mel Spectrogram
    # Using librosa to match SpeechT5 parameters (80 bins, 16kHz)
    S = librosa.feature.melspectrogram(y=orig_de_audio, sr=16000, n_mels=80, fmax=8000)
    S_dB_orig = librosa.power_to_db(S, ref=np.max)
    
    # Preprocessed Mel Spectrogram (already log-mel from SpeechT5Processor)
    # Use the reshaped de_mel from before (which was (1, T, 80) or similar)
    # Let's get it back to (T, 80) then transpose for specshow (80, T)
    S_prep = de_mel.cpu().squeeze().numpy().T
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    librosa.display.specshow(S_dB_orig, x_axis='time', y_axis='mel', sr=16000, fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Original German Mel Spectrogram (librosa)')
    
    plt.subplot(2, 1, 2)
    # prep_sample['de_mel_spectrogram'] is log-mel features from SpeechT5Processor
    librosa.display.specshow(S_prep, x_axis='time', y_axis='mel', sr=16000, fmax=8000)
    plt.colorbar()
    plt.title('Preprocessed German Mel Spectrogram (SpeechT5)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spectrogram_comparison.png"))
    plt.close()

    # 5. Verification
    print("\nVerification Results:")
    files_to_check = [
        "original_en.wav",
        "original_de.wav",
        "reconstructed_en_w2v.wav",
        "reconstructed_de_mel.wav",
        "spectrogram_comparison.png"
    ]
    
    for f in files_to_check:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"[OK] {f} exists, size: {size} bytes")
        else:
            print(f"[FAILED] {f} is missing")

if __name__ == "__main__":
    main()
