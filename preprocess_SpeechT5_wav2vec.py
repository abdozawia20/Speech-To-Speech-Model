import os
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from dataset_loader import load_data
from encoders import Wav2VecSpeechT5Encoder, SpeechT5MelSpectrogramEncoder

def main():
    # 1. Data Loading
    print("Loading data...")
    # Load exactly 100 aligned samples for 'en' and 'de' from 'fleurs'
    datasets = load_data(dataset=['fleurs'], lang=['en', 'de'], num_samples=100)
    
    en_ds = datasets['en']
    de_ds = datasets['de']
    
    # Ensure they are aligned and have the same number of samples
    assert len(en_ds) == len(de_ds), f"Datasets not aligned: {len(en_ds)} vs {len(de_ds)}"
    
    # 2. Encoding Initialization
    print("Initializing encoders...")
    # Wav2VecSpeechT5Encoder for English (source)
    en_encoder = Wav2VecSpeechT5Encoder(load_decoder=False)
    # SpeechT5MelSpectrogramEncoder for German (target)
    de_encoder = SpeechT5MelSpectrogramEncoder()
    
    # 3. Data Processing Loop
    print("Processing data...")
    processed_data = []
    
    for i in tqdm(range(len(en_ds))):
        en_sample = en_ds[i]
        de_sample = de_ds[i]
        
        # Ensure IDs match
        assert en_sample['id'] == de_sample['id'], f"ID mismatch at index {i}: {en_sample['id']} != {de_sample['id']}"
        
        common_id = en_sample['id']
        en_audio = en_sample['audio']['array']
        de_audio = de_sample['audio']['array']
        
        # Map English audio through Wav2Vec encoder -> (1, Seq_Len, 768)
        # en_encoder.encode returns a torch.Tensor on CPU
        en_hidden_states = en_encoder.encode(en_audio)
        
        # Map German audio through Mel Spectrogram encoder -> (1, Seq_Len, 80)
        # de_encoder.encode returns inputs.input_values (typically a list/array)
        de_mel_features = de_encoder.encode(de_audio)
        
        # Ensure de_mel_features is a tensor or numpy array for consistency
        if isinstance(de_mel_features, list):
            de_mel_features = torch.tensor(de_mel_features)
        elif isinstance(de_mel_features, np.ndarray):
            de_mel_features = torch.from_numpy(de_mel_features)
            
        # Standardize shapes to (Seq_Len, Hidden/Mel) for saving
        # en_hidden_states is (1, S, 768) -> (S, 768)
        en_hidden_states = en_hidden_states.squeeze(0)
        # de_mel_features is likely (S, 80) or (1, S, 80) depending on processor
        if de_mel_features.ndim == 3:
            de_mel_features = de_mel_features.squeeze(0)
            
        processed_data.append({
            'id': common_id,
            'en_hidden_states': en_hidden_states.numpy(),
            'de_mel_spectrogram': de_mel_features.numpy()
        })
        
    # 4. Data Persistence
    print("Saving processed data...")
    output_dir = "preprocessed_speecht5_wav2vec"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a HF Dataset for easy loading later
    dataset = Dataset.from_list(processed_data)
    dataset.save_to_disk(output_dir)
    
    print(f"Successfully saved 100 samples to {output_dir}")

if __name__ == "__main__":
    main()
