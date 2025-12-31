
import os
import sys

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.getcwd(), ''))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset_loader import load_data
from encoders import SpeechT5Encoder

# 1. Initialize Encoder
print("Initializing SpeechT5 Encoder...")
encoder = SpeechT5Encoder() # Defaults to microsoft/speecht5_tts

# 2. Load and Preprocess Data
# "load_data" will automatically apply the encoding passed to it.
print("Loading and Preprocessing Dataset...")
processed_dataset = load_data(
    start_idx=0, 
    num_samples=60000, # Set to None for full dataset
    lang=['en', 'de'], # Load both source and target
    split="train",
    dataset=["seamless_align"],
    encoding=encoder # This triggers the SpeechT5 preprocessing
)

# 3. Inspect Result
print("\nPreprocessing Complete.")
for lang, ds in processed_dataset.items():
    if ds is None: continue
    print(f"\nLanguage: {lang}")
    print(f"Sample Count: {len(ds)}")
    
    if len(ds) > 0:
        sample = ds[0]
        # 'audio' should now be a numpy array of shape (T, 80)
        print(f"Sample 0 Audio Shape: {len(sample['audio'])}") 
        print(f"Sample 0 Keys: {sample.keys()}")
        
        # Verify it's not a raw audio dict anymore
        if isinstance(sample['audio'], dict):
             print("WARNING: Audio is still a dictionary (Preprocessing failed?)")
        else:
             import numpy as np
             arr = np.array(sample['audio'])
             print(f"Spectrogram Shape: {arr.shape} (Time, MelBins)")

save_path = os.path.join("datasets", "processed_speecht5_de_en")
for lang, ds in processed_dataset.items():
   if ds:
       ds.save_to_disk(os.path.join(save_path, lang))
print(f"Saved processed dataset to {save_path}")
