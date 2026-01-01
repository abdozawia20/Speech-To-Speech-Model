import os
import torch
import numpy as np
from datasets import load_from_disk, Dataset, disable_progress_bar
from transformers import SpeechT5Processor
from tqdm import tqdm
import dataset_loader
import multiprocessing

# CONFIGURATION
dataset_loader.NUM_PROC = 4
SOURCE_LANG = "en"
TARGET_LANG = "de"
# CRITICAL FOR VRAM: Limit audio to 8 seconds. 
MAX_DURATION_SECONDS = 8.0 
OUTPUT_DIR = dataset_loader.DATASETS_DIR
PROCESSOR_NAME = "microsoft/speecht5_vc"

# Initialize processor globally so it can be pickled by multiprocessing workers
try:
    processor = SpeechT5Processor.from_pretrained(PROCESSOR_NAME)
except Exception as e:
    print(f"Warning: Failed to load processor globally: {e}")
    processor = None

def compute_duration(batch):
    """Helper to calculate duration of audio clips in batch."""
    durations = []
    for x in batch["audio"]:
        # Accessing 'array' triggers decoding
        durations.append(len(x['array']) / x['sampling_rate'])
    return {"duration": durations}

def process_source_batch(batch):
    """Normalize source audio to inputs."""
    audio_arrays = [x["array"] for x in batch["audio"]]
    # processing with padding=False returns a list of variable length arrays
    out = processor(audio=audio_arrays, sampling_rate=16000)
    return {"audio": out.input_values}

def process_target_batch(batch):
    """Convert target audio to spectrograms."""
    audio_arrays = [x["array"] for x in batch["audio"]]
    
    # FIX: Use feature_extractor directly to avoid 'labels' KeyError.
    # The feature_extractor returns 'input_values' (Log-Mel Spectrograms).
    # We explicitly request it to handle the batch.
    out = processor.feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        return_attention_mask=False
    )
    
    # The output key is always 'input_values' from the feature_extractor
    return {"audio": out.input_values}

def preprocess_and_save():
    print(f"--- Preprocessing {SOURCE_LANG} -> {TARGET_LANG} with Multithreading ---")
    print(f"Max Duration: {MAX_DURATION_SECONDS}s")
    
    num_proc = 3
    print(f"Using {num_proc} CPU workers.")

    # 1. Load Raw Data
    print("Loading raw datasets...")
    datasets = dataset_loader.load_data(
        lang=[SOURCE_LANG, TARGET_LANG], 
        split="train", 
        dataset=['seamless_align'],
        num_samples=15000
    )
    
    source_ds = datasets[SOURCE_LANG]
    target_ds = datasets[TARGET_LANG]
    
    # Align lengths
    min_len = min(len(source_ds), len(target_ds))
    source_ds = source_ds.select(range(min_len))
    target_ds = target_ds.select(range(min_len))
    
    print(f"Initial aligned pairs: {min_len}")

    # 2. Filter by Duration (Parallelized)
    print("Calculating durations...")
    # We map a new 'duration' column to both datasets in parallel
    source_ds = source_ds.map(compute_duration, batched=True, num_proc=num_proc, desc="Calc Source Durations")
    target_ds = target_ds.map(compute_duration, batched=True, num_proc=num_proc, desc="Calc Target Durations")

    print(f"Filtering samples > {MAX_DURATION_SECONDS}s...")
    
    # Now we filter indices based on the metadata we just computed
    # Converting to numpy makes boolean operations fast
    src_durs = np.array(source_ds['duration'])
    tgt_durs = np.array(target_ds['duration'])
    
    # Find indices where BOTH are within limits
    valid_mask = (src_durs <= MAX_DURATION_SECONDS) & (tgt_durs <= MAX_DURATION_SECONDS)
    valid_indices = np.where(valid_mask)[0]
    
    print(f"Kept {len(valid_indices)} / {min_len} samples.")
    
    # Select only short samples
    source_ds = source_ds.select(valid_indices)
    target_ds = target_ds.select(valid_indices)
    
    # Drop the duration column to clean up
    source_ds = source_ds.remove_columns(["duration"])
    target_ds = target_ds.remove_columns(["duration"])

    # 3. Process Audio & Spectrograms (Parallelized)
    print("Generating Features & Spectrograms (Parallel)...")
    
    # Map the processing functions
    # This replaces the 'audio' column with the processed tensors
    source_ds = source_ds.map(
        process_source_batch, 
        batched=True, 
        batch_size=32, 
        num_proc=num_proc, 
        desc="Processing Source Audio"
    )
    
    target_ds = target_ds.map(
        process_target_batch, 
        batched=True, 
        batch_size=32, 
        num_proc=num_proc, 
        desc="Processing Target Spectrograms"
    )

    # 4. Save to Disk
    out_path = os.path.join(OUTPUT_DIR, f"processed_speecht5_{SOURCE_LANG}_{TARGET_LANG}_v2")
    
    print(f"Saving to {out_path}...")
    source_ds.save_to_disk(os.path.join(out_path, SOURCE_LANG))
    target_ds.save_to_disk(os.path.join(out_path, TARGET_LANG))
    
    print("SUCCESS! Preprocessing complete.")

if __name__ == "__main__":
    preprocess_and_save()