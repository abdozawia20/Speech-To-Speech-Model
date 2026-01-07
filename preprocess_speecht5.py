import os
import torch
import numpy as np
from datasets import load_from_disk, Dataset, disable_progress_bar
from transformers import SpeechT5Processor
from tqdm import tqdm
import dataset_loader
import multiprocessing
from faster_whisper import WhisperModel

# CONFIGURATION
dataset_loader.NUM_PROC = 4
SOURCE_LANG = "en"
TARGET_LANG = "de"
# CRITICAL FOR VRAM: Limit audio to 8 seconds. 
MAX_DURATION_SECONDS = 8.0 
OUTPUT_DIR = dataset_loader.DATASETS_DIR
PROCESSOR_NAME = "microsoft/speecht5_vc"

# Global variables for workers
processor = None
language_model = None

def init_worker():
    """Initialize global models in each worker process."""
    global processor, language_model
    
    # Initialize SpeechT5 Processor
    try:
        processor = SpeechT5Processor.from_pretrained(PROCESSOR_NAME)
    except Exception as e:
        print(f"Warning: Failed to load processor in worker: {e}")
        processor = None
        
    # Initialize Whisper for Language Detection (CPU, Tiny model for speed)
    try:
        # Use 'tiny' model on CPU with int8 quantization for minimal overhead
        language_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    except Exception as e:
        print(f"Warning: Failed to load Whisper model in worker: {e}")
        language_model = None

def compute_duration(batch):
    """Helper to calculate duration of audio clips in batch."""
    durations = []
    for x in batch["audio"]:
        # Accessing 'array' triggers decoding
        durations.append(len(x['array']) / x['sampling_rate'])
    return {"duration": durations}

def check_language(batch, expected_lang):
    """
    Detects language of the audio and checks if it matches expected_lang.
    Returns a list of booleans (True if match/valid, False otherwise).
    """
    global language_model
    
    # Lazy loading for workers
    if language_model is None:
        try:
            print(f"Loading Whisper model in worker process (PID: {os.getpid()})...")
            language_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        except Exception as e:
            print(f"Warning: Failed to load Whisper model in worker: {e}")
            # Fallback: accept all if model fails
            return {"is_valid_lang": [True] * len(batch["audio"])}

    is_valid = []
    
    for x in batch["audio"]:
        audio = x['array']
        sr = x['sampling_rate']
        
        # Whisper expects 16k
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        try:
            # detect_language returns (segments, info) but for just detection we can use
            # segments, info = language_model.transcribe(audio, beam_size=5) -> this is full transcription
            # faster_whisper has direct language detection? No, usually usually part of transcribe info.
            # We can use a short segment or just run transcribe with task='transcribe' 
            # and check info.language.
            
            # Using verify=True or just running transcribe on the first 30s (which it is <8s anyway)
            segments, info = language_model.transcribe(
                audio, 
                beam_size=1, # Greedy is faster
                language=None, # Auto-detect
                task="transcribe" 
            )
            
            detected = info.language
            
            # Logic: 
            # If we expect 'de', we definitely don't want 'en'.
            # If we expect 'en', we want 'en'.
            if detected == expected_lang:
                is_valid.append(True)
            else:
                 # Optional: Allow high confidence mismatch only? 
                 # For now, strict check: if detected != expected, drop it.
                 # Debug print could be useful but spammy in multiprocessing.
                 is_valid.append(False)

        except Exception as e:
            # On error, keep the sample? Or drop? Let's keep to be safe against glitches.
            is_valid.append(True)
            
    return {"is_valid_lang": is_valid}

def get_processor():
    """Lazy load processor."""
    global processor
    if processor is None:
        try:
            print(f"Loading SpeechT5 Processor in worker (PID: {os.getpid()})...")
            processor = SpeechT5Processor.from_pretrained(PROCESSOR_NAME)
        except Exception as e:
            print(f"Error loading processor: {e}")
            return None
    return processor

def process_source_batch(batch):
    """Normalize source audio to inputs."""
    proc = get_processor()
    if proc is None:
        # Should handle error gracefully or raise
        raise RuntimeError("Processor failed to initialize")
        
    audio_arrays = [x["array"] for x in batch["audio"]]
    # processing with padding=False returns a list of variable length arrays
    out = proc(audio=audio_arrays, sampling_rate=16000)
    return {"audio": out.input_values}

def process_target_batch(batch):
    """Convert target audio to spectrograms."""
    proc = get_processor()
    if proc is None:
        raise RuntimeError("Processor failed to initialize")
        
    audio_arrays = [x["array"] for x in batch["audio"]]
    
    # FIX: Use feature_extractor directly to avoid 'labels' KeyError.
    # The feature_extractor returns 'input_values' (Log-Mel Spectrograms).
    # We explicitly request it to handle the batch.
    out = proc.feature_extractor(
        audio_arrays, 
        sampling_rate=16000, 
        return_attention_mask=False
    )
    
    # The output key is always 'input_values' from the feature_extractor
    return {"audio": out.input_values}

def preprocess_and_save():
    print(f"--- Preprocessing {SOURCE_LANG} -> {TARGET_LANG} with Multithreading ---")
    print(f"Max Duration: {MAX_DURATION_SECONDS}s")
    
    # Use fewer workers because Whisper consumes more CPU
    num_proc = 2 
    print(f"Using {num_proc} CPU workers (reduced for Whisper).")

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
    # Note: init_worker isn't needed for duration but beneficial for next steps if we reused pool (hf datasets creates new pools)
    source_ds = source_ds.map(compute_duration, batched=True, num_proc=num_proc, desc="Calc Source Durations")
    target_ds = target_ds.map(compute_duration, batched=True, num_proc=num_proc, desc="Calc Target Durations")

    print(f"Filtering samples > {MAX_DURATION_SECONDS}s...")
    
    src_durs = np.array(source_ds['duration'])
    tgt_durs = np.array(target_ds['duration'])
    
    valid_mask = (src_durs <= MAX_DURATION_SECONDS) & (tgt_durs <= MAX_DURATION_SECONDS)
    valid_indices = np.where(valid_mask)[0]
    
    print(f"Kept {len(valid_indices)} / {min_len} samples (Duration Filter).")
    
    source_ds = source_ds.select(valid_indices)
    target_ds = target_ds.select(valid_indices)
    
    source_ds = source_ds.remove_columns(["duration"])
    target_ds = target_ds.remove_columns(["duration"])

    # 3. Filter by Language (New Step)
    print("Filtering by Language (Whisper Detection)...")
    
    # Source Language Check
    # We define partial functions to pass expected language using fn_kwargs
    source_ds = source_ds.map(
        check_language,
        batched=True,
        batch_size=8, # Smaller batch size for inference
        num_proc=num_proc, 
        fn_kwargs={"expected_lang": SOURCE_LANG},
        desc=f"Checking Source Language ({SOURCE_LANG})" 
        # Note: We need to pass init_worker to load models, but load_from_disk + map with num_proc 
        # doesn't easily accept an init function unless we use a custom implementation or start_method changes.
        # Actually, 'map' in datasets with 'num_proc' spawns processes that import the script.
        # So we can use a hack: check if model is loaded in `check_language`, if not, load it.
        # But `init_worker` is cleaner if we can hook it. 
        # HF Datasets doesn't support 'initializer' arg in map directly.
        # So we'll rely on global lazy loading inside check_language or global scope execution.
        # 'multiprocessing' default start method on Linux is fork, so globals might be copied?
        # No, 'spawn' is safer for CUDA/heavy libraries but 'fork' is default.
        # Let's rely on standard laziness: We'll put the load inside `check_language` if None.
    )
    
    # Target Language Check
    target_ds = target_ds.map(
        check_language,
        batched=True,
        batch_size=8,
        num_proc=num_proc,
        fn_kwargs={"expected_lang": TARGET_LANG},
        desc=f"Checking Target Language ({TARGET_LANG})"
    )

    src_valid = np.array(source_ds['is_valid_lang'])
    tgt_valid = np.array(target_ds['is_valid_lang'])
    
    # Both must be valid
    lang_mask = src_valid & tgt_valid
    lang_valid_indices = np.where(lang_mask)[0]
    
    print(f"Kept {len(lang_valid_indices)} / {len(source_ds)} samples (Language Filter).")

    source_ds = source_ds.select(lang_valid_indices)
    target_ds = target_ds.select(lang_valid_indices)
    
    source_ds = source_ds.remove_columns(["is_valid_lang"])
    target_ds = target_ds.remove_columns(["is_valid_lang"])


    # 4. Process Audio & Spectrograms (Parallelized)
    print("Generating Features & Spectrograms (Parallel)...")
    
    # Use 'init_worker' idea? 
    # Since we can't easily pass init to map, we'll simple do lazy init in the function if needed,
    # or rely on the fact that we define it at top level.
    # Actually, for 'fork', globals are preserved.
    # But to be safe, we'll do the "if model is None: init()" check in the function.
                                  
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

    # 5. Save to Disk
    out_path = os.path.join(OUTPUT_DIR, f"processed_speecht5_{SOURCE_LANG}_{TARGET_LANG}_v2_cleaned")
    
    print(f"Saving to {out_path}...")
    source_ds.save_to_disk(os.path.join(out_path, SOURCE_LANG))
    target_ds.save_to_disk(os.path.join(out_path, TARGET_LANG))
    
    print("SUCCESS! Preprocessing complete.")

if __name__ == "__main__":
    # Ensure start method is compatible (optional)
    # multiprocessing.set_start_method("spawn", force=True) 
    # Fork is usually fine for CPU-only libraries, but tokenizers sometimes deadlock.
    # If we had issues, we'd uncomment above.
    preprocess_and_save()