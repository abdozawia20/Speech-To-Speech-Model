import os
import torch
import numpy as np
from datasets import load_from_disk, Dataset, disable_progress_bar
from transformers import SpeechT5Processor, Wav2Vec2Processor, Wav2Vec2Model
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
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base-960h"

# Global variables for workers
processor = None
language_model = None
wav2vec_processor = None

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
    """Lazy load SpeechT5 processor."""
    global processor
    if processor is None:
        try:
            print(f"Loading SpeechT5 Processor in worker (PID: {os.getpid()})...")
            processor = SpeechT5Processor.from_pretrained(PROCESSOR_NAME)
        except Exception as e:
            print(f"Error loading processor: {e}")
            return None
    return processor


def get_wav2vec_processor():
    """Lazy load Wav2Vec2 processor for worker processes."""
    global wav2vec_processor
    if wav2vec_processor is None:
        try:
            print(f"Loading Wav2Vec2 Processor in worker (PID: {os.getpid()})...")
            wav2vec_processor = Wav2Vec2Processor.from_pretrained(WAV2VEC_MODEL_NAME)
        except Exception as e:
            print(f"Error loading Wav2Vec2 processor: {e}")
            return None
    return wav2vec_processor


def process_source_batch(batch):
    """Normalize source audio to SpeechT5 input_values (raw waveform, mean-var normalised)."""
    proc = get_processor()
    if proc is None:
        # Should handle error gracefully or raise
        raise RuntimeError("Processor failed to initialize")
        
    audio_arrays = [x["array"] for x in batch["audio"]]
    # processing with padding=False returns a list of variable length arrays
    out = proc(audio=audio_arrays, sampling_rate=16000)
    return {"audio": out.input_values}


def process_source_batch_wav2vec(batch):
    """
    Encode source audio into Wav2Vec2 hidden states compatible with SpeechT5.

    Each sample is processed individually (no padding) to keep the exact
    sequence length for that utterance.  The output stored per sample is a
    2-D numpy array of shape (Seq_Len, 768).

    This function is intentionally separate from process_source_batch so
    both preprocessing strategies can coexist in the same codebase.
    """
    proc = get_wav2vec_processor()
    if proc is None:
        raise RuntimeError("Wav2Vec2 Processor failed to initialize.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Lazy-load the model within the worker (CPU only to avoid CUDA forking issues)
    # Use a module-level cache so it is reused across batches in the same worker.
    global _wav2vec_model_cache
    if "_wav2vec_model_cache" not in globals() or _wav2vec_model_cache is None:
        try:
            print(f"Loading Wav2Vec2 Model in worker (PID: {os.getpid()}, device={device})...")
            _wav2vec_model_cache = Wav2Vec2Model.from_pretrained(WAV2VEC_MODEL_NAME).to(device)
            _wav2vec_model_cache.eval()
            for p in _wav2vec_model_cache.parameters():
                p.requires_grad = False
        except Exception as e:
            raise RuntimeError(f"Wav2Vec2 Model failed to load in worker: {e}")

    model = _wav2vec_model_cache
    encoded_list = []

    for audio_item in batch["audio"]:
        audio_array = np.array(audio_item["array"], dtype=np.float32).flatten()
        sr = audio_item["sampling_rate"]

        # Wav2Vec2Processor normalises mean/variance and resamples if needed
        inputs = proc(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.to(device)  # (1, T)

        with torch.no_grad():
            outputs = model(input_values)

        # Shape: (1, Seq_Len, 768) -> squeeze to (Seq_Len, 768) numpy array
        hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()
        encoded_list.append(hidden)

    return {"audio": encoded_list}

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

def preprocess_and_save_wav2vec():
    """
    Preprocess the dataset using Wav2Vec2 hidden states for the source side
    and 80-bin log-mel spectrograms for the target side, then save to disk.

    Source output:  numpy arrays of shape (Seq_Len, 768)  — Wav2Vec2 hidden states
    Target output:  numpy arrays of shape (Time, 80)       — log-mel spectrograms

    Output directory:
        processed_speecht5_wav2vec_{SOURCE_LANG}_{TARGET_LANG}_v1/
            {SOURCE_LANG}/   ← Arrow dataset with Wav2Vec hidden states
            {TARGET_LANG}/   ← Arrow dataset with mel spectrograms
    """
    print(f"--- Wav2Vec Preprocessing {SOURCE_LANG} -> {TARGET_LANG} ---")
    print(f"Max Duration: {MAX_DURATION_SECONDS}s")

    # NOTE: num_proc=1 is safest when loading heavy neural net models inside
    # worker processes via fork. Increase cautiously.
    num_proc = 1
    print(f"Using {num_proc} CPU worker(s) (conservative for Wav2Vec2 + Whisper).")

    # 1. Load Raw Data
    print("Loading raw datasets...")
    datasets = dataset_loader.load_data(
        lang=[SOURCE_LANG, TARGET_LANG],
        split="train",
        dataset=["seamless_align"],
        num_samples=15000,
    )

    source_ds = datasets[SOURCE_LANG]
    target_ds = datasets[TARGET_LANG]

    # Align lengths
    min_len = min(len(source_ds), len(target_ds))
    source_ds = source_ds.select(range(min_len))
    target_ds = target_ds.select(range(min_len))
    print(f"Initial aligned pairs: {min_len}")

    # 2. Filter by Duration
    print("Calculating durations...")
    source_ds = source_ds.map(compute_duration, batched=True, num_proc=num_proc, desc="Calc Source Durations")
    target_ds = target_ds.map(compute_duration, batched=True, num_proc=num_proc, desc="Calc Target Durations")

    print(f"Filtering samples > {MAX_DURATION_SECONDS}s...")
    src_durs = np.array(source_ds["duration"])
    tgt_durs = np.array(target_ds["duration"])
    valid_mask = (src_durs <= MAX_DURATION_SECONDS) & (tgt_durs <= MAX_DURATION_SECONDS)
    valid_indices = np.where(valid_mask)[0]

    print(f"Kept {len(valid_indices)} / {min_len} samples (Duration Filter).")
    source_ds = source_ds.select(valid_indices)
    target_ds = target_ds.select(valid_indices)
    source_ds = source_ds.remove_columns(["duration"])
    target_ds = target_ds.remove_columns(["duration"])

    # 3. Filter by Language
    print("Filtering by Language (Whisper Detection)...")
    source_ds = source_ds.map(
        check_language,
        batched=True,
        batch_size=8,
        num_proc=num_proc,
        fn_kwargs={"expected_lang": SOURCE_LANG},
        desc=f"Checking Source Language ({SOURCE_LANG})",
    )
    target_ds = target_ds.map(
        check_language,
        batched=True,
        batch_size=8,
        num_proc=num_proc,
        fn_kwargs={"expected_lang": TARGET_LANG},
        desc=f"Checking Target Language ({TARGET_LANG})",
    )

    src_valid = np.array(source_ds["is_valid_lang"])
    tgt_valid = np.array(target_ds["is_valid_lang"])
    lang_mask = src_valid & tgt_valid
    lang_valid_indices = np.where(lang_mask)[0]

    print(f"Kept {len(lang_valid_indices)} / {len(source_ds)} samples (Language Filter).")
    source_ds = source_ds.select(lang_valid_indices)
    target_ds = target_ds.select(lang_valid_indices)
    source_ds = source_ds.remove_columns(["is_valid_lang"])
    target_ds = target_ds.remove_columns(["is_valid_lang"])

    # 4. Encode Source with Wav2Vec2, Target with Mel Spectrogram
    print("Encoding Source with Wav2Vec2 (hidden states)...")
    source_ds = source_ds.map(
        process_source_batch_wav2vec,
        batched=True,
        batch_size=16,   # smaller batch: each sample is processed individually inside
        num_proc=num_proc,
        desc="Wav2Vec2 Encoding Source Audio",
    )

    print("Encoding Target with Mel Spectrogram...")
    target_ds = target_ds.map(
        process_target_batch,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        desc="Processing Target Spectrograms",
    )

    # 5. Save to Disk
    out_path = os.path.join(
        OUTPUT_DIR,
        f"processed_speecht5_wav2vec_{SOURCE_LANG}_{TARGET_LANG}_v1",
    )
    print(f"Saving to {out_path}...")
    source_ds.save_to_disk(os.path.join(out_path, SOURCE_LANG))
    target_ds.save_to_disk(os.path.join(out_path, TARGET_LANG))

    print("SUCCESS! Wav2Vec preprocessing complete.")


if __name__ == "__main__":
    # Change to preprocess_and_save_wav2vec() to use the Wav2Vec2 pipeline.
    # preprocess_and_save()          # <- mel-spectrogram source
    preprocess_and_save_wav2vec()    # <- Wav2Vec2 hidden-state source