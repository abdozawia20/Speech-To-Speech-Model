import os
import torch
import numpy as np
from datasets import load_from_disk, Dataset, disable_progress_bar
from transformers import SpeechT5Processor, Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import dataset_loader
import multiprocessing

# CONFIGURATION
dataset_loader.NUM_PROC = 4
SOURCE_LANG = "en"
TARGET_LANG = "de"
OUTPUT_DIR = dataset_loader.DATASETS_DIR
PROCESSOR_NAME = "microsoft/speecht5_vc"
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base-960h"

# Global variables for workers
processor = None
wav2vec_processor = None

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
    """
    Convert target audio to SpeechT5-compatible 80-bin log-mel spectrograms.

    The correct path is ``SpeechT5Processor(audio_target=audio)`` which
    internally calls ``SpeechT5FeatureExtractor`` in *decoder* mode and
    returns a properly normalised log-mel spectrogram of shape (Time, 80).

    WRONG (previous bug)::
        proc.feature_extractor(audio_arrays).input_values
        # → raw, mean-var normalised waveform  shape=(N_samples,)  ❌

    CORRECT::
        proc(audio_target=audio, sampling_rate=16000).input_values
        # → log-mel spectrogram  shape=(Time, 80)  ✅

    SpeechT5 mel-spec settings (from microsoft/speecht5_vc):
        num_mel_bins = 80, hop_length = 16, win_length = 64, n_fft = 1024,
        sampling_rate = 16000, fmin = 80, fmax = 7600, reduction_factor = 2.
    """
    proc = get_processor()
    if proc is None:
        raise RuntimeError("Processor failed to initialize")

    spectrograms = []
    for audio_item in batch["audio"]:
        audio_array = np.array(audio_item["array"], dtype=np.float32).flatten()
        sr = audio_item.get("sampling_rate", 16000)

        # Resample to 16 kHz if needed
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

        # Use the *decoder* path of SpeechT5Processor:
        #   proc(audio_target=audio, sampling_rate=16000)
        # This returns input_values of shape (1, Time, 80).
        out = proc(audio_target=audio_array, sampling_rate=16000)
        # Squeeze batch dim → (Time, 80) numpy array
        mel = np.array(out.input_values[0], dtype=np.float32)  # (Time, 80)
        spectrograms.append(mel)

    return {"audio": spectrograms}

def preprocess_and_save():
    print(f"--- Preprocessing {SOURCE_LANG} -> {TARGET_LANG} with Multithreading ---")
    
    num_proc = dataset_loader.NUM_PROC
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

    # 2. Process Audio & Spectrograms (Parallelized)
    print("Generating Features & Spectrograms (Parallel)...")
    
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

    # 3. Save to Disk
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

    num_proc = 1
    print(f"Using {num_proc} CPU worker(s) (conservative for Wav2Vec2).")

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

    # 2. Encode Source with Wav2Vec2, Target with Mel Spectrogram
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

    # 3. Save to Disk
    out_path = os.path.join(
        OUTPUT_DIR,
        f"processed_speecht5_wav2vec_{SOURCE_LANG}_{TARGET_LANG}_v3",
    )
    print(f"Saving to {out_path}...")
    source_ds.save_to_disk(os.path.join(out_path, SOURCE_LANG))
    target_ds.save_to_disk(os.path.join(out_path, TARGET_LANG))

    print("SUCCESS! Wav2Vec preprocessing complete.")


if __name__ == "__main__":
    # Change to preprocess_and_save_wav2vec() to use the Wav2Vec2 pipeline.
    # preprocess_and_save()          # <- mel-spectrogram source
    preprocess_and_save_wav2vec()    # <- Wav2Vec2 hidden-state source
