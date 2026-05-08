import os
import torch
import numpy as np
import dataset_loader

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
dataset_loader.NUM_PROC = 1
SOURCE_LANG = "en"
TARGET_LANG = "de"
OUTPUT_DIR = dataset_loader.DATASETS_DIR

# WavLM model for SOURCE encoding (English → continuous hidden states)
WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"

# SpeechT5 processor for TARGET encoding (German → 80-bin log-mel spectrogram)
SPEECHT5_MODEL_NAME = "microsoft/speecht5_vc"

# ---------------------------------------------------------------------------
# Global state for worker processes
# ---------------------------------------------------------------------------
_wavlm_proc_cache = None       # Wav2Vec2FeatureExtractor for WavLM
_wavlm_model_cache = None      # Frozen WavLMModel backbone
_speecht5_proc_cache = None    # SpeechT5Processor (mel-spectrogram extractor)


# ---------------------------------------------------------------------------
# SOURCE modality: WavLM hidden states  (English → [Seq_Len, 768])
# ---------------------------------------------------------------------------

def _get_wavlm_model():
    """
    Lazy-load and cache the WavLM feature extractor and frozen WavLMModel.
    Moves the model to GPU if available for faster preprocessing.
    """
    global _wavlm_proc_cache, _wavlm_model_cache
    if _wavlm_model_cache is None:
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading WavLM ({WAVLM_MODEL_NAME}) on {device} in worker (PID: {os.getpid()})...")
        
        _wavlm_proc_cache = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_NAME)
        _wavlm_model_cache = WavLMModel.from_pretrained(WAVLM_MODEL_NAME).to(device)
        
        # Freeze all WavLM parameters
        for p in _wavlm_model_cache.parameters():
            p.requires_grad_(False)
        _wavlm_model_cache.eval()
        
    return _wavlm_proc_cache, _wavlm_model_cache


def process_source_wavlm(batch):
    """
    MODALITY BRIDGE — Source (English audio → WavLM continuous hidden states).

    WavLM replaces SpeechT5's built-in CNN feature extractor entirely.
    Output shape per sample: (Seq_Len, 768)
    """
    proc, model = _get_wavlm_model()
    device = next(model.parameters()).device

    encoded_list = []
    with torch.no_grad():
        for audio_item in batch["audio"]:
            audio_array = np.array(audio_item["array"], dtype=np.float32)
            sr = audio_item["sampling_rate"]

            # Resample to 16 kHz
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

            # Normalise and convert to model input on the correct device
            inputs = proc(audio_array, sampling_rate=16000, return_tensors="pt", padding=False)
            input_values = inputs.input_values.to(device)
            
            out = model(input_values)                    # forward through frozen WavLM
            hidden = out.last_hidden_state.squeeze(0)    # (Seq_Len, 768)
            encoded_list.append(hidden.cpu().numpy())    # move back to CPU for storage

    return {"input_values": encoded_list}


# ---------------------------------------------------------------------------
# TARGET modality: 80-bin log-mel spectrogram  (German audio → [T, 80])
# ---------------------------------------------------------------------------

# SpeechT5's exact mel-spectrogram parameters (from modeling_speecht5.py)
_MEL_TRANSFORM = None

def _get_mel_transform():
    """
    Lazy-load and cache the torchaudio MelSpectrogram transform using
    SpeechT5's exact parameters so the stored mel-spectrograms match
    what the decoder expects.
    """
    global _MEL_TRANSFORM
    if _MEL_TRANSFORM is None:
        import torchaudio
        _MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=80.0,
            f_max=7600.0,
            n_mels=80,
            mel_scale="htk",
        )
    return _MEL_TRANSFORM


def process_target_mel(batch):
    """
    MODALITY BRIDGE — Target (German audio → 80-bin log-mel spectrogram).

    Uses the exact MelSpectrogram parameters from SpeechT5's internal
    feature extraction pipeline, followed by log compression, so that
    the stored 'labels' are in the same space the decoder is trained to predict.

    Output shape per sample: (T, 80)
      • T = number of mel frames  (hop_length=256 → ~62.5 frames/sec)
      • 80 mel bins

    Stored under the key 'labels' so the training loop can compute L1/MSE
    between the predicted mel and the ground-truth mel.
    """
    import torch
    mel_transform = _get_mel_transform()

    mel_list = []
    for audio_item in batch["audio"]:
        audio_array = np.array(audio_item["array"], dtype=np.float32)
        sr = audio_item["sampling_rate"]

        # Resample to 16 kHz if needed
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

        waveform = torch.from_numpy(audio_array)  # (N,)

        # mel_transform expects (N,) or (1, N) and returns (80, T)
        mel = mel_transform(waveform)              # (80, T)

        # Apply log compression (same as SpeechT5 internals)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))  # (80, T)

        # Transpose to (T, 80) — time-major format expected by the decoder
        log_mel = log_mel.T                        # (T, 80)

        mel_list.append(log_mel.numpy())

    return {"labels": mel_list}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess_and_save_wavlm():
    """
    Hybrid preprocessing pipeline.

    SOURCE modality  (English):
        WavLM frozen backbone → continuous hidden states (Seq_Len, 768)
        Saved under key: 'input_values'

    TARGET modality  (German):
        SpeechT5Processor feature extractor → 80-bin log-mel spectrogram (T, 80)
        Saved under key: 'labels'

    Both modalities are merged into a single Arrow dataset with columns:
        ['input_values', 'labels']

    This unified layout avoids the need for two separate language directories
    and simplifies DataLoader logic: one row = one aligned (EN, DE) pair.

    Output directory:
        processed_wavlm_{SOURCE_LANG}_{TARGET_LANG}_v2/
            ← Arrow dataset with 'input_values' and 'labels' columns
    """
    print(f"--- Hybrid WavLM/SpeechT5 Preprocessing {SOURCE_LANG} -> {TARGET_LANG} ---")
    print(f"  Source modality : WavLM hidden states ({WAVLM_MODEL_NAME})")
    print(f"  Target modality : 80-bin mel-spectrogram ({SPEECHT5_MODEL_NAME})")

    # num_proc=1 is safest when loading heavy neural net models inside
    # worker processes via fork; avoids CUDA context clashes and OOM.
    num_proc = 1
    print(f"Using {num_proc} CPU worker(s) (conservative for model-based encoding).")

    # ------------------------------------------------------------------
    # 1. Load Raw Data
    # ------------------------------------------------------------------
    print("Loading raw datasets...")
    datasets = dataset_loader.load_data(
        lang=[SOURCE_LANG, TARGET_LANG],
        split="train",
        dataset=["fleurs"],
        num_samples=20000,
    )

    source_ds = datasets.get(SOURCE_LANG)
    target_ds = datasets.get(TARGET_LANG)

    if source_ds is None or target_ds is None:
        print(f"ERROR: Could not load one or both datasets ({SOURCE_LANG}, {TARGET_LANG}).")
        print("Please check the logs above for download/access errors.")
        return

    # Align lengths so every row is a valid (EN, DE) pair
    min_len = min(len(source_ds), len(target_ds))
    source_ds = source_ds.select(range(min_len))
    target_ds = target_ds.select(range(min_len))
    print(f"Initial aligned pairs: {min_len}")

    # ------------------------------------------------------------------
    # 2. SOURCE: Encode English audio → WavLM hidden states (input_values)
    # ------------------------------------------------------------------
    print("Encoding SOURCE (EN) with frozen WavLM → hidden states (Seq_Len, 768)...")
    source_ds = source_ds.map(
        process_source_wavlm,
        batched=True,
        batch_size=64,        # small batch to keep RAM usage bounded
        num_proc=num_proc,
        desc="WavLM Encoding Source Audio",
        remove_columns=["audio"],  # drop raw waveform to save disk space
    )

    # ------------------------------------------------------------------
    # 3. TARGET: Encode German audio → 80-bin mel-spectrogram (labels)
    # ------------------------------------------------------------------
    print("Encoding TARGET (DE) with SpeechT5Processor → 80-bin mel-spectrogram (T, 80)...")
    target_ds = target_ds.map(
        process_target_mel,
        batched=True,
        batch_size=16,
        num_proc=num_proc,
        desc="SpeechT5 Mel Encoding Target Audio",
        remove_columns=["audio"],  # drop raw waveform to save disk space
    )

    # ------------------------------------------------------------------
    # 4. Merge into a single paired dataset
    # ------------------------------------------------------------------
    # Both datasets now have the same number of rows (aligned pairs).
    # We add the 'labels' column from target_ds into source_ds so the
    # result is a single dataset with columns ['input_values', 'labels'].
    print("Merging source hidden states and target mel-spectrograms...")
    
    # Remove overlapping columns (metadata like 'id', 'transcription', etc.) 
    # from target_ds to avoid duplicate column errors during axis=1 merge.
    target_ds = target_ds.remove_columns([c for c in target_ds.column_names if c in source_ds.column_names])
    
    from datasets import concatenate_datasets
    paired_ds = concatenate_datasets([source_ds, target_ds], axis=1)

    # ------------------------------------------------------------------
    # 5. Save to Disk
    # ------------------------------------------------------------------
    out_path = os.path.join(
        OUTPUT_DIR,
        f"processed_wavlm_{SOURCE_LANG}_{TARGET_LANG}_v3",
    )
    print(f"Saving paired dataset to {out_path}...")
    paired_ds.save_to_disk(out_path)

    print(f"\nSUCCESS! Hybrid preprocessing complete.")
    print(f"  Samples      : {len(paired_ds)}")
    print(f"  Columns      : {paired_ds.column_names}")
    print(f"  Output path  : {out_path}")


if __name__ == "__main__":
    preprocess_and_save_wavlm()
