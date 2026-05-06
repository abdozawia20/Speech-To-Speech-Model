import os
import sys
import torch
import numpy as np
from faster_whisper import WhisperModel

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import dataset_loader

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
dataset_loader.NUM_PROC = 1 # Recommended for model mapping steps
SOURCE_LANG = "en"
TARGET_LANG = "de"
MAX_DURATION_SECONDS = 8.0
OUTPUT_DIR = dataset_loader.DATASETS_DIR

# WavLM model for SOURCE encoding (English → continuous hidden states)
WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"

# SpeechT5 model name for reference
SPEECHT5_MODEL_NAME = "microsoft/speecht5_vc"

# ---------------------------------------------------------------------------
# Global state for worker processes
# ---------------------------------------------------------------------------
language_model = None          # Whisper language detector (language filter step)
_wavlm_proc_cache = None       # Wav2Vec2FeatureExtractor for WavLM
_wavlm_model_cache = None      # Frozen WavLMModel backbone
_MEL_TRANSFORM = None

# ---------------------------------------------------------------------------
# Worker utilities — language detection
# ---------------------------------------------------------------------------

def compute_duration(batch):
    """Calculate duration of audio clips in batch."""
    durations = []
    for x in batch["audio"]:
        durations.append(len(x["array"]) / x["sampling_rate"])
    return {"duration": durations}


def check_language(batch, expected_lang):
    """
    Detect the language of each audio clip using faster-whisper and return
    a boolean column 'is_valid_lang'.
    """
    global language_model

    if language_model is None:
        try:
            print(f"Loading Whisper model in worker (PID: {os.getpid()})...")
            language_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        except Exception as e:
            print(f"Warning: Failed to load Whisper model in worker: {e}")
            return {"is_valid_lang": [True] * len(batch["audio"])}

    is_valid = []
    for x in batch["audio"]:
        audio = x["array"]
        sr = x["sampling_rate"]

        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        try:
            segments, info = language_model.transcribe(
                audio,
                beam_size=1,
                language=None,
                task="transcribe",
            )
            is_valid.append(info.language == expected_lang)
        except Exception:
            is_valid.append(True)  # keep on transcription error

    return {"is_valid_lang": is_valid}


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
# TARGET modality: Mel Spectrogram and Waveform (German audio)
# ---------------------------------------------------------------------------

def _get_mel_transform():
    """
    Lazy-load and cache the torchaudio MelSpectrogram transform using
    SpeechT5's exact parameters.
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


def process_target_vocoder(batch):
    """
    Extracts:
      - 'labels': 80-bin log-mel spectrogram (T, 80)
      - 'target_waveform': Raw 16kHz audio waveform
    """
    import torch
    mel_transform = _get_mel_transform()

    mel_list = []
    waveform_list = []
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

        # Apply log compression
        log_mel = torch.log(torch.clamp(mel, min=1e-5))  # (80, T)

        # Transpose to (T, 80)
        log_mel = log_mel.T                        # (T, 80)

        mel_list.append(log_mel.numpy())
        waveform_list.append(audio_array)

    return {"labels": mel_list, "target_waveform": waveform_list}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess_vocoder_data(num_samples=None):
    """
    Prepares a paired dataset for HiFi-GAN vocoder fine-tuning.
    
    Columns:
      - input_values: WavLM hidden states (from English audio)
      - labels: 80-bin log-mel spectrograms (from German audio)
      - target_waveform: 16kHz raw waveform (from German audio)
    """
    print(f"--- Vocoder Preprocessing {SOURCE_LANG} -> {TARGET_LANG} ---")
    print(f"  Source modality : WavLM hidden states ({WAVLM_MODEL_NAME})")
    print(f"  Target modality : 80-bin mel-spectrogram + 16kHz Waveform")
    print(f"  Max Duration    : {MAX_DURATION_SECONDS}s")

    num_proc = 2
    
    # 1. Load Raw Data
    print("Loading raw datasets...")
    datasets = dataset_loader.load_data(
        lang=[SOURCE_LANG, TARGET_LANG],
        split="train",
        dataset=["seamless_align"],
        num_samples=num_samples if num_samples else 15000,
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

    # 4. SOURCE: Encode English audio → WavLM hidden states (input_values)
    print("Encoding SOURCE (EN) with frozen WavLM...")
    source_ds = source_ds.map(
        process_source_wavlm,
        batched=True,
        batch_size=64,
        num_proc=num_proc,
        desc="WavLM Encoding Source Audio",
        remove_columns=["audio"],
    )

    # 5. TARGET: Encode German audio → Mel + Waveform
    print("Encoding TARGET (DE) → 80-bin mel-spectrogram + Waveform...")
    target_ds = target_ds.map(
        process_target_vocoder,
        batched=True,
        batch_size=16,
        num_proc=num_proc,
        desc="Vocoder Encoding Target Audio",
        remove_columns=["audio"],
    )

    # 6. Merge
    print("Merging source hidden states and target vocoder data...")
    target_ds = target_ds.remove_columns([c for c in target_ds.column_names if c in source_ds.column_names])
    
    from datasets import concatenate_datasets
    paired_ds = concatenate_datasets([source_ds, target_ds], axis=1)

    # 7. Save
    out_path = os.path.join(
        OUTPUT_DIR,
        f"processed_vocoder_{SOURCE_LANG}_{TARGET_LANG}_v1",
    )
    print(f"Saving paired dataset to {out_path}...")
    paired_ds.save_to_disk(out_path)

    print(f"\nSUCCESS! Vocoder preprocessing complete.")
    print(f"  Samples      : {len(paired_ds)}")
    print(f"  Columns      : {paired_ds.column_names}")
    print(f"  Output path  : {out_path}")


if __name__ == "__main__":
    # For testing, use a small number of samples if requested or default to 15000
    test_mode = "--test" in sys.argv
    num_samples = 10 if test_mode else 15000
    preprocess_vocoder_data(num_samples=num_samples)
