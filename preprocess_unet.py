"""
preprocess_unet.py
==================
Preprocessing pipeline for the UNet Mel-Spectrogram model.

Loads 10,000 aligned English/German audio pairs from the seamless_align dataset,
filters by duration and language, encodes both sides as 80-bin log-mel spectrograms
using SpectrogramEncoder, and saves the results to disk ready for UNet training.

Output directory layout:
    datasets/processed_spectrogram_unet_en_de_v1/
        en/   ← Arrow dataset, each sample's 'audio' is a (80, T) numpy float32 array
        de/   ← Arrow dataset, each sample's 'audio' is a (80, T) numpy float32 array

Usage:
    python preprocess_unet.py
"""

import os
import sys
import numpy as np
from faster_whisper import WhisperModel

import dataset_loader
from encoders import SpectrogramEncoder

# ---------------------------------------------------------------------------
# Cap dataset_loader's global worker count BEFORE any load_data() call so
# the internal validation maps also respect the 4-worker memory budget.
# ---------------------------------------------------------------------------
dataset_loader.NUM_PROC = 4

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
SOURCE_LANG = "en"
TARGET_LANG = "de"
NUM_SAMPLES = 25000
MAX_DURATION_SECONDS = 8.0
NUM_PROC = 4

OUTPUT_DIR = dataset_loader.DATASETS_DIR
OUTPUT_NAME = f"processed_spectrogram_unet_{SOURCE_LANG}_{TARGET_LANG}_v1"

# SpectrogramEncoder parameters — consistent with encoders.py defaults
N_MELS = 80
N_FFT = 1024
HOP_LENGTH = 256
SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Module-level globals (lazy-loaded per worker process)
# ---------------------------------------------------------------------------
_language_model = None   # WhisperModel instance
_spectrogram_encoder = None  # SpectrogramEncoder instance


def _get_language_model():
    """Lazily load the Whisper tiny model (CPU, int8) for language detection."""
    global _language_model
    if _language_model is None:
        print(f"[PID {os.getpid()}] Loading Whisper tiny model for language detection...")
        _language_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _language_model


def _get_spectrogram_encoder():
    """Lazily initialise SpectrogramEncoder (no weights to load — pure librosa)."""
    global _spectrogram_encoder
    if _spectrogram_encoder is None:
        _spectrogram_encoder = SpectrogramEncoder(
            sample_rate=SAMPLE_RATE,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
    return _spectrogram_encoder


# ---------------------------------------------------------------------------
# Dataset map functions
# ---------------------------------------------------------------------------

def compute_duration(batch):
    """Compute audio duration (seconds) for each sample in a batch."""
    durations = []
    for x in batch["audio"]:
        durations.append(len(x["array"]) / x["sampling_rate"])
    return {"duration": durations}


def check_language(batch, expected_lang):
    """
    Use Whisper to detect the spoken language and flag mismatches.

    Returns:
        {"is_valid_lang": [bool, ...]}  — True if the sample matches expected_lang.
    """
    model = _get_language_model()
    is_valid = []

    for x in batch["audio"]:
        audio = np.array(x["array"], dtype=np.float32)
        sr = x["sampling_rate"]

        # Whisper expects 16 kHz; resample if necessary
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        try:
            _, info = model.transcribe(
                audio,
                beam_size=1,
                language=None,
                task="transcribe",
            )
            is_valid.append(info.language == expected_lang)
        except Exception:
            # On error, keep the sample rather than silently dropping valid data
            is_valid.append(True)

    return {"is_valid_lang": is_valid}


def apply_spectrogram_encoding(batch):
    """
    Encode each audio sample as an 80-bin log-mel spectrogram.

    Input:  batch["audio"] is a list of HuggingFace audio dicts
            (with 'array' and 'sampling_rate' keys).
    Output: {"audio": [np.ndarray of shape (80, T), ...]}
    """
    encoder = _get_spectrogram_encoder()
    encoded = []

    for x in batch["audio"]:
        audio_array = np.array(x["array"], dtype=np.float32)
        sr = x["sampling_rate"]

        # SpectrogramEncoder.encode returns (80, T) float32 ndarray in dB scale
        mel_spec = encoder.encode(audio_array, sr=sr)
        encoded.append(mel_spec.astype(np.float32))

    return {"audio": encoded}


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_and_save():
    """
    Full preprocessing pipeline:
      1. Load seamless_align (en + de, 10 k samples)
      2. Filter by duration (<= 8 s)
      3. Filter by detected language (Whisper)
      4. Encode both sides as 80-bin log-mel spectrograms
      5. Save to disk
    """
    print("=" * 60)
    print("  UNet Mel-Spectrogram Preprocessing")
    print(f"  Source: {SOURCE_LANG}  |  Target: {TARGET_LANG}")
    print(f"  Samples: {NUM_SAMPLES}  |  Max duration: {MAX_DURATION_SECONDS}s")
    print(f"  Workers: {NUM_PROC}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load raw dataset
    # ------------------------------------------------------------------
    print("\n[Step 1] Loading raw datasets from seamless_align...")
    datasets = dataset_loader.load_data(
        lang=[SOURCE_LANG, TARGET_LANG],
        split="train",
        dataset=["seamless_align"],
        num_samples=NUM_SAMPLES,
    )

    source_ds = datasets.get(SOURCE_LANG)
    target_ds = datasets.get(TARGET_LANG)

    if source_ds is None or target_ds is None:
        print("ERROR: Could not load one or both language splits. Aborting.")
        sys.exit(1)

    # Align to equal length
    min_len = min(len(source_ds), len(target_ds))
    source_ds = source_ds.select(range(min_len))
    target_ds = target_ds.select(range(min_len))
    print(f"  Aligned pairs after loading: {min_len}")

    # ------------------------------------------------------------------
    # 2. Filter by duration
    # ------------------------------------------------------------------
    print(f"\n[Step 2] Filtering samples longer than {MAX_DURATION_SECONDS}s...")

    source_ds = source_ds.map(
        compute_duration, batched=True, num_proc=NUM_PROC, desc="Calc source durations"
    )
    target_ds = target_ds.map(
        compute_duration, batched=True, num_proc=NUM_PROC, desc="Calc target durations"
    )

    import numpy as np_local  # local alias to avoid shadow
    src_durs = np_local.array(source_ds["duration"])
    tgt_durs = np_local.array(target_ds["duration"])
    valid_mask = (src_durs <= MAX_DURATION_SECONDS) & (tgt_durs <= MAX_DURATION_SECONDS)
    valid_indices = np_local.where(valid_mask)[0]

    print(f"  Kept {len(valid_indices)} / {min_len} samples after duration filter.")

    source_ds = source_ds.select(valid_indices).remove_columns(["duration"])
    target_ds = target_ds.select(valid_indices).remove_columns(["duration"])

    # ------------------------------------------------------------------
    # 3. Filter by language
    # ------------------------------------------------------------------
    print("\n[Step 3] Filtering by detected language (Whisper)...")

    source_ds = source_ds.map(
        check_language,
        batched=True,
        batch_size=8,
        num_proc=1,  # Whisper model is not fork-safe; use single worker
        fn_kwargs={"expected_lang": SOURCE_LANG},
        desc=f"Checking source language ({SOURCE_LANG})",
    )
    target_ds = target_ds.map(
        check_language,
        batched=True,
        batch_size=8,
        num_proc=1,
        fn_kwargs={"expected_lang": TARGET_LANG},
        desc=f"Checking target language ({TARGET_LANG})",
    )

    src_valid = np_local.array(source_ds["is_valid_lang"])
    tgt_valid = np_local.array(target_ds["is_valid_lang"])
    lang_mask = src_valid & tgt_valid
    lang_valid_indices = np_local.where(lang_mask)[0]

    print(f"  Kept {len(lang_valid_indices)} / {len(source_ds)} samples after language filter.")

    source_ds = source_ds.select(lang_valid_indices).remove_columns(["is_valid_lang"])
    target_ds = target_ds.select(lang_valid_indices).remove_columns(["is_valid_lang"])

    # ------------------------------------------------------------------
    # 4. Encode as mel-spectrograms
    # ------------------------------------------------------------------
    print("\n[Step 4] Encoding audio as 80-bin log-mel spectrograms...")
    print(f"  Parameters: n_mels={N_MELS}, n_fft={N_FFT}, hop_length={HOP_LENGTH}")

    # SpectrogramEncoder is pure numpy/librosa — safe for multi-proc with fork
    source_ds = source_ds.map(
        apply_spectrogram_encoding,
        batched=True,
        batch_size=32,
        num_proc=NUM_PROC,
        desc=f"Encoding {SOURCE_LANG} spectrograms",
    )
    target_ds = target_ds.map(
        apply_spectrogram_encoding,
        batched=True,
        batch_size=32,
        num_proc=NUM_PROC,
        desc=f"Encoding {TARGET_LANG} spectrograms",
    )

    # ------------------------------------------------------------------
    # 5. Save to disk
    # ------------------------------------------------------------------
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    src_out = os.path.join(out_path, SOURCE_LANG)
    tgt_out = os.path.join(out_path, TARGET_LANG)

    print(f"\n[Step 5] Saving processed datasets to:\n  {out_path}")
    source_ds.save_to_disk(src_out)
    target_ds.save_to_disk(tgt_out)

    print("\n" + "=" * 60)
    print("  SUCCESS — Preprocessing complete!")
    print(f"  Source samples : {len(source_ds)}")
    print(f"  Target samples : {len(target_ds)}")
    print(f"  Each 'audio' column holds a (80, T) float32 mel-spectrogram.")
    print("=" * 60)


if __name__ == "__main__":
    preprocess_and_save()
