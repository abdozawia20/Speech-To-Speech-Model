import os
import torch
import numpy as np
from faster_whisper import WhisperModel
import dataset_loader

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
dataset_loader.NUM_PROC = 4
SOURCE_LANG = "en"
TARGET_LANG = "de"
MAX_DURATION_SECONDS = 8.0
OUTPUT_DIR = dataset_loader.DATASETS_DIR

# ---------------------------------------------------------------------------
# Global state for worker processes (language detection only)
# ---------------------------------------------------------------------------
language_model = None

# ---------------------------------------------------------------------------
# Worker utilities — language detection (shared with preprocess_speecht5.py)
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
# Source + Target: WavLM hidden states via WavLMSpeechT5Encoder (encoders.py)
# ---------------------------------------------------------------------------

def process_batch_wavlm(batch):
    """
    Encode audio into WavLM hidden states of shape (Seq_Len, 768).

    Applied to BOTH source and target datasets so the model learns a
    WavLM-feature-space to WavLM-feature-space mapping.

    Uses WavLMSpeechT5Encoder from encoders.py with load_decoder=False so only
    the WavLM backbone is initialised — no SpeechT5 or HiFi-GAN is loaded.
    Model name and processor are controlled by the class constants in
    WavLMSpeechT5Encoder, so nothing is hardcoded here.
    """
    from encoders import WavLMSpeechT5Encoder

    global _wavlm_encoder_cache
    if "_wavlm_encoder_cache" not in globals() or _wavlm_encoder_cache is None:
        try:
            print(f"Loading WavLMSpeechT5Encoder in worker (PID: {os.getpid()})...")
            # load_decoder=False: only the WavLM backbone is needed here;
            # SpeechT5 + HiFi-GAN are used at inference time via encode()/decode().
            _wavlm_encoder_cache = WavLMSpeechT5Encoder(load_decoder=False)
        except Exception as e:
            raise RuntimeError(f"WavLM Encoder failed to load in worker: {e}")

    encoder = _wavlm_encoder_cache
    encoded_list = []

    for audio_item in batch["audio"]:
        audio_array = np.array(audio_item["array"], dtype=np.float32)
        sr = audio_item["sampling_rate"]

        # encode() returns (1, Seq_Len, 768) on CPU
        hidden_tensor = encoder.encode(audio_array, sr)
        hidden = hidden_tensor.squeeze(0).numpy()  # (Seq_Len, 768)
        encoded_list.append(hidden)

    return {"audio": encoded_list}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess_and_save_wavlm():
    """
    Full preprocessing pipeline using WavLM hidden states for BOTH source
    and target audio.

    Source output:  numpy arrays of shape (Seq_Len, 768)  — WavLM hidden states (EN)
    Target output:  numpy arrays of shape (Seq_Len, 768)  — WavLM hidden states (DE)

    The model learns a WavLM-feature-space → WavLM-feature-space mapping.
    At inference time, WavLMSpeechT5Encoder.decode() converts predicted target
    hidden states back to audio via the SpeechT5 decoder + HiFi-GAN vocoder.

    Output directory:
        processed_wavlm_{SOURCE_LANG}_{TARGET_LANG}_v1/
            {SOURCE_LANG}/   ← Arrow dataset with WavLM hidden states
            {TARGET_LANG}/   ← Arrow dataset with WavLM hidden states
    """
    print(f"--- WavLM Preprocessing {SOURCE_LANG} -> {TARGET_LANG} ---")
    print(f"Max Duration: {MAX_DURATION_SECONDS}s")

    # NOTE: num_proc=1 is safest when loading heavy neural net models inside
    # worker processes via fork. Increase cautiously.
    num_proc = 1
    print(f"Using {num_proc} CPU worker(s) (conservative for WavLM).")

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

    # 3. Filter by Language (Whisper Detection)
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

    # 4. Encode Source and Target with WavLM hidden states
    print("Encoding Source with WavLM (hidden states)...")
    source_ds = source_ds.map(
        process_batch_wavlm,
        batched=True,
        batch_size=16,
        num_proc=num_proc,
        desc="WavLM Encoding Source Audio",
    )

    print("Encoding Target with WavLM (hidden states)...")
    target_ds = target_ds.map(
        process_batch_wavlm,
        batched=True,
        batch_size=16,
        num_proc=num_proc,
        desc="WavLM Encoding Target Audio",
    )

    # 5. Save to Disk
    out_path = os.path.join(
        OUTPUT_DIR,
        f"processed_wavlm_{SOURCE_LANG}_{TARGET_LANG}_v1",
    )
    print(f"Saving to {out_path}...")
    source_ds.save_to_disk(os.path.join(out_path, SOURCE_LANG))
    target_ds.save_to_disk(os.path.join(out_path, TARGET_LANG))

    print("SUCCESS! WavLM preprocessing complete.")


if __name__ == "__main__":
    preprocess_and_save_wavlm()
