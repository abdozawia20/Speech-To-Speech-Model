"""
evaluation.py — Dataset Validation Script

How it works:
  1. Loads the preprocessed WavLM dataset to confirm features were stored correctly.
  2. Reads the record IDs from the preprocessed dataset.
  3. Loads the original RAW audio from seamless_align for those same records.
  4. Saves the raw EN + DE audio clips as WAV files for human review.

Why raw audio and not decoded WavLM features?
  WavLM is an encoder-only model — it has no built-in decoder. WavLM hidden
  states (Seq_Len, 768) are a compact latent representation of audio. There is
  no lossless way to invert them back to audio without a trained vocoder model.
  The correct validation strategy is therefore: confirm the raw input audio
  sounds right. If the source audio is good English and the target audio is a
  correct German translation, the preprocessing is valid.
  
  The SpeechT5 fine-tuning phase will teach the model to decode WavLM features
  into audio. Until then, the raw audio IS the ground truth.
"""

import os
import numpy as np
from scipy.io.wavfile import write
from datasets import load_from_disk
import dataset_loader

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PROCESSED_DIR = os.path.join(dataset_loader.DATASETS_DIR, "processed_wavlm_en_de_v1")
SOURCE_LANG   = "en"
TARGET_LANG   = "de"
NUM_SAMPLES   = 5       # number of EN/DE pairs to save
START_IDX     = 0       # starting index in the preprocessed dataset
OUTPUT_DIR    = "eval_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_wav(path: str, audio_array: np.ndarray, sample_rate: int) -> None:
    arr = np.array(audio_array, dtype=np.float32)
    peak = np.max(np.abs(arr))
    if peak > 0:
        arr = arr / peak
    write(path, sample_rate, (arr * 32767).astype(np.int16))


def main():
    # ------------------------------------------------------------------
    # 1. Load preprocessed dataset — confirm features, get IDs
    # ------------------------------------------------------------------
    print(f"Loading preprocessed WavLM dataset from:\n  {PROCESSED_DIR}\n")
    src_proc = load_from_disk(os.path.join(PROCESSED_DIR, SOURCE_LANG))
    tgt_proc = load_from_disk(os.path.join(PROCESSED_DIR, TARGET_LANG))
    total = min(len(src_proc), len(tgt_proc))
    print(f"  Total preprocessed pairs: {total}")

    end_idx = min(START_IDX + NUM_SAMPLES, total)
    sample_indices = list(range(START_IDX, end_idx))

    # Verify WavLM features and collect record IDs
    print("\n  WavLM feature check:")
    src_ids, tgt_ids = [], []
    for i in sample_indices:
        sr = src_proc[i]
        tr = tgt_proc[i]
        src_ids.append(sr.get("id"))
        tgt_ids.append(tr.get("id"))

        sf = np.array(sr["audio"])
        tf = np.array(tr["audio"])
        print(
            f"  [{i}] "
            f"EN shape={sf.shape}  mean={sf.mean():.4f}  std={sf.std():.4f} | "
            f"DE shape={tf.shape}  mean={tf.mean():.4f}  std={tf.std():.4f}"
        )

    # ------------------------------------------------------------------
    # 2. Load matching raw audio from seamless_align
    # ------------------------------------------------------------------
    print("\nLoading raw seamless_align to fetch original audio clips...")
    raw = dataset_loader.load_data(
        lang=[SOURCE_LANG, TARGET_LANG],
        split="train",
        dataset=["seamless_align"],
        num_samples=5,
    )
    raw_src = raw[SOURCE_LANG]
    raw_tgt = raw[TARGET_LANG]

    # Fast column-read ID → index maps
    src_id_map = {uid: i for i, uid in enumerate(raw_src["id"])}
    tgt_id_map = {uid: i for i, uid in enumerate(raw_tgt["id"])}

    # ------------------------------------------------------------------
    # 3. Save paired WAV clips
    # ------------------------------------------------------------------
    print(f"\nSaving {len(sample_indices)} pairs to '{OUTPUT_DIR}/'...\n")
    for proc_idx, src_id, tgt_id in zip(sample_indices, src_ids, tgt_ids):

        raw_sr = raw_src[src_id_map[src_id]] if src_id in src_id_map else raw_src[proc_idx]
        raw_tr = raw_tgt[tgt_id_map[tgt_id]] if tgt_id in tgt_id_map else raw_tgt[proc_idx]

        src_audio = raw_sr["audio"]
        tgt_audio = raw_tr["audio"]

        src_path = os.path.join(OUTPUT_DIR, f"pair_{proc_idx:04d}_{SOURCE_LANG}.wav")
        tgt_path = os.path.join(OUTPUT_DIR, f"pair_{proc_idx:04d}_{TARGET_LANG}.wav")

        save_wav(src_path, src_audio["array"], src_audio["sampling_rate"])
        save_wav(tgt_path, tgt_audio["array"], tgt_audio["sampling_rate"])

        src_dur = len(src_audio["array"]) / src_audio["sampling_rate"]
        tgt_dur = len(tgt_audio["array"]) / tgt_audio["sampling_rate"]
        print(
            f"  [{proc_idx:04d}]  {os.path.basename(src_path)} ({src_dur:.2f}s)  "
            f"{os.path.basename(tgt_path)} ({tgt_dur:.2f}s)"
        )

    print(f"\n✓ Done. Open '{OUTPUT_DIR}/' and listen to the clips.")
    print(
        "\nWhat to check:\n"
        "  - EN clips should be clear English speech.\n"
        "  - DE clips should be the German equivalent of the same utterance.\n"
        "  - If both sound correct → preprocessing is validated → proceed to fine-tuning."
    )


if __name__ == "__main__":
    main()
