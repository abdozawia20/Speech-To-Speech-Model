import sys
import os
import collections
import concurrent.futures
import torch
import numpy as np
import soundfile as sf
import hashlib
import io
import traceback
import aiohttp
from datasets import load_dataset, concatenate_datasets, DownloadConfig, IterableDataset, Dataset, load_from_disk, load_dataset_builder
from IPython.display import Audio
from encoders import *

project_root = os.path.abspath(os.path.join(os.getcwd(), ''))
sys.path.append(project_root)

# ---------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------------------------------
NUM_PROC = 1
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
CVSS_LANGS = ['ar', 'ca', 'cy', 'de', 'el', 'es', 'et', 'fa', 'fr', 'id', 'it', 'ja', 'lv', 'mn', 'nl', 'pt', 'ru', 'sl', 'sv', 'tr', 'zh']

# ---------------------------------------------------------------------------
# SEAMLESS-ALIGN LANGUAGE CONFIGURATION
# ---------------------------------------------------------------------------
_SEAMLESS_S2S_LANGS = [
    'ar', 'bg', 'ca', 'cs', 'cy', 'da', 'de', 'el',
    'es', 'et', 'fi', 'fr', 'ga', 'gl', 'hr', 'hu',
    'hy', 'id', 'it', 'ja', 'ko', 'lt', 'lv', 'mk',
    'ml', 'mt', 'nl', 'pl', 'pt', 'ro', 'ru', 'sk',
    'sl', 'sr', 'sv', 'sw', 'ta', 'te', 'th', 'tr',
    'uk', 'ur', 'uz', 'vi', 'zh',
]

def _build_pair_mapping(lang_list):
    """
    Generates PAIR_MAPPING for any subset of _SEAMLESS_S2S_LANGS.
    Config name = alphabetically sorted 2-letter codes joined by '-',
    each suffixed with 'A'.  e.g. de+en → 'deA-enA', en+es → 'enA-esA'.
    """
    mapping = {}
    for lang in lang_list:
        key = tuple(sorted((lang, 'en')))
        cfg = f"{key[0]}A-{key[1]}A"
        mapping[key] = cfg
    return mapping

SEAMLESS_PAIR_MAPPING = _build_pair_mapping(_SEAMLESS_S2S_LANGS)
_SEAMLESS_EXPRESSIVE_LANGS = ['de', 'es', 'fr', 'it', 'zh']
SEAMLESS_EXPRESSIVE_PAIR_MAPPING = _build_pair_mapping(_SEAMLESS_EXPRESSIVE_LANGS)

# Audio quality thresholds
MIN_DURATION_S  = 1.0   # discard clips shorter than 1 second (too short for LID/speech)
MAX_DURATION_S  = 20.0  # discard clips longer than 20 s (very long utterances hurt seq models)
LID_THRESHOLD   = 0.75  # Whisper language probability threshold
LID_BATCH_SIZE  = 64    # samples per LID batch — feed the thread pool generously

# Force Whisper LID to run on CPU regardless of GPU availability.
WHISPER_FORCE_CPU = False

# Multi Threading
LID_NUM_THREADS = min(8, os.cpu_count() or 4)

# ---------------------------------------------------------------------------
# WORKER GLOBALS (Lazy-loaded per process)
# ---------------------------------------------------------------------------
_shared_language_model = None

def _get_whisper_model(model_size="tiny"):
    """
    Lazily loads the Whisper model for language identification.

    Always uses CPU (int8) when WHISPER_FORCE_CPU=True (default).
    The GPU is typically occupied by the main speech/translation model;
    loading Whisper on the same GPU causes VRAM OOM and kills the kernel.
    Set WHISPER_FORCE_CPU=False only if you have a dedicated second GPU
    with free VRAM.
    """
    global _shared_language_model
    if _shared_language_model is None:
        try:
            # Fix for ctranslate2/faster-whisper not finding cuDNN 9 libraries on Linux
            import platform
            if platform.system() != "Windows":
                try:
                    import nvidia.cudnn
                    import nvidia.cublas
                    import ctypes
                    
                    cudnn_path = os.path.join(nvidia.cudnn.__path__[0], 'lib')
                    cublas_path = os.path.join(nvidia.cublas.__path__[0], 'lib')
                    
                    # Load cuDNN libraries with RTLD_GLOBAL
                    if os.path.exists(cudnn_path):
                        for lib in os.listdir(cudnn_path):
                            if "libcudnn" in lib and ".so.9" in lib:
                                 try:
                                      ctypes.CDLL(os.path.join(cudnn_path, lib), mode=ctypes.RTLD_GLOBAL)
                                 except Exception:
                                      pass

                    # Load cuBLAS libraries with RTLD_GLOBAL
                    if os.path.exists(cublas_path):
                        try:
                             ctypes.CDLL(os.path.join(cublas_path, "libcublas.so.12"), mode=ctypes.RTLD_GLOBAL)
                             ctypes.CDLL(os.path.join(cublas_path, "libcublasLt.so.12"), mode=ctypes.RTLD_GLOBAL)
                        except Exception:
                            pass

                except ImportError:
                    pass # Optional nvidia libs not installed
                except Exception as e:
                    print(f"Warning: Error preloading nvidia libraries: {e}")

            from faster_whisper import WhisperModel
            if torch.cuda.is_available() and not WHISPER_FORCE_CPU:
                device, compute_type = "cuda", "float16"
                cpu_threads, num_workers = 1, 1
                print(f"[PID {os.getpid()}] Loading Whisper-{model_size} on GPU (float16)...")
            else:
                device, compute_type = "cpu", "int8"
                cpu_threads  = max(1, LID_NUM_THREADS // 2)
                num_workers  = LID_NUM_THREADS
                print(f"[PID {os.getpid()}] Loading Whisper-{model_size} on CPU "
                      f"(int8, {num_workers} workers × {cpu_threads} threads)...")
            _shared_language_model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
            )
        except ImportError:
            print("Warning: faster-whisper not installed. Language filtering will be disabled.")
            return None
        except Exception as e:
            print(f"Warning: failed to load Whisper ({e}). Language filtering will be disabled.")
            return None
    return _shared_language_model

# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------

def generate_id_from_string(s):
    """Consistently generates a 64-bit integer ID from a string using MD5."""
    try:
        return int(s)
    except (ValueError, TypeError):
        return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:16], 16)

def _decode_audio(audio) -> np.ndarray:
    """Normalise an HF audio field to a 16 kHz float32 numpy array."""
    try:
        if isinstance(audio, dict):
            if audio.get('array') is not None:
                arr = np.array(audio['array'], dtype=np.float32)
                sr  = audio.get('sampling_rate', 16000)
            elif audio.get('bytes') is not None:
                arr, sr = sf.read(io.BytesIO(audio['bytes']))
                arr = arr.astype(np.float32)
                if arr.ndim > 1: arr = arr.mean(axis=1)  # stereo → mono
            else:
                return None, None
        else:
            return None, None

        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        return arr, 16000
    except Exception:
        return None, None

# ---------------------------------------------------------------------------
# DATASET MAP FUNCTIONS (Batched)
# ---------------------------------------------------------------------------

def compute_duration_batch(batch):
    """Batch calculation of audio duration in seconds."""
    durations = []
    for x in batch["audio"]:
        try:
            durations.append(len(x['array']) / x['sampling_rate'])
        except Exception:
            durations.append(0.0)
    return {"duration": durations}

def _lid_single(audio_field, expected_lang, threshold):
    """
    Run Whisper LID on one audio clip.  Designed to be called from a
    ThreadPoolExecutor — CTranslate2 releases the Python GIL during
    matrix operations, so N threads give ~N× throughput on CPU.

    Returns True  → clip matches expected_lang above threshold
            False → mismatch, too short/long, or decode error
    """
    try:
        arr, _ = _decode_audio(audio_field)
        if arr is None:
            return False
        dur = len(arr) / 16000
        if dur < MIN_DURATION_S or dur > MAX_DURATION_S:
            return False
        model = _get_whisper_model()
        if model is None:
            return True   # conservative: keep when model unavailable
        _, info = model.transcribe(arr, beam_size=1, task="transcribe",
                                   without_timestamps=True)
        return (info.language == expected_lang and
                info.language_probability >= threshold)
    except Exception:
        return True  # conservative on unexpected errors


def check_language_batch(batch, expected_lang, threshold=LID_THRESHOLD):
    """
    Detects language using Whisper for every clip in the batch.
    Compatible with datasets.map(batched=True).
    """
    if _get_whisper_model() is None:
        return {"is_valid_lang": [True] * len(batch["audio"])}

    with concurrent.futures.ThreadPoolExecutor(max_workers=LID_NUM_THREADS) as pool:
        futures = [
            pool.submit(_lid_single, af, expected_lang, threshold)
            for af in batch["audio"]
        ]
        is_valid = [f.result() for f in futures]

    return {"is_valid_lang": is_valid}

# ---------------------------------------------------------------------------
# QUALITY FILTER (duration + audio validity)
# ---------------------------------------------------------------------------

def quality_filter_batch(batch,
                          min_dur=MIN_DURATION_S,
                          max_dur=MAX_DURATION_S):
    """
    Fast pre-filter: removes silent/corrupt/too-short/too-long clips
    WITHOUT running Whisper. Intended as a first pass before LID.
    """
    keep = []
    for audio_field in batch["audio"]:
        try:
            arr, sr = _decode_audio(audio_field)
            if arr is None or len(arr) == 0:
                keep.append(False)
                continue
            dur = len(arr) / 16000
            # Duration gate
            if dur < min_dur or dur > max_dur:
                keep.append(False)
                continue
            # Silence gate: RMS < -60 dBFS treated as empty
            rms = float(np.sqrt(np.mean(arr ** 2)))
            keep.append(rms > 1e-3)
        except Exception:
            keep.append(False)
    return {"quality_ok": keep}

# ---------------------------------------------------------------------------
# PAIRED LID FILTER (filters src+tgt together so IDs stay aligned)
# ---------------------------------------------------------------------------

def verify_paired_language(src_ds, tgt_ds,
                            src_lang, tgt_lang,
                            threshold=LID_THRESHOLD):
    """
    Runs LID on both sides of a parallel corpus and returns only pairs
    where BOTH src and tgt pass their respective language checks.

    Args:
        src_ds:   HF Dataset with 'id' and 'audio' columns (source side)
        tgt_ds:   HF Dataset with 'id' and 'audio' columns (target side)
        src_lang: expected ISO-639-1 language code for src_ds
        tgt_lang: expected ISO-639-1 language code for tgt_ds
        threshold: Whisper language probability threshold

    Returns:
        (filtered_src_ds, filtered_tgt_ds) with only valid pairs
    """
    print(f"Paired LID: verifying ({src_lang}, {tgt_lang}) — "
          f"{len(src_ds)} pairs...")

    fn_kwargs_src = {"expected_lang": src_lang, "threshold": threshold}
    fn_kwargs_tgt = {"expected_lang": tgt_lang, "threshold": threshold}

    # num_proc=1 is intentional: threading happens *inside* check_language_batch
    src_mask = src_ds.map(
        check_language_batch, batched=True, batch_size=LID_BATCH_SIZE,
        num_proc=1, fn_kwargs=fn_kwargs_src,
        remove_columns=src_ds.column_names,
        desc=f"LID({src_lang})",
    )
    tgt_mask = tgt_ds.map(
        check_language_batch, batched=True, batch_size=LID_BATCH_SIZE,
        num_proc=1, fn_kwargs=fn_kwargs_tgt,
        remove_columns=tgt_ds.column_names,
        desc=f"LID({tgt_lang})",
    )

    # Keep only indices where BOTH sides pass
    valid_indices = [
        i for i, (s, t) in enumerate(
            zip(src_mask["is_valid_lang"], tgt_mask["is_valid_lang"])
        )
        if s and t
    ]

    n_total  = len(src_ds)
    n_kept   = len(valid_indices)
    n_src_fail = sum(1 for v in src_mask["is_valid_lang"] if not v)
    n_tgt_fail = sum(1 for v in tgt_mask["is_valid_lang"] if not v)
    print(
        f"Paired LID ({src_lang}↔{tgt_lang}): "
        f"kept {n_kept}/{n_total} pairs | "
        f"src_fail={n_src_fail}, tgt_fail={n_tgt_fail}"
    )

    return src_ds.select(valid_indices), tgt_ds.select(valid_indices)

# ---------------------------------------------------------------------------
# DATASET LOADING INTERNALS
# ---------------------------------------------------------------------------

def transform_internal(batch):
    batch['id'] = int(batch['id'])
    batch['gender'] = str(batch['gender'])
    batch['language'] = str(batch['language'])
    return batch

def _get_fleurs_config(lang):
    mapping = {
        'en': 'en_us', 'de': 'de_de', 'ar': 'ar_eg', 'tr': 'tr_tr',
        'fr': 'fr_fr', 'es': 'es_419', 'it': 'it_it', 'zh': 'cmn_hans_cn',
    }
    return mapping.get(lang, lang)

def _get_seamless_config(lang, context_langs=None, expressive=False):
    """
    Returns (config_name, is_paired) for SeamlessAlign.

    Args:
        lang:          2-letter ISO code of the language to load
        context_langs: other languages being loaded in the same call
                       (used to pick the right bilingual config)
        expressive:    if True, use the 5-pair expressive variant;
                       if False (default), use the full 35-pair dataset.
    """
    pair_mapping   = SEAMLESS_EXPRESSIVE_PAIR_MAPPING if expressive else SEAMLESS_PAIR_MAPPING
    supported_langs = {code for key in pair_mapping for code in key}

    lang_code = lang[:2]
    if lang_code not in supported_langs:
        return None, False

    # If we know the peer language (from context), pick the exact config
    if context_langs:
        ctx_codes = {l[:2] for l in context_langs} - {lang_code}
        for other_code in ctx_codes:
            pair_key = tuple(sorted((lang_code, other_code)))
            if pair_key in pair_mapping:
                return pair_mapping[pair_key], True

    # Fallback: if lang is non-English, pick the en pair
    if lang_code != 'en':
        pair_key = tuple(sorted((lang_code, 'en')))
        if pair_key in pair_mapping:
            return pair_mapping[pair_key], True

    # English with no peer specified — return first available pair
    for key, cfg in pair_mapping.items():
        if 'en' in key:
            return cfg, True

    return None, False

def _process_seamless_dataset(ds, lang_code, config_name,
                               start_idx=0, num_samples=None):
    """
    Given a raw WebDataset shard (src.tar.gz + tgt.tar.gz merged),
    selects the correct side (src or tgt) for lang_code and returns
    a normalised HF Dataset with columns:
        id, audio, language, gender, transcription
    """
    if ds is None or 'mp3' not in ds.column_names:
        return ds

    try:
        parts    = config_name.split('-')
        src_lang = parts[0][:2]
        tgt_lang = parts[1][:2]
    except Exception:
        print(f"Cannot parse config '{config_name}' — skipping.")
        return None

    if lang_code == src_lang:
        prefix = 'src/'
    elif lang_code == tgt_lang:
        prefix = 'tgt/'
    else:
        print(f"lang_code {lang_code!r} not in config '{config_name}'.")
        return None

    all_keys   = ds['__key__']
    src_indices, tgt_indices = {}, {}
    for idx, k in enumerate(all_keys):
        if k.startswith('src/'):
            src_indices[k[4:]] = idx
        elif k.startswith('tgt/'):
            tgt_indices[k[4:]] = idx

    common_ids = sorted(set(src_indices) & set(tgt_indices))
    if prefix == 'src/':
        indices = [src_indices[uid] for uid in common_ids]
    else:
        indices = [tgt_indices[uid] for uid in common_ids]

    if num_samples is not None:
        indices = indices[start_idx:start_idx + num_samples]

    ds = ds.select(indices)

    def map_seamless_cols(batch):
        clean_id = batch['__key__'].replace(prefix, '').replace('/', '')
        return {
            'id':           generate_id_from_string(clean_id),
            'audio':        batch['mp3'],
            'language':     lang_code,
            'gender':       'unknown',
            'transcription': '',
        }

    return ds.map(map_seamless_cols,
                  remove_columns=['mp3', '__key__', '__url__'])

def _process_seamless_dataset_paired(ds, config_name,
                                      start_idx=0, num_samples=None):
    """
    Extracts BOTH src and tgt sides from a raw SeamlessAlign shard as
    **lazy** (index-mapped) HF Datasets.

    Returns:
        (src_raw, tgt_raw, src_lang, tgt_lang)
        src_raw / tgt_raw each have columns: ['audio', '__key__', '__url__']
        where 'audio' is the raw MP3 bytes dict (NOT yet decoded).
        Returns (None, None, src_lang, tgt_lang) on failure.
    """
    if ds is None or 'mp3' not in ds.column_names:
        return None, None, None, None

    try:
        parts    = config_name.split('-')
        src_lang = parts[0][:2]
        tgt_lang = parts[1][:2]
    except Exception:
        print(f"Cannot parse config '{config_name}' — skipping.")
        return None, None, None, None

    # Read ONLY the __key__ string column — very fast, no audio decoding
    all_keys = ds['__key__']
    src_map, tgt_map = {}, {}
    for idx, k in enumerate(all_keys):
        if k.startswith('src/'):
            src_map[k[4:]] = idx
        elif k.startswith('tgt/'):
            tgt_map[k[4:]] = idx

    common_ids = sorted(set(src_map) & set(tgt_map))
    if num_samples is not None:
        common_ids = common_ids[start_idx : start_idx + num_samples]

    if not common_ids:
        print(f"[SeamlessAlign] No common pairs found in '{config_name}'.")
        return None, None, src_lang, tgt_lang

    src_indices = [src_map[uid] for uid in common_ids]
    tgt_indices = [tgt_map[uid] for uid in common_ids]

    # ds.select() returns an index-mapped view: LAZY, zero bytes copied.
    # rename_column() only changes the schema metadata: also LAZY.
    src_raw = ds.select(src_indices).rename_column('mp3', 'audio')
    tgt_raw = ds.select(tgt_indices).rename_column('mp3', 'audio')

    return src_raw, tgt_raw, src_lang, tgt_lang

def _load_or_download_generic(dataset_key, hf_dataset_name, hf_config,
                               local_config_name, split, data_files=None,
                               num_samples=None, start_idx=0, **kwargs):
    cache_dir  = os.path.join(DATASETS_DIR, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(DATASETS_DIR, dataset_key, local_config_name, split)

    if os.path.exists(local_path):
        print(f"Loading {hf_dataset_name} ({local_config_name}) from local storage...")
        ds = load_from_disk(local_path)
        if num_samples is not None and dataset_key not in ('seamless_align', 'cvss', 'cv4'):
            ds = ds.select(range(start_idx, min(start_idx + num_samples, len(ds)))) \
                 if start_idx < len(ds) else ds.select([])
        return ds

    token = (kwargs.get('token')
             or os.environ.get('HF_TOKEN')
             or os.environ.get('HUGGING_FACE_HUB_TOKEN'))

    size_info = "Size info unavailable"
    try:
        builder = load_dataset_builder(hf_dataset_name, hf_config,
                                       trust_remote_code=True,
                                       cache_dir=cache_dir, token=token,
                                       **kwargs)
        dl_s, ds_s = builder.info.download_size, builder.info.dataset_size
        if dl_s and ds_s:
            size_info = (f"Download: {dl_s/1024**3:.2f} GB, "
                         f"Generated: {ds_s/1024**3:.2f} GB")
    except Exception:
        pass

    print(f"Dataset {hf_dataset_name} ({local_config_name}) not found locally. {size_info}")

    if sys.stdin.isatty():
        ans = input(f"Proceed with download of {dataset_key}? (Y/n): ").strip().lower()
        if ans not in ('', 'y', 'yes'):
            return None
    else:
        print(f"Non-interactive environment: Proceeding with download of {dataset_key}...")

    print(f"Downloading {hf_dataset_name}...")
    dl_config = DownloadConfig(
        resume_download=True,
        max_retries=20,
        num_proc=NUM_PROC,
        storage_options={"client_kwargs": {
            "timeout": aiohttp.ClientTimeout(total=3600)
        }},
    )
    try:
        dataset = load_dataset(
            hf_dataset_name, hf_config, split=split,
            trust_remote_code=True, download_config=dl_config,
            cache_dir=cache_dir, data_files=data_files,
            token=token, **kwargs
        )
        print(f"Saving to {local_path}...")
        dataset.save_to_disk(local_path)
        if num_samples is not None and dataset_key not in ('seamless_align', 'cvss', 'cv4'):
            dataset = dataset.select(
                range(start_idx, min(start_idx + num_samples, len(dataset)))
            ) if start_idx < len(dataset) else dataset.select([])
        return dataset
    except Exception as e:
        print(f"FAILED to download/load {hf_dataset_name}: {e}")
        if "gated" in str(e).lower() or "403" in str(e) or "access" in str(e).lower():
            print("ERROR: This dataset is GATED. "
                  "Run `huggingface-cli login` or set HF_TOKEN.")
        else:
            traceback.print_exc()
        return None

# ---------------------------------------------------------------------------
# DATASET LOADER FUNCTIONS
# ---------------------------------------------------------------------------

def _load_fleurs_data(split, lang_list, start_idx, num_samples, **kwargs):
    datasets_dict = {}
    if lang_list is None:
        return datasets_dict
    cols = ['id', 'audio', 'transcription', 'language', 'gender']

    for lang in lang_list:
        cfg = _get_fleurs_config(lang)
        ds  = _load_or_download_generic(
            'fleurs', 'google/fleurs', cfg, cfg, split,
            num_samples=num_samples, start_idx=start_idx, **kwargs
        )
        if ds:
            datasets_dict[lang] = ds.map(
                transform_internal,
                remove_columns=[c for c in ds.features
                                if c not in cols and c != 'audio'],
                num_proc=NUM_PROC,
            )
    return datasets_dict


# ---------------------------------------------------------------------------
# LAZY FILTER HELPERS  (no Arrow materialisation until _finalize_seamless_side)
# ---------------------------------------------------------------------------

def _quality_filter_indices(ds, audio_col='audio', batch_size=128):
    """
    Scans audio in mini-batches and returns a list of valid row indices that
    pass duration and silence checks.  Decoded audio arrays are discarded
    after each batch — peak RAM is O(batch_size * avg_audio_size) only.
    Does NOT write any Arrow cache file.
    """
    valid = []
    n = len(ds)
    for start in range(0, n, batch_size):
        batch = ds[start : min(start + batch_size, n)]
        for j, audio_field in enumerate(batch[audio_col]):
            try:
                arr, _ = _decode_audio(audio_field)
                if arr is None or len(arr) == 0:
                    continue
                dur = len(arr) / 16000
                if dur < MIN_DURATION_S or dur > MAX_DURATION_S:
                    continue
                if float(np.sqrt(np.mean(arr ** 2))) > 1e-3:
                    valid.append(start + j)
            except Exception:
                pass  # reject corrupt / undecodeable audio
    return valid


def _lid_filter_indices(ds, expected_lang, audio_col='audio',
                        batch_size=1,           # kept for API compat — ignored internally
                        threshold=LID_THRESHOLD,
                        lid_secs=5.0):
    """
    Runs Whisper detect_language on each row and returns a list of valid
    row indices where the detected language matches expected_lang.

    Does NOT write any Arrow cache file.
    """
    model = _get_whisper_model()
    if model is None:
        print(f"  [LID] Whisper unavailable — keeping all {len(ds)} samples.")
        return list(range(len(ds)))

    valid = []
    n = len(ds)
    lid_samples = int(lid_secs * 16000)  # max samples to feed Whisper

    from tqdm import tqdm
    for i in tqdm(range(n), desc=f"LID ({expected_lang})"):
        try:
            # Fetch EXACTLY ONE row — avoids batching decoded audio in RAM
            audio_field = ds[i][audio_col]

            arr, _ = _decode_audio(audio_field)
            audio_field = None   # release immediately

            if arr is None:
                valid.append(i)  # keep on decode error (conservative)
                continue

            dur = len(arr) / 16000
            if not (MIN_DURATION_S <= dur <= MAX_DURATION_S):
                arr = None
                continue  # already handled by quality filter

            # Trim to lid_secs — language ID is reliable from short clips
            if len(arr) > lid_samples:
                arr = arr[:lid_samples].copy()   # .copy() frees the tail immediately

            # detect_language: encoder-only, ~5-10x faster than transcribe
            language, probs = model.detect_language(arr)
            arr = None   # release decoded audio before next iteration

            prob = float(probs.get(expected_lang, 0.0))
            if language == expected_lang and prob >= threshold:
                valid.append(i)

        except Exception:
            valid.append(i)  # conservative: keep on error

    return valid



def _crosslang_duplicate_filter_indices(src_ds, tgt_ds,
                                        audio_col='audio',
                                        batch_size=64,
                                        waveform_threshold=0.85,
                                        spectrum_threshold=0.82,
                                        check_sr=4000,
                                        max_check_secs=3.0):
    """
    Rejects pairs where src and tgt audio are the same or nearly identical
    recording.  This is a common artifact in web-mined parallel corpora
    (e.g. SeamlessAlign) where the mining pipeline assigns the same speech
    file to both sides of a pair.

    Two complementary checks are applied:
    Check 1 — Lag-0 waveform cosine similarity (fast)
    Check 2 — Magnitude spectrum cosine similarity (shift/speed-invariant)

    A pair is rejected if EITHER check fires.

    Typical similarity values (measured on real SeamlessAlign deA-enA shard)
    -------------------------------------------------------------------------
    Scenario                               | waveform | spectrum
    -------------------------------------- | -------- | --------
    Same file, same volume                 | ~1.00    | ~1.00
    Same file, different volume            | ~1.00    | ~1.00
    Same file, 50-100 ms time offset       | ~0.04    | ~0.90
    Same file, ~3 %% speed difference      | ~0.04    | ~0.90
    Different content, same speaker        | ~0.00-0.15 | ~0.50-0.65
    Genuine translation pair               | ~0.00-0.05 | ~0.30-0.55

    Returns a list of row indices to KEEP to ensure the cleanest dataset.
    """
    check_samples     = int(check_sr * max_check_secs)
    downsample_factor = max(1, 16000 // check_sr)
    min_samples       = int(check_sr * 0.5)

    valid      = []
    n          = min(len(src_ds), len(tgt_ds))
    n_rejected = 0

    for start in range(0, n, batch_size):
        end       = min(start + batch_size, n)
        src_batch = src_ds[start:end]
        tgt_batch = tgt_ds[start:end]

        for j, (src_audio, tgt_audio) in enumerate(
                zip(src_batch[audio_col], tgt_batch[audio_col])):
            row = start + j
            try:
                src_arr, _ = _decode_audio(src_audio)
                tgt_arr, _ = _decode_audio(tgt_audio)

                if src_arr is None or tgt_arr is None:
                    valid.append(row)
                    continue

                # Decimate to check_sr (no anti-aliasing — fast, sufficient)
                src_down = src_arr[::downsample_factor].astype(np.float32)
                tgt_down = tgt_arr[::downsample_factor].astype(np.float32)

                min_len = min(len(src_down), len(tgt_down), check_samples)
                if min_len < min_samples:
                    valid.append(row)   # too short — keep conservatively
                    continue

                src_seg = src_down[:min_len]
                tgt_seg = tgt_down[:min_len]

                # RMS normalise — makes both checks volume-invariant
                src_rms = float(np.sqrt(np.mean(src_seg ** 2)))
                tgt_rms = float(np.sqrt(np.mean(tgt_seg ** 2)))
                if src_rms < 1e-6 or tgt_rms < 1e-6:
                    valid.append(row)
                    continue

                src_norm = src_seg / src_rms
                tgt_norm = tgt_seg / tgt_rms

                # ── Check 1: lag-0 waveform similarity ───────────────────
                cos_sim = float(np.mean(src_norm * tgt_norm))
                if abs(cos_sim) >= waveform_threshold:
                    n_rejected += 1
                    continue

                # ── Check 2: magnitude spectrum similarity ────────────────
                src_spec = np.abs(np.fft.rfft(src_norm))
                tgt_spec = np.abs(np.fft.rfft(tgt_norm))
                src_spec = src_spec / (np.linalg.norm(src_spec) + 1e-8)
                tgt_spec = tgt_spec / (np.linalg.norm(tgt_spec) + 1e-8)
                spec_sim = float(np.dot(src_spec, tgt_spec))

                if spec_sim >= spectrum_threshold:
                    n_rejected += 1
                else:
                    valid.append(row)

            except Exception:
                valid.append(row)   # keep on error (conservative)

    if n_rejected:
        print(f"  Dedup: dropped {n_rejected} same-audio pairs "
              f"(waveform>={waveform_threshold} OR spectrum>={spectrum_threshold}).")
    return valid


def _finalize_seamless_side(raw_ds, lang_code, prefix, audio_col='audio'):
    """
    Performs the final column rename/add on an already-filtered (small) dataset.

    Input columns:  [audio_col, '__key__', '__url__', ...]
    Output columns: ['id', 'audio', 'language', 'gender', 'transcription']
    """
    keys = raw_ds['__key__']
    ids  = [
        generate_id_from_string(k.replace(prefix, '').replace('/', ''))
        for k in keys
    ]

    def _map(batch, indices):
        n = len(batch[audio_col])
        return {
            'id':            [ids[i] for i in indices],
            'audio':         batch[audio_col],
            'language':      [lang_code] * n,
            'gender':        ['unknown'] * n,
            'transcription': [''] * n,
        }

    cols_to_drop = [c for c in raw_ds.column_names if c != audio_col]
    return raw_ds.map(
        _map,
        batched=True,
        with_indices=True,
        remove_columns=cols_to_drop,
        num_proc=1,           # must be 1: _map captures the ids list
        desc=f"Finalising {lang_code}",
    )


def _load_seamless_data(split, lang_list, start_idx, num_samples,
                         expressive=False,
                         use_tgt_lid=True,
                         use_paired_lid=False,
                         **kwargs):
    """
    Load SeamlessAlign (or SeamlessAlign-Expressive) for the requested languages.

    Key filtering strategy for web-scraped data
    -------------------------------------------
    1. Quality pre-filter  — duration gates + silence detection (cheap, CPU)
    2. Cross-lingual dedup — spectral fingerprint check to detect duplicates
    3. Target-side LID     — Whisper detect_language on the TARGET (non-English) side only. For detecting misslabeled audios.
    4. Full paired LID     — Whisper detect_language on BOTH sides. For detecting src contamination (e.g. German audio on the EN side) in addition to tgt contamination.
    5. Common-ID intersection in load_data() — final cross-language alignment.

    Args:
        expressive:      load the 5-pair expressive variant instead of the 35-pair one
        use_tgt_lid:     run LID on the target (non-source) language side.
                         Catches EN-labeled-as-DE contamination.  Default True.
        use_paired_lid:  run LID on BOTH sides (superset of use_tgt_lid).
                         Catches all wrong-language contamination. Default False.
    """
    datasets_dict = {}
    if lang_list is None:
        return datasets_dict

    hf_name = ('jhu-clsp/seamless-align-expressive' if expressive
                else 'jhu-clsp/seamless-align')
    base_url = (
        "https://huggingface.co/datasets/jhu-clsp/seamless-align-expressive/resolve/main"
        if expressive else
        "https://huggingface.co/datasets/jhu-clsp/seamless-align/resolve/main"
    )

    # Group language pairs so we load each bilingual shard only once.
    pairs_to_load: dict[str, set] = {}
    for lang_code in lang_list:
        pair_cfg, is_pair = _get_seamless_config(
            lang_code, lang_list, expressive=expressive
        )
        if not pair_cfg:
            print(f"[SeamlessAlign] No config for '{lang_code}' — skipping.")
            continue
        pairs_to_load.setdefault(pair_cfg, set()).add(lang_code)

    for pair_cfg, langs_needed in pairs_to_load.items():
        data_files = [
            f"{base_url}/data/{pair_cfg}/src.tar.gz",
            f"{base_url}/data/{pair_cfg}/tgt.tar.gz",
        ]
        raw_ds = _load_or_download_generic(
            'seamless_align', hf_name, "default", pair_cfg, split,
            data_files=data_files, **kwargs
        )
        if raw_ds is None:
            continue

        # ── Lazy extraction: NO audio is decoded or copied here ──────────
        src_raw, tgt_raw, src_lang, tgt_lang = _process_seamless_dataset_paired(
            raw_ds, pair_cfg, start_idx, num_samples
        )
        if src_raw is None or tgt_raw is None:
            continue

        n_initial = len(src_raw)
        print(f"[SeamlessAlign] {pair_cfg}: {n_initial} candidate pairs")

        # ── 1. Quality filter (always runs, even without LID) ────────────
        print(f"  Quality filter (duration + silence)...")
        src_q = set(_quality_filter_indices(src_raw))
        tgt_q = set(_quality_filter_indices(tgt_raw))
        both_q = sorted(src_q & tgt_q)   # position-based: src_raw[i] ↔ tgt_raw[i]
        print(f"  Quality: {len(both_q)}/{n_initial} pairs passed")

        if not both_q:
            print(f"  [warn] No pairs survived quality filter for {pair_cfg}.")
            continue

        # Lazy indexed views of quality-filtered pairs (zero bytes copied)
        src_raw = src_raw.select(both_q)
        tgt_raw = tgt_raw.select(both_q)

        # ── 2. Cross-lingual duplicate detection ──────────────────────────
        n_after_q = len(src_raw)
        print(f"  Cross-lingual dedup (waveform≥0.85, spectrum≥0.82)...")
        valid_dedup = _crosslang_duplicate_filter_indices(src_raw, tgt_raw)
        src_raw = src_raw.select(valid_dedup)
        tgt_raw = tgt_raw.select(valid_dedup)
        print(f"  Dedup: {len(valid_dedup)}/{n_after_q} genuinely different pairs kept")

        if not valid_dedup:
            print(f"  [warn] All pairs were duplicates for {pair_cfg}.")
            continue

        # ── 3. Non-English-side LID ───────────────────────────────────────────
        if use_tgt_lid and not use_paired_lid:
            n_after_dedup = len(src_raw)

            # Find the non-English side — that's where contamination appears
            if src_lang != 'en':
                lid_lang = src_lang   # e.g. 'de' for deA-enA
                lid_raw  = src_raw
            else:
                lid_lang = tgt_lang   # e.g. 'fr' if src happened to be 'en'
                lid_raw  = tgt_raw

            est_min = n_after_dedup / 300  # detect_language ≈ 0.2 s/sample on CPU int8
            if est_min > 2:
                print(
                    f"  ⚠️  Non-EN LID will check {n_after_dedup} clips "
                    f"(~{est_min:.0f} min on CPU). "
                    f"Set seamless_tgt_lid=False to skip."
                )
            print(f"  Non-EN-side LID (checking '{lid_lang}' is not English)...")
            valid_l = _lid_filter_indices(lid_raw, lid_lang)
            src_raw = src_raw.select(valid_l)
            tgt_raw = tgt_raw.select(valid_l)
            print(f"  Non-EN LID: {len(valid_l)}/{n_after_dedup} pairs passed")

            if not valid_l:
                print(f"  [warn] No pairs survived non-EN LID for {pair_cfg}.")
                continue


        # ── 4. Full paired LID (optional — checks BOTH sides) ─────────────
        if use_paired_lid:
            n_after_dedup = len(src_raw)
            est_min = (n_after_dedup * 2) / 60
            if est_min > 10:
                print(
                    f"  ⚠️  Full paired LID will process {n_after_dedup * 2} audio files "
                    f"(~{est_min:.0f} min on CPU; much faster on GPU). "
                    f"Pass seamless_paired_lid=False to use tgt-only LID instead."
                )
            print(f"  Paired LID ({src_lang}↔{tgt_lang})...")
            tgt_l = set(_lid_filter_indices(tgt_raw, tgt_lang))
            src_l = set(_lid_filter_indices(src_raw, src_lang))
            both_l = sorted(tgt_l & src_l)
            print(f"  Paired LID: {len(both_l)}/{n_after_dedup} pairs passed")

            if not both_l:
                print(f"  [warn] No pairs survived paired LID for {pair_cfg}.")
                continue

            src_raw = src_raw.select(both_l)
            tgt_raw = tgt_raw.select(both_l)

        # ── 4. Finalise: add metadata + materialise (small dataset only) ──
        n_final = len(src_raw)
        print(f"  Finalising {n_final} pairs (writing Arrow cache)...")
        src_ds = _finalize_seamless_side(src_raw, src_lang, 'src/')
        tgt_ds = _finalize_seamless_side(tgt_raw, tgt_lang, 'tgt/')

        # Store results for whichever side(s) were requested
        for lang_code in langs_needed:
            if lang_code == src_lang:
                datasets_dict[lang_code] = src_ds
            elif lang_code == tgt_lang:
                datasets_dict[lang_code] = tgt_ds

    return datasets_dict


def _load_cvss_data(split, lang_list, start_idx, num_samples, **kwargs):
    """
    Loads Google CVSS dataset (English translation audio/text) AND
    Mozilla Common Voice 4.0 for the source audio.
    """
    datasets_dict = {}
    cols = ['id', 'audio', 'transcription', 'language', 'gender']
    if not lang_list:
        return datasets_dict

    en_parts = []
    for lang in lang_list:
        if lang == 'en' or lang not in CVSS_LANGS:
            continue

        # Source side: Common Voice 4.0 (Original Lang)
        ds_src = _load_or_download_generic(
            'cv4', 'mozilla-foundation/common_voice_4_0', lang, lang, split,
            num_samples=num_samples, start_idx=start_idx, **kwargs
        )
        if ds_src:
            def transform_cv4(batch):
                clean_path = os.path.splitext(batch['path'])[0]
                return {
                    'id':           generate_id_from_string(str(clean_path)),
                    'audio':        batch['audio'],
                    'transcription': batch['sentence'],
                    'language':     lang,
                    'gender':       batch.get('gender', 'unknown'),
                }
            datasets_dict[lang] = ds_src.map(
                transform_cv4,
                remove_columns=[c for c in ds_src.features
                                if c not in cols and c != 'audio'],
                num_proc=NUM_PROC,
            )

        # Target side: CVSS (English translation)
        ds_tgt = _load_or_download_generic(
            'cvss', 'google/cvss', "cvss_t", f"cvss_t_{lang}", split,
            num_samples=num_samples, start_idx=start_idx,
            languages=[lang], **kwargs
        )
        if ds_tgt:
            def transform_cvss(batch):
                return {
                    'id':            generate_id_from_string(str(batch['id'])),
                    'audio':         batch['audio'],
                    'transcription': batch['text'],
                    'language':      'en',
                    'gender':        'unknown',
                }
            en_parts.append(ds_tgt.map(
                transform_cvss,
                remove_columns=[c for c in ds_tgt.features
                                if c not in cols and c != 'audio'],
                num_proc=NUM_PROC,
            ))

    if en_parts:
        datasets_dict['en'] = (
            concatenate_datasets(en_parts) if len(en_parts) > 1 else en_parts[0]
        )
    return datasets_dict

# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def load_data(start_idx=0, num_samples=10000, encoding=None, lang=None,
              split="train", dataset=None,
              seamless_expressive=False,
              seamless_tgt_lid=True,
              seamless_paired_lid=False,
              **kwargs):
    """
    Unified data loader.

    Args:
        seamless_expressive:  use seamless-align-expressive (5 pairs) instead of
                              the full seamless-align (35 pairs).
        seamless_tgt_lid:     run Whisper LID on the TARGET (non-source) language
                              side of each SeamlessAlign pair.
        seamless_paired_lid:  run paired LID on BOTH sides of each SeamlessAlign
                              pair.
        start_idx, num_samples, lang, split, dataset, encoding: as before.
    """
    if dataset is None:
        dataset = ['fleurs']
    lang = lang or []
    final_datasets = collections.defaultdict(list)

    if 'fleurs' in dataset:
        for l, ds in _load_fleurs_data(split, lang, 0, None, **kwargs).items():
            final_datasets[l].append(ds)

    if 'seamless_align' in dataset:
        sa_limit = (num_samples * 2) if num_samples is not None else None

        # Speed guidance for LID
        active_lid = seamless_paired_lid or seamless_tgt_lid
        if active_lid and num_samples is not None and num_samples > 1000:
            n_calls = num_samples * (4 if seamless_paired_lid else 2)
            est_h = n_calls / 3600
            lid_mode = 'paired' if seamless_paired_lid else 'tgt-only'
            print(
                f"[SeamlessAlign] {lid_mode} LID with num_samples={num_samples}: "
                f"~{n_calls} Whisper detect_language calls (~{est_h:.1f}h CPU / "
                f"~{est_h*0.01:.1f}h GPU).\n"
                f"GPU strongly recommended for large loads."
            )

        for l, ds in _load_seamless_data(
            split, lang, start_idx, sa_limit,
            expressive=seamless_expressive,
            use_tgt_lid=seamless_tgt_lid,
            use_paired_lid=seamless_paired_lid,
            **kwargs,
        ).items():
            final_datasets[l].append(ds)

    if 'cvss' in dataset:
        for l, ds in _load_cvss_data(split, lang, 0, None, **kwargs).items():
            final_datasets[l].append(ds)

    # Concatenate and ensure all requested languages exist
    datasets = {}
    for l in lang:
        ds_list = [d for d in final_datasets[l] if d is not None]
        if ds_list:
            datasets[l] = (concatenate_datasets(ds_list)
                           if len(ds_list) > 1 else ds_list[0])
        else:
            datasets[l] = Dataset.from_dict(
                {'id': [], 'audio': [], 'transcription': [], 'language': [], 'gender': []}
            )

    # Deduplication and validation
    valid_id_sets = []

    def check_validity_batch(batch):
        mask = []
        for a in batch['audio']:
            try:
                mask.append(a is not None and a.get('array') is not None)
            except Exception:
                mask.append(False)
        return {'id': batch['id'], 'is_valid': mask}

    for l, ds in datasets.items():
        if ds is None or len(ds) == 0:
            valid_id_sets.append(set())
            continue

        print(f"Validating {l} (audio & uniqueness)...")
        meta = ds.map(
            check_validity_batch, batched=True,
            remove_columns=ds.column_names,
            num_proc=NUM_PROC, desc=f"Validating {l}",
        )

        ids, is_valid    = meta['id'], meta['is_valid']
        current_valid, seen = set(), set()
        for i, uid in enumerate(ids):
            if uid not in seen:
                seen.add(uid)
                if is_valid[i]:
                    current_valid.add(uid)
        valid_id_sets.append(current_valid)

    common_ids = valid_id_sets[0] if valid_id_sets else set()
    for s in valid_id_sets[1:]:
        common_ids.intersection_update(s)

    print(f"Final Count: {len(common_ids)} common valid samples.")
    sorted_common = (
        sorted(list(common_ids))[start_idx:start_idx + num_samples]
        if num_samples else
        sorted(list(common_ids))[start_idx:]
    )

    for l in datasets:
        if datasets[l] is None:
            continue
        src_ids  = datasets[l]['id']
        id_to_idx = {uid: idx for idx, uid in enumerate(src_ids)}
        indices   = [id_to_idx[uid] for uid in sorted_common if uid in id_to_idx]
        datasets[l] = (datasets[l].select(indices).sort("id")
                       if indices else datasets[l].select([]))

    if encoding:
        print(f"Applying encoding: {type(encoding).__name__}")
        for l, ds in datasets.items():
            if not ds or len(ds) == 0:
                continue
            if isinstance(encoding, SpectrogramEncoder):
                datasets[l] = ds.map(
                    apply_encoding_spectrogram,
                    fn_kwargs={"encoder": encoding},
                    num_proc=8, desc=f"Encoding {l} (Spectrogram)",
                )
            elif isinstance(encoding, (Wav2VecEncoder, VQGANEncoder)):
                datasets[l] = ds.map(
                    apply_encoding_gpu,
                    fn_kwargs={"encoder": encoding},
                    batch_size=32, desc=f"Encoding {l} ({type(encoding).__name__})",
                )
            elif isinstance(encoding, SpeechT5Encoder):
                datasets[l] = ds.map(
                    apply_encoding_speecht5,
                    fn_kwargs={"encoder": encoding},
                    batched=True, batch_size=32,
                    num_proc=NUM_PROC, desc=f"Encoding {l} (SpeechT5)",
                )

    return datasets

# ---------------------------------------------------------------------------
# STANDALONE HELPERS
# ---------------------------------------------------------------------------

def verify_dataset_language(ds, target_lang,
                             model_size="tiny", threshold=LID_THRESHOLD):
    """
    Single-side LID filter (kept for backwards compatibility and non-paired
    use cases such as CVSS or FLEURS spot-checking).
    For SeamlessAlign use verify_paired_language() instead.
    """
    mask_ds = ds.map(
        check_language_batch, batched=True, batch_size=LID_BATCH_SIZE,
        num_proc=1,
        fn_kwargs={"expected_lang": target_lang, "threshold": threshold},
        remove_columns=ds.column_names,
        desc=f"LID Verification ({target_lang})",
    )
    valid_indices = [i for i, val in enumerate(mask_ds['is_valid_lang']) if val]
    print(f"LID Filter ({target_lang}): Kept {len(valid_indices)}/{len(ds)} samples.")
    return ds.select(valid_indices)

def play_audio(record):
    return Audio(data=record['audio']['array'], rate=record['audio']['sampling_rate'])

def save_audio(record, path='audio.wav'):
    sf.write(path, record['audio']['array'], record['audio']['sampling_rate'])