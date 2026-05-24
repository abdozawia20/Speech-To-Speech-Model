import sys
import os
import collections
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

# Supported languages for CVSS (to English)
CVSS_LANGS = ['ar', 'ca', 'cy', 'de', 'el', 'es', 'et', 'fa', 'fr', 'id', 'it', 'ja', 'lv', 'mn', 'nl', 'pt', 'ru', 'sl', 'sv', 'tr', 'zh']

# ---------------------------------------------------------------------------
# SEAMLESS-ALIGN LANGUAGE CONFIGURATION
# ---------------------------------------------------------------------------
# All 35 SeamlessAlign S2S language pairs (all paired with English).
# Config names follow the convention: sorted alphabetically by 2-letter code,
# e.g. ('de', 'en') → 'deA-enA', ('en', 'es') → 'enA-esA'.
# We generate the mapping programmatically so it's easy to extend.
_SEAMLESS_S2S_LANGS = [
    'ar', 'bg', 'ca', 'cs', 'cy', 'da', 'de', 'el',  # before 'e'
    'es', 'et', 'fi', 'fr', 'ga', 'gl', 'hr', 'hu',  # e onwards
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

# Full set supported by seamless-align (35 non-English langs × en)
SEAMLESS_PAIR_MAPPING = _build_pair_mapping(_SEAMLESS_S2S_LANGS)

# Only the 5 pairs available in seamless-align-EXPRESSIVE
_SEAMLESS_EXPRESSIVE_LANGS = ['de', 'es', 'fr', 'it', 'zh']
SEAMLESS_EXPRESSIVE_PAIR_MAPPING = _build_pair_mapping(_SEAMLESS_EXPRESSIVE_LANGS)

# Audio quality thresholds
MIN_DURATION_S  = 1.0   # discard clips shorter than 1 second (too short for LID/speech)
MAX_DURATION_S  = 20.0  # discard clips longer than 20 s (very long utterances hurt seq models)
LID_THRESHOLD   = 0.75  # Whisper language probability threshold
LID_BATCH_SIZE  = 16    # samples per LID batch

# ---------------------------------------------------------------------------
# WORKER GLOBALS (Lazy-loaded per process)
# ---------------------------------------------------------------------------
_shared_language_model = None

def _get_whisper_model(model_size="tiny", device="cpu", compute_type="int8"):
    """Lazily loads the Whisper model for language identification."""
    global _shared_language_model
    if _shared_language_model is None:
        try:
            from faster_whisper import WhisperModel
            print(f"[PID {os.getpid()}] Loading Whisper model ({model_size}) on {device}...")
            _shared_language_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except ImportError:
            print("Warning: faster-whisper not installed. Language filtering will be disabled.")
            return None
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
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

def check_language_batch(batch, expected_lang, threshold=LID_THRESHOLD):
    """
    Detects language using Whisper and returns a boolean mask.
    Compatible with datasets.map(batched=True).
    Also gates on duration to skip Whisper for clips that are
    clearly too short or too long.
    """
    model = _get_whisper_model()
    if model is None:
        return {"is_valid_lang": [True] * len(batch["audio"])}

    is_valid = []
    for audio_field in batch["audio"]:
        try:
            arr, sr = _decode_audio(audio_field)
            if arr is None:
                is_valid.append(False)
                continue

            duration = len(arr) / 16000
            if duration < MIN_DURATION_S or duration > MAX_DURATION_S:
                is_valid.append(False)
                continue

            _, info = model.transcribe(arr, beam_size=1)
            is_match = (info.language == expected_lang and
                        info.language_probability >= threshold)
            is_valid.append(is_match)
        except Exception as e:
            # Keep on failure to be conservative — flagged by quality check later
            is_valid.append(True)
    return {"is_valid_lang": is_valid}

# ---------------------------------------------------------------------------
# QUALITY FILTER (duration + audio validity)
# ---------------------------------------------------------------------------

def quality_filter_batch(batch,
                          min_dur=MIN_DURATION_S,
                          max_dur=MAX_DURATION_S):
    """
    Fast pre-filter: removes silent/corrupt/too-short/too-long clips
    WITHOUT running Whisper.  Intended as a first pass before LID.
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

    This is the correct approach for web-scraped data: a pair is only
    valid if both audio segments truly contain the expected language.

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
    pair_mapping   = SEAMLESS_EXPRESSIVE_PAIR_MAPPING if expressive \
                     else SEAMLESS_PAIR_MAPPING
    # pair_mapping keys are tuples: ('de', 'en'), ('en', 'fr'), etc.
    # Unpack both language codes from each key tuple directly.
    # (The previous set-comprehension over pair_mapping[k][0][:2] was wrong:
    # pair_mapping[k] is a string like 'deA-enA', so [0] gives char 'd', not 'de'.)
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
    Variant of _process_seamless_dataset that extracts BOTH src and tgt
    sides simultaneously (from a single loaded shard).  Returns
    (src_ds, tgt_ds) with aligned rows so that src_ds[i] and tgt_ds[i]
    are a translation pair.

    Use this when you need paired LID filtering before merging into the
    per-language datasets dict.
    """
    if ds is None or 'mp3' not in ds.column_names:
        return None, None

    try:
        parts    = config_name.split('-')
        src_lang = parts[0][:2]
        tgt_lang = parts[1][:2]
    except Exception:
        print(f"Cannot parse config '{config_name}' — skipping.")
        return None, None

    all_keys   = ds['__key__']
    src_map, tgt_map = {}, {}
    for idx, k in enumerate(all_keys):
        if k.startswith('src/'):
            src_map[k[4:]] = idx
        elif k.startswith('tgt/'):
            tgt_map[k[4:]] = idx

    common_ids = sorted(set(src_map) & set(tgt_map))
    if num_samples is not None:
        common_ids = common_ids[start_idx:start_idx + num_samples]

    src_indices = [src_map[uid] for uid in common_ids]
    tgt_indices = [tgt_map[uid] for uid in common_ids]

    def _make_side(raw_ds, side_indices, lang_code, prefix):
        side = raw_ds.select(side_indices)
        def _map(batch):
            clean_id = batch['__key__'].replace(prefix, '').replace('/', '')
            return {
                'id':            generate_id_from_string(clean_id),
                'audio':         batch['mp3'],
                'language':      lang_code,
                'gender':        'unknown',
                'transcription': '',
            }
        return side.map(_map, remove_columns=['mp3', '__key__', '__url__'])

    src_ds = _make_side(ds, src_indices, src_lang, 'src/')
    tgt_ds = _make_side(ds, tgt_indices, tgt_lang, 'tgt/')
    return src_ds, tgt_ds

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


def _load_seamless_data(split, lang_list, start_idx, num_samples,
                         expressive=False, use_paired_lid=True, **kwargs):
    """
    Load SeamlessAlign (or SeamlessAlign-Expressive) for the requested languages.

    Key filtering strategy for web-scraped data
    -------------------------------------------
    Unlike FLEURS, SeamlessAlign is mined from the open web with SONAR
    embeddings.  Language labels come from the __key__ prefix (src/ vs tgt/)
    — they are NOT ground-truth.  Three filtering layers are applied:

    1. Quality pre-filter  — duration gates + silence detection (cheap, CPU)
    2. Paired LID          — Whisper LID on BOTH sides simultaneously so only
                             pairs where src AND tgt match their expected
                             language are kept.  This is critical: filtering
                             each side independently can keep a pair where one
                             side is wrong language.
    3. Common-ID intersection in load_data() — final alignment across datasets.

    Args:
        expressive:     load the 5-pair expressive variant instead of the 35-pair one
        use_paired_lid: run paired LID filtering (recommended; disable only for
                        quick iteration / debugging)
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
    # pairs_to_load: config_name → set of lang_codes we need from it
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

        if use_paired_lid:
            # ── Paired path: extract both sides, filter together ──────────
            src_ds, tgt_ds = _process_seamless_dataset_paired(
                raw_ds, pair_cfg, start_idx, num_samples
            )
            if src_ds is None or tgt_ds is None:
                continue

            # Quality pre-filter (duration + silence) on both sides
            for side_ds, label in [(src_ds, 'src'), (tgt_ds, 'tgt')]:
                q_mask = side_ds.map(
                    quality_filter_batch, batched=True, batch_size=64,
                    remove_columns=side_ds.column_names,
                    num_proc=NUM_PROC, desc=f"QualityFilter({pair_cfg}/{label})"
                )
                valid = [i for i, ok in enumerate(q_mask['quality_ok']) if ok]
                if label == 'src':
                    src_ds = src_ds.select(valid)
                else:
                    tgt_ds = tgt_ds.select(valid)

            # Re-align after quality filter (both sides may have dropped rows)
            # IDs were generated from the same __key__ root, so intersect by id
            src_id_set = set(src_ds['id'])
            tgt_id_set = set(tgt_ds['id'])
            common     = src_id_set & tgt_id_set
            src_ds = src_ds.filter(lambda b: [i in common for i in b['id']],
                                   batched=True)
            tgt_ds = tgt_ds.filter(lambda b: [i in common for i in b['id']],
                                   batched=True)

            # Paired LID (the core filter for web-scraped data)
            parts    = pair_cfg.split('-')
            src_lang = parts[0][:2]
            tgt_lang = parts[1][:2]
            src_ds, tgt_ds = verify_paired_language(
                src_ds, tgt_ds, src_lang, tgt_lang
            )

            # Store results for whichever side(s) were requested
            for lang_code in langs_needed:
                if lang_code == src_lang:
                    datasets_dict[lang_code] = src_ds
                elif lang_code == tgt_lang:
                    datasets_dict[lang_code] = tgt_ds

        else:
            # ── Non-paired path (legacy / debug mode) ────────────────────
            for lang_code in langs_needed:
                ds = _process_seamless_dataset(
                    raw_ds, lang_code, pair_cfg, start_idx, num_samples
                )
                if ds is not None:
                    print(f"LID filter ({lang_code}) — legacy mode...")
                    datasets_dict[lang_code] = verify_dataset_language(
                        ds, lang_code
                    )

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
              seamless_paired_lid=True,
              **kwargs):
    """
    Unified data loader.

    Args:
        seamless_expressive:  use seamless-align-expressive (5 pairs) instead of
                              the full seamless-align (35 pairs).
        seamless_paired_lid:  run paired LID on both sides of each SeamlessAlign
                              pair before adding to the dataset. Highly recommended
                              for web-scraped data. Disable only for debugging.
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
        # Pass a buffered sample limit so quality-filter and paired LID only
        # process a reasonable number of rows instead of the entire shard
        # (which can be 1.3M+ rows).  5× buffer accounts for the typical
        # 30-50% LID rejection rate on web-scraped data, ensuring we end up
        # with at least num_samples clean pairs after filtering.
        sa_limit = (num_samples * 5) if num_samples is not None else None
        for l, ds in _load_seamless_data(
            split, lang, start_idx, sa_limit,
            expressive=seamless_expressive,
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
# LEGACY / STANDALONE HELPERS
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