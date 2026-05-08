import sys
import os
import collections
import torch
import numpy as np
import soundfile as sf
import hashlib
import io
import traceback
from datasets import load_dataset, concatenate_datasets, DownloadConfig, IterableDataset, Dataset, load_from_disk, load_dataset_builder
from IPython.display import Audio
from encoders import *

project_root = os.path.abspath(os.path.join(os.getcwd(), ''))
sys.path.append(project_root)

# ---------------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ---------------------------------------------------------------------------
NUM_PROC = 4
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

# Supported languages for CVSS (to English)
CVSS_LANGS = ['ar', 'ca', 'cy', 'de', 'el', 'es', 'et', 'fa', 'fr', 'id', 'it', 'ja', 'lv', 'mn', 'nl', 'pt', 'ru', 'sl', 'sv', 'tr', 'zh']

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
                if arr.ndim > 1: arr = arr.mean(axis=1) # stereo -> mono
            else:
                return None
        else:
            return None

        if sr != 16000:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
        return arr
    except Exception:
        return None

# ---------------------------------------------------------------------------
# DATASET MAP FUNCTIONS (Batched)
# ---------------------------------------------------------------------------

def compute_duration_batch(batch):
    """Batch calculation of audio duration in seconds."""
    durations = []
    for x in batch["audio"]:
        try:
            # Trigger decoding if necessary
            durations.append(len(x['array']) / x['sampling_rate'])
        except Exception:
            durations.append(0.0)
    return {"duration": durations}

def check_language_batch(batch, expected_lang, threshold=0.75):
    """
    Detects language using Whisper and returns a boolean mask.
    Compatible with datasets.map(batched=True).
    """
    model = _get_whisper_model()
    if model is None:
        return {"is_valid_lang": [True] * len(batch["audio"])}

    is_valid = []
    for audio_field in batch["audio"]:
        try:
            arr = _decode_audio(audio_field)
            if arr is None:
                is_valid.append(False)
                continue
            
            # Faster-whisper transcribe for LID
            _, info = model.transcribe(arr, beam_size=1)
            is_match = (info.language == expected_lang and info.language_probability > threshold)
            is_valid.append(is_match)
        except Exception as e:
            is_valid.append(True) # Keep on failure to be conservative
    return {"is_valid_lang": is_valid}

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

def _get_seamless_config(lang, context_langs=None):
    PAIR_MAPPING = {
        tuple(sorted(('de', 'en'))): 'deA-enA',
        tuple(sorted(('en', 'es'))): 'enA-esA',
        tuple(sorted(('en', 'fr'))): 'enA-frA',
        tuple(sorted(('en', 'it'))): 'enA-itA',
        tuple(sorted(('en', 'zh'))): 'enA-zhA',
    }
    SEAMLESS_SINGLE_SUPPORTED = {'en', 'de', 'es', 'fr', 'it', 'zh'}
    lang_code = lang[:2]
    
    if context_langs:
        ctx_codes = set(l[:2] for l in context_langs)
        if lang_code in ctx_codes:
            for other_code in ctx_codes:
                if other_code == lang_code: continue
                pair_key = tuple(sorted((lang_code, other_code)))
                if pair_key in PAIR_MAPPING:
                    return PAIR_MAPPING[pair_key], True
    
    if lang_code in SEAMLESS_SINGLE_SUPPORTED:
         return "default", False
    return None, False

def _process_seamless_dataset(ds, lang_code, config_name, start_idx=0, num_samples=None):
    if ds is None or 'mp3' not in ds.column_names: return ds 
    
    try:
        parts = config_name.split('-')
        src_lang, tgt_lang = parts[0][:2], parts[1][:2]
    except:
        return ds 

    prefix = 'src/' if lang_code == src_lang else 'tgt/' if lang_code == tgt_lang else None
    if not prefix: return ds

    all_keys = ds['__key__']
    src_indices, tgt_indices = {}, {}
    for idx, k in enumerate(all_keys):
        if k.startswith('src/'): src_indices[k[4:]] = idx
        elif k.startswith('tgt/'): tgt_indices[k[4:]] = idx
    
    common_ids = sorted(list(set(src_indices.keys()) & set(tgt_indices.keys())))
    indices = [src_indices[uid] if prefix == 'src/' else tgt_indices[uid] for uid in common_ids]

    if num_samples is not None:
        indices = indices[start_idx:start_idx + num_samples]

    ds = ds.select(indices)
    
    def map_seamless_cols(batch):
        clean_id = batch['__key__'].replace(prefix, '').replace('/', '')
        return {
            'id': generate_id_from_string(clean_id),
            'audio': batch['mp3'], 
            'language': lang_code,
            'gender': 'unknown', 'transcription': '' 
        }

    return ds.map(map_seamless_cols, remove_columns=['mp3', '__key__', '__url__'])

def _load_or_download_generic(dataset_key, hf_dataset_name, hf_config, local_config_name, split, data_files=None, num_samples=None, start_idx=0, **kwargs):
    cache_dir = os.path.join(DATASETS_DIR, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(DATASETS_DIR, dataset_key, local_config_name, split)
    
    if os.path.exists(local_path):
        print(f"Loading {hf_dataset_name} ({local_config_name}) from local storage...")
        ds = load_from_disk(local_path)
        if num_samples is not None and dataset_key not in ('seamless_align', 'cvss', 'cv4'):
             ds = ds.select(range(start_idx, min(start_idx + num_samples, len(ds)))) if start_idx < len(ds) else ds.select([])
        return ds
        
    # Check for Hugging Face token
    token = kwargs.get('token') or os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    
    size_info = "Size info unavailable"
    try:
         builder = load_dataset_builder(hf_dataset_name, hf_config, trust_remote_code=True, cache_dir=cache_dir, token=token, **kwargs)
         dl_s, ds_s = builder.info.download_size, builder.info.dataset_size
         if dl_s and ds_s: size_info = f"Download: {dl_s/1024**3:.2f} GB, Generated: {ds_s/1024**3:.2f} GB"
    except Exception: pass
         
    print(f"Dataset {hf_dataset_name} ({local_config_name}) not found locally. {size_info}")
    
    if sys.stdin.isatty():
        if input(f"Proceed with download of {dataset_key}? (Y/n): ").strip().lower() not in ('', 'y', 'yes'):
            return None
    else:
        print(f"Non-interactive environment: Proceeding with download of {dataset_key}...")
        
    print(f"Downloading {hf_dataset_name}...")
    dl_config = DownloadConfig(resume_download=True, max_retries=15)
    try:
        dataset = load_dataset(hf_dataset_name, hf_config, split=split, trust_remote_code=True, 
                               download_config=dl_config, cache_dir=cache_dir, data_files=data_files, 
                               token=token, **kwargs)
        print(f"Saving to {local_path}...")
        dataset.save_to_disk(local_path)
        if num_samples is not None and dataset_key not in ('seamless_align', 'cvss', 'cv4'):
             dataset = dataset.select(range(start_idx, min(start_idx + num_samples, len(dataset)))) if start_idx < len(dataset) else dataset.select([])
        return dataset
    except Exception as e:
        print(f"FAILED to download/load {hf_dataset_name}: {e}")
        if "gated" in str(e).lower() or "403" in str(e):
            print("ERROR: This dataset is GATED. Please run `huggingface-cli login` or set HF_TOKEN environment variable.")
        else:
            traceback.print_exc()
        return None

# ---------------------------------------------------------------------------
# DATASET LOADER FUNCTIONS
# ---------------------------------------------------------------------------

def _load_fleurs_data(split, lang_list, start_idx, num_samples, **kwargs):
    datasets_dict = {}
    if lang_list is None: return datasets_dict
    cols = ['id', 'audio', 'transcription', 'language', 'gender']

    for lang in lang_list:
        cfg = _get_fleurs_config(lang)
        ds = _load_or_download_generic('fleurs', 'google/fleurs', cfg, cfg, split, num_samples=num_samples, start_idx=start_idx, **kwargs)
        if ds:
             datasets_dict[lang] = ds.map(
                transform_internal,
                remove_columns=[c for c in ds.features if c not in cols and c != 'audio'],
                num_proc=NUM_PROC
             )
    return datasets_dict

def _load_seamless_data(split, lang_list, start_idx, num_samples, **kwargs):
    datasets_dict = {}
    if lang_list is None: return datasets_dict

    for lang_code in lang_list:
         pair_cfg, is_pair = _get_seamless_config(lang_code, lang_list)
         if not pair_cfg: continue
             
         data_files = None
         if is_pair:
             base = "https://huggingface.co/datasets/jhu-clsp/seamless-align-expressive/resolve/main"
             data_files = [f"{base}/data/{pair_cfg}/src.tar.gz", f"{base}/data/{pair_cfg}/tgt.tar.gz"]
         
         ds = _load_or_download_generic('seamless_align', 'jhu-clsp/seamless-align-expressive', "default", pair_cfg, split, data_files=data_files, **kwargs)
         if ds:
             ds = _process_seamless_dataset(ds, lang_code, pair_cfg, start_idx, num_samples)
             print(f"Enabling LID filter for Seamless Align ({lang_code})...")
             datasets_dict[lang_code] = verify_dataset_language(ds, lang_code)
    return datasets_dict

def _load_cvss_data(split, lang_list, start_idx, num_samples, **kwargs):
    """
    Loads Google CVSS dataset (English translation audio/text).
    Mozilla Common Voice (cv4) has been removed per user request.
    Note: This only provides the English side of the translation.
    """
    datasets_dict = {}
    cols = ['id', 'audio', 'transcription', 'language', 'gender']
    if not lang_list: return datasets_dict

    en_parts = []
    for lang in lang_list:
        if lang == 'en' or lang not in CVSS_LANGS: continue
            
        # CVSS provides (Original Lang) -> English pairs.
        # We load the 'transferred' version (cvss_t) which preserves speaker voice.
        ds_tgt = _load_or_download_generic('cvss', 'google/cvss', "cvss_t", f"cvss_t_{lang}", split, 
                                           num_samples=num_samples, start_idx=start_idx, languages=[lang], **kwargs)
        if ds_tgt:
            def transform_cvss(batch):
                return {
                    'id': generate_id_from_string(str(batch['id'])),
                    'audio': batch['audio'], 
                    'transcription': batch['text'], 
                    'language': 'en', 
                    'gender': 'unknown'
                }
            en_parts.append(ds_tgt.map(transform_cvss, remove_columns=[c for c in ds_tgt.features if c not in cols and c != 'audio'], num_proc=NUM_PROC))

    if en_parts:
        datasets_dict['en'] = concatenate_datasets(en_parts) if len(en_parts) > 1 else en_parts[0]
    return datasets_dict

# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def load_data(start_idx=0, num_samples=10000, encoding=None, lang=None, split="train", dataset=None, **kwargs):
    if dataset is None: dataset = ['fleurs']
    lang = lang or []
    final_datasets = collections.defaultdict(list)
    
    if 'fleurs' in dataset:
        for l, ds in _load_fleurs_data(split, lang, 0, None, **kwargs).items(): final_datasets[l].append(ds)
    if 'seamless_align' in dataset:
        for l, ds in _load_seamless_data(split, lang, 0, None, **kwargs).items(): final_datasets[l].append(ds)
    if 'cvss' in dataset:
        for l, ds in _load_cvss_data(split, lang, 0, None, **kwargs).items(): final_datasets[l].append(ds)
            
    # Concatenate and Ensure all requested languages exist (even if empty)
    datasets = {}
    for l in lang:
        ds_list = [d for d in final_datasets[l] if d is not None]
        if ds_list:
            datasets[l] = concatenate_datasets(ds_list) if len(ds_list) > 1 else ds_list[0]
        else:
            # Create an empty dataset with the correct schema to prevent KeyErrors
            datasets[l] = Dataset.from_dict({
                'id': [], 'audio': [], 'transcription': [], 'language': [], 'gender': []
            })
            
    # Optimized Deduplication and Validation
    valid_id_sets = []
    def check_validity_batch(batch):
        mask = []
        for a in batch['audio']:
            try: mask.append(a is not None and a.get('array') is not None)
            except: mask.append(False)
        return {'id': batch['id'], 'is_valid': mask}

    for l, ds in datasets.items():
        if ds is None or len(ds) == 0:
            valid_id_sets.append(set())
            continue
        
        print(f"Validating {l} (audio & uniqueness)...")
        meta = ds.map(check_validity_batch, batched=True, remove_columns=ds.column_names, num_proc=NUM_PROC, desc=f"Validating {l}")
        
        ids, is_valid = meta['id'], meta['is_valid']
        current_valid, seen = set(), set()
        for i, uid in enumerate(ids):
            if uid not in seen:
                seen.add(uid)
                if is_valid[i]: current_valid.add(uid)
        valid_id_sets.append(current_valid)

    common_ids = valid_id_sets[0] if valid_id_sets else set()
    for s in valid_id_sets[1:]: common_ids.intersection_update(s)
            
    print(f"Final Count: {len(common_ids)} common valid samples.")
    sorted_common = sorted(list(common_ids))[start_idx:start_idx + num_samples] if num_samples else sorted(list(common_ids))[start_idx:]

    for l in datasets:
        if datasets[l] is None: continue
        src_ids = datasets[l]['id']
        id_to_idx = {uid: idx for idx, uid in enumerate(src_ids)}
        indices = [id_to_idx[uid] for uid in sorted_common if uid in id_to_idx]
        datasets[l] = datasets[l].select(indices).sort("id") if indices else datasets[l].select([])

    if encoding:
        print(f"Applying encoding: {type(encoding).__name__}")
        for l, ds in datasets.items():
            if not ds or len(ds) == 0: continue
            if isinstance(encoding, SpectrogramEncoder):
                datasets[l] = ds.map(apply_encoding_spectrogram, fn_kwargs={"encoder": encoding}, num_proc=8, desc=f"Encoding {l} (Spectrogram)")
            elif isinstance(encoding, (Wav2VecEncoder, VQGANEncoder)):
                datasets[l] = ds.map(apply_encoding_gpu, fn_kwargs={"encoder": encoding}, batch_size=32, desc=f"Encoding {l} ({type(encoding).__name__})")
            elif isinstance(encoding, SpeechT5Encoder):
                datasets[l] = ds.map(apply_encoding_speecht5, fn_kwargs={"encoder": encoding}, batched=True, batch_size=32, num_proc=NUM_PROC, desc=f"Encoding {l} (SpeechT5)")
    
    return datasets

def verify_dataset_language(ds, target_lang, model_size="tiny", threshold=0.75):
    """Filters dataset to ensure samples match target_lang using LID."""
    mask_ds = ds.map(check_language_batch, batched=True, batch_size=16, num_proc=1, 
                     fn_kwargs={"expected_lang": target_lang, "threshold": threshold},
                     remove_columns=ds.column_names, desc=f"LID Verification ({target_lang})")
    valid_indices = [i for i, val in enumerate(mask_ds['is_valid_lang']) if val]
    print(f"LID Filter ({target_lang}): Kept {len(valid_indices)}/{len(ds)} samples.")
    return ds.select(valid_indices)

def play_audio(record): return Audio(data=record['audio']['array'], rate=record['audio']['sampling_rate'])
def save_audio(record, path='audio.wav'): sf.write(path, record['audio']['array'], record['audio']['sampling_rate'])
