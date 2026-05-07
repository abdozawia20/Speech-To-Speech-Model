from datasets import load_dataset, concatenate_datasets, DownloadConfig, IterableDataset, Dataset, load_from_disk, load_dataset_builder
from IPython.display import Audio
import sys
import os
import collections
import torch
import numpy as np
from encoders import *
import soundfile as sf
project_root = os.path.abspath(os.path.join(os.getcwd(), ''))
sys.path.append(project_root)

NUM_PROC = 4

CVSS_LANGS = ['ar', 'ca', 'cy', 'de', 'el', 'es', 'et', 'fa', 'fr', 'id', 'it', 'ja', 'lv', 'mn', 'nl', 'pt', 'ru', 'sl', 'sv', 'tr', 'zh']

def generate_id_from_string(s):
    """
    Consistently generates a 64-bit integer ID from a string.
    If the string is numeric, it converts it directly.
    Otherwise, it uses a stable hash (non-randomized) to ensure consistency across sessions.
    """
    try:
        return int(s)
    except (ValueError, TypeError):
        import hashlib
        # Use md5 for a stable hash across different python processes/sessions
        return int(hashlib.md5(s.encode('utf-8')).hexdigest()[:16], 16)

def transform_internal(batch):
    batch['id'] = int(batch['id'])
    batch['gender'] = str(batch['gender'])
    batch['language'] = str(batch['language'])
    return batch

def transform_voxpopuli_internal(batch):
    batch['id'] = batch['audio_id']
    batch['transcription'] = batch['normalized_text']
    batch['language'] = str(batch['language'])
    return batch

def apply_encoding_spectrogram(example, encoder):
    audio_array = np.array(example['audio']['array'])
    sr = example['audio']['sampling_rate']
    encoded = encoder.encode(audio_array, sr)
    # We replace the 'audio' dictionary with the encoded array directly
    # or inside a structure. Assuming we replace 'audio' content with the embedding/spectrogram.
    example['audio'] = encoded
    return example

def apply_encoding_gpu(batch, encoder):
    audio_arrays = [x['array'] for x in batch['audio']]
    # Assume consistent sampling rate from the first example
    sr = batch['audio'][0]['sampling_rate']
    
    # Process inputs
    inputs = encoder.processor(audio_arrays, sampling_rate=sr, padding=True, return_tensors="pt")
    
    with torch.no_grad():
        encoded = encoder.encode(inputs.input_values)
        
    # Return as list of numpy arrays (or tensors if dataset supports it, but numpy is safer)
    return {"audio": list(encoded.cpu().numpy())}

def apply_encoding_speecht5(batch, encoder):
    audio_arrays = [x['array'] for x in batch['audio']]
    # Assume consistent sampling rate
    sr = batch['audio'][0]['sampling_rate']
    
    # Encoder handles processor call
    # Returns list of variable length arrays
    encoded = encoder.encode(audio_arrays, sr=sr)
    
    # Ensure they are numpy arrays (processor might return lists)
    import numpy as np
    encoded_np = [np.array(x) for x in encoded]
    
    return {"audio": encoded_np}

# Ensure DATASETS_DIR is absolute and robust to CWD
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")


def _get_fleurs_config(lang):
    mapping = {
        'en': 'en_us',
        'de': 'de_de',
        'ar': 'ar_eg',
        'tr': 'tr_tr',
        'fr': 'fr_fr',
        'es': 'es_419',
        'it': 'it_it',
        'zh': 'cmn_hans_cn',
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
    
    detected_pair = None
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
    if ds is None: return None
    if 'mp3' not in ds.column_names: return ds 
    
    try:
        parts = config_name.split('-')
        src_lang = parts[0][:2]
        tgt_lang = parts[1][:2]
    except:
        return ds 

    prefix = None
    if lang_code == src_lang:
        prefix = 'src/'
    elif lang_code == tgt_lang:
        prefix = 'tgt/'
    
    if prefix:
        all_keys = ds['__key__']
        src_indices = {}
        tgt_indices = {}
        
        for idx, k in enumerate(all_keys):
            if k.startswith('src/'):
                src_indices[k[4:]] = idx
            elif k.startswith('tgt/'):
                tgt_indices[k[4:]] = idx
        
        common_ids = sorted(list(set(src_indices.keys()) & set(tgt_indices.keys())))
        
        if prefix == 'src/':
            final_indices = [src_indices[uid] for uid in common_ids]
        else:
            final_indices = [tgt_indices[uid] for uid in common_ids]

        if num_samples is not None:
            end_idx = start_idx + num_samples
            if end_idx > len(final_indices): end_idx = len(final_indices)
            
            if start_idx < len(final_indices):
                 final_indices = final_indices[start_idx:end_idx]
            else:
                 final_indices = []

        ds = ds.select(final_indices)
        
        def map_seamless_cols(batch):
            key = batch['__key__']
            clean_id = key.replace(prefix, '').replace('/', '')
            int_id = generate_id_from_string(clean_id)
            
            return {
                'id': int_id,
                'audio': batch['mp3'], 
                'language': lang_code,
                'gender': 'unknown', 
                'transcription': '' 
            }

        ds = ds.map(map_seamless_cols, remove_columns=['mp3', '__key__', '__url__'])
    return ds

def _load_or_download_generic(dataset_key, hf_dataset_name, hf_config, local_config_name, split, data_files=None, num_samples=None, start_idx=0, **kwargs):
    cache_dir = os.path.join(DATASETS_DIR, ".cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    local_path = os.path.join(DATASETS_DIR, dataset_key, local_config_name, split)
    
    if os.path.exists(local_path):
        print(f"Loading {hf_dataset_name} ({local_config_name}) from local storage: {local_path}...")
        ds = load_from_disk(local_path)
        
        if num_samples is not None and dataset_key not in ('seamless_align', 'cvss', 'cv4'):
             total = len(ds)
             start = start_idx
             end = min(start + num_samples, total)
             if start < total:
                 print(f"Slicing loaded dataset ({start}:{end})...")
                 ds = ds.select(range(start, end))
             else:
                 ds = ds.select([])
        return ds
        
    print(f"Dataset {hf_dataset_name} ({local_config_name}) not found locally at {local_path}.")
    
    size_info = "Size info unavailable"
    try:
         builder = load_dataset_builder(hf_dataset_name, hf_config, trust_remote_code=True, cache_dir=cache_dir, **kwargs)
         dl_s = builder.info.download_size
         ds_s = builder.info.dataset_size
         if dl_s and ds_s:
             size_info = f"Download: {dl_s/1024**3:.2f} GB, Generated: {ds_s/1024**3:.2f} GB"
    except Exception as e:
         pass
         
    print(f"This will download to {local_path}. {size_info}")
    
    if sys.stdin.isatty():
        response = input(f"Proceed with download of {dataset_key}? (Y/n): ").strip().lower()
        if response not in ('', 'y', 'yes'):
            print("Aborted.")
            return None
    else:
        print(f"Non-interactive environment detected. Proceeding with download of {dataset_key}...")
        
    print(f"Downloading {hf_dataset_name}...")
    config = DownloadConfig(resume_download=True, max_retries=10)
    try:
        dataset = load_dataset(hf_dataset_name, hf_config, split=split, trust_remote_code=True, download_config=config, cache_dir=cache_dir, data_files=data_files, **kwargs)
        
        print(f"Saving to {local_path}...")
        dataset.save_to_disk(local_path)
        print("Save complete.")
        
        if num_samples is not None and dataset_key not in ('seamless_align', 'cvss', 'cv4'):
             total = len(dataset)
             start = start_idx
             end = min(start + num_samples, total)
             if start < total:
                  dataset = dataset.select(range(start, end))
             else:
                  dataset = dataset.select([])
        return dataset
    except Exception as e:
        print(f"Failed: {e}")
        return None

def _load_fleurs_data(split, lang_list, start_idx, num_samples):
    datasets_dict = {}
    columns_to_keep = ['id', 'audio', 'transcription', 'language', 'gender']
    
    if lang_list is None: return datasets_dict

    for lang in lang_list:
        cfg = _get_fleurs_config(lang)
        ds = _load_or_download_generic('fleurs', 'google/fleurs', cfg, cfg, split, num_samples=num_samples, start_idx=start_idx)
        if ds:
             ds = ds.map(
                transform_internal,
                remove_columns=[col for col in ds.features if col not in columns_to_keep and col != 'audio'],
                num_proc=NUM_PROC
             )
             datasets_dict[lang] = ds
    return datasets_dict

def _load_seamless_data(split, lang_list, start_idx, num_samples):
    datasets_dict = {}
    
    if lang_list is None: return datasets_dict

    for lang_code in lang_list:
         pair_config, is_pair = _get_seamless_config(lang_code, lang_list)
         
         if not pair_config:
             print(f"Skipping {lang_code} for Seamless (no config found)")
             continue
             
         data_files = None
         if is_pair:
             base_url = "https://huggingface.co/datasets/jhu-clsp/seamless-align-expressive/resolve/main"
             data_files = [
                 f"{base_url}/data/{pair_config}/src.tar.gz",
                 f"{base_url}/data/{pair_config}/tgt.tar.gz"
             ]
         
         ds = _load_or_download_generic(
             'seamless_align', 
             'jhu-clsp/seamless-align-expressive', 
             "default", 
             pair_config, 
             split, 
             data_files=data_files
         )
         
         if ds:
             ds = _process_seamless_dataset(ds, lang_code, pair_config, start_idx, num_samples)
             print(f"--- Language Filtering Enabled Exceptionally for Seamless Align ---")
             ds = verify_dataset_language(ds, lang_code)
             datasets_dict[lang_code] = ds
             
    return datasets_dict

def _load_cvss_data(split, lang_list, start_idx, num_samples):
    datasets_dict = {}
    columns_to_keep = ['id', 'audio', 'transcription', 'language', 'gender']
    
    if lang_list is None: return datasets_dict

    en_targets = []
    for lang in lang_list:
        if lang == 'en' or lang not in CVSS_LANGS:
            continue
            
        # Source side: Common Voice 4.0
        # Note: mozilla-foundation/common_voice_4_0 is gated and may require a token.
        ds_src = _load_or_download_generic('cv4', 'mozilla-foundation/common_voice_4_0', lang, lang, split, num_samples=num_samples, start_idx=start_idx)
        if ds_src:
            def transform_cv4(batch):
                # Strip extension from path to match CVSS IDs
                clean_path = os.path.splitext(batch['path'])[0]
                return {
                    'id': generate_id_from_string(str(clean_path)),
                    'audio': batch['audio'],
                    'transcription': batch['sentence'],
                    'language': lang,
                    'gender': batch['gender'] if 'gender' in batch else 'unknown'
                }
            ds_src = ds_src.map(
                transform_cv4, 
                remove_columns=[col for col in ds_src.features if col not in columns_to_keep and col != 'audio'],
                num_proc=NUM_PROC
            )
            datasets_dict[lang] = ds_src
            
        # Target side: CVSS (English translation)
        cvss_config = "cvss_t"
        local_cfg = f"cvss_t_{lang}"
        ds_tgt = _load_or_download_generic('cvss', 'google/cvss', cvss_config, local_cfg, split, num_samples=num_samples, start_idx=start_idx, languages=[lang])
        if ds_tgt:
            def transform_cvss(batch):
                return {
                    'id': generate_id_from_string(str(batch['id'])),
                    'audio': batch['audio'],
                    'transcription': batch['text'],
                    'language': 'en',
                    'gender': 'unknown'
                }
            ds_tgt = ds_tgt.map(
                transform_cvss, 
                remove_columns=[col for col in ds_tgt.features if col not in columns_to_keep and col != 'audio'],
                num_proc=NUM_PROC
            )
            en_targets.append(ds_tgt)

    if en_targets:
        if len(en_targets) > 1:
            datasets_dict['en'] = concatenate_datasets(en_targets)
        else:
            datasets_dict['en'] = en_targets[0]
            
    return datasets_dict

def load_data(start_idx=0, num_samples=10000, encoding=None, lang=None, split="train", dataset=None):
    if dataset is None: dataset = ['fleurs']
    if lang is None: lang = []
    
    final_datasets = collections.defaultdict(list)
    
    if 'fleurs' in dataset:
        fleurs_res = _load_fleurs_data(split, lang, 0, None)
        for l, ds in fleurs_res.items():
            final_datasets[l].append(ds)
            
    if 'seamless_align' in dataset:
        seamless_res = _load_seamless_data(split, lang, 0, None)
        for l, ds in seamless_res.items():
            final_datasets[l].append(ds)

    if 'cvss' in dataset:
        cvss_res = _load_cvss_data(split, lang, 0, None)
        for l, ds in cvss_res.items():
            final_datasets[l].append(ds)
            
    # Concatenate
    datasets = {}
    for l, ds_list in final_datasets.items():
        ds_list = [d for d in ds_list if d is not None]
        if not ds_list:
            datasets[l] = None
        elif len(ds_list) == 1:
            datasets[l] = ds_list[0]
        else:
            datasets[l] = concatenate_datasets(ds_list)
            
    # Optimized Deduplication and Filtering
    valid_id_sets = []

    def check_validity_batch(batch):
        valid_mask = []
        for audio in batch['audio']:
            try:
                is_valid = audio is not None and audio.get('array') is not None
            except Exception:
                is_valid = False
            valid_mask.append(is_valid)
        return {'id': batch['id'], 'is_valid': valid_mask}

    for lang_key in datasets:
        if datasets[lang_key] is None: continue
        
        print(f"Validating {lang_key} (checking audio & uniqueness)...")
        
        if len(datasets[lang_key]) == 0:
            print(f"  Skipping empty dataset {lang_key}")
            valid_id_sets.append(set())
            continue

        meta_ds = datasets[lang_key].map(
            check_validity_batch,
            batched=True,
            remove_columns=datasets[lang_key].column_names,
            num_proc=NUM_PROC,
            desc=f"Validating {lang_key}"
        )
        
        ids = meta_ds['id']
        is_valid = meta_ds['is_valid']
        
        current_valid_ids = set()
        seen_ids = set()
        
        for i, uid in enumerate(ids):
            if uid in seen_ids:
                continue 
            seen_ids.add(uid)
            
            if is_valid[i]:
                current_valid_ids.add(uid)
        
        print(f"  {len(current_valid_ids)} valid unique IDs found.")
        valid_id_sets.append(current_valid_ids)

    if not valid_id_sets:
        common_ids = set()
    else:
        common_ids = valid_id_sets[0]
        for s in valid_id_sets[1:]:
            common_ids.intersection_update(s)
            
    print(f"Final Count: {len(common_ids)} common valid samples.")

    # Apply slicing to common IDs
    sorted_common = sorted(list(common_ids))
    if num_samples is not None:
        end_idx = start_idx + num_samples
        sorted_common = sorted_common[start_idx:end_idx]
    else:
        sorted_common = sorted_common[start_idx:]

    for lang_key in datasets:
        if datasets[lang_key] is None: continue
        src_ids = datasets[lang_key]['id']
        id_to_idx = {} 
        for idx, uid in enumerate(src_ids):
             if uid not in id_to_idx:
                 id_to_idx[uid] = idx
        
        final_indices = [id_to_idx[uid] for uid in sorted_common if uid in id_to_idx]
        
        datasets[lang_key] = datasets[lang_key].select(final_indices)
        if len(datasets[lang_key]) > 0:
             datasets[lang_key] = datasets[lang_key].sort("id")

    if encoding is not None:
        print(f"Applying encoding: {type(encoding).__name__}")
        for lang_key, ds in datasets.items():
            if ds is None or len(ds) == 0:
                continue
                
            if isinstance(encoding, SpectrogramEncoder):
                datasets[lang_key] = ds.map(
                    apply_encoding_spectrogram, 
                    fn_kwargs={"encoder": encoding},
                    num_proc=8,
                    desc=f"Encoding {lang_key} with Spectrogram"
                )
            elif isinstance(encoding, (Wav2VecEncoder, VQGANEncoder)):
                datasets[lang_key] = ds.map(
                    apply_encoding_gpu,
                    fn_kwargs={"encoder": encoding},
                    batch_size=32,
                    desc=f"Encoding {lang_key} with {type(encoding).__name__}"
                )
            elif isinstance(encoding, SpeechT5Encoder):
                 datasets[lang_key] = ds.map(
                    apply_encoding_speecht5,
                    fn_kwargs={"encoder": encoding},
                    batched=True,
                    batch_size=32,
                    num_proc=NUM_PROC,
                    desc=f"Encoding {lang_key} with SpeechT5 Spectrograms"
                )
    
    return datasets

def verify_dataset_language(ds, target_lang, model_size="tiny", threshold=0.75):
    """
    Filters the dataset to ensure all samples match target_lang using LID.
    Uses faster-whisper for fast inference.

    Handles two audio formats that HuggingFace may produce:
      - Decoded : {'array': np.ndarray, 'sampling_rate': int}
      - Raw bytes: {'bytes': bytes,     'path': str | None}
    The seamless_align pipeline stores audio as raw MP3 bytes until the
    dataset is cast to an Audio() feature, so we must handle both.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("Warning: faster-whisper not installed. Skipping language filtering.")
        return ds

    import io

    # Always use CPU + int8 for LID.
    # CTranslate2 (used by faster-whisper) requires a cuDNN version that may not
    # be present on all systems; LID is fast enough on CPU with a tiny model.
    print(f"Initializing Whisper LID model ({model_size}) on cpu...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def _decode_audio(audio) -> np.ndarray:
        """
        Normalise an HF audio field to a 16 kHz float32 numpy array.
        Returns None if decoding fails.
        """
        try:
            if isinstance(audio, dict):
                # --- Already-decoded Audio feature ---
                if audio.get('array') is not None:
                    arr = np.array(audio['array'], dtype=np.float32)
                    sr  = audio.get('sampling_rate', 16000)
                # --- Raw bytes (MP3 / WAV etc.) ---
                elif audio.get('bytes') is not None:
                    arr, sr = sf.read(io.BytesIO(audio['bytes']))
                    arr = arr.astype(np.float32)
                    if arr.ndim > 1:           # stereo → mono
                        arr = arr.mean(axis=1)
                else:
                    return None
            else:
                return None

            # Resample to 16 kHz if needed
            if sr != 16000:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)

            return arr
        except Exception:
            return None

    def check_lang_batch(batch):
        mask = []
        for audio in batch['audio']:
            try:
                arr = _decode_audio(audio)
                if arr is None:
                    mask.append(False)
                    continue
                _, info = model.transcribe(arr, beam_size=1)
                is_match = (info.language == target_lang and info.language_probability > threshold)
                mask.append(is_match)
            except Exception as e:
                print(f"Error during LID: {e}")
                mask.append(False)
        return {"is_correct_lang": mask}

    # Single-process map to avoid spawning Whisper model in every worker
    mask_ds = ds.map(
        check_lang_batch,
        batched=True,
        batch_size=16,
        num_proc=1,
        remove_columns=ds.column_names,
        desc=f"LID Verification ({target_lang})"
    )

    valid_indices = [i for i, val in enumerate(mask_ds['is_correct_lang']) if val]
    print(f"LID Filter ({target_lang}): Kept {len(valid_indices)}/{len(ds)} samples.")
    return ds.select(valid_indices)


def play_audio(record):
  return Audio(data=record['audio']['array'], rate=record['audio']['sampling_rate'])

def save_audio(record, path='audio.wav'):
    sf.write(path, record['audio']['array'], record['audio']['sampling_rate'])  