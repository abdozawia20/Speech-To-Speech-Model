from datasets import load_dataset, concatenate_datasets, DownloadConfig, IterableDataset, Dataset, load_from_disk, load_dataset_builder
from IPython.display import Audio
import sys
import os
import collections
import torch
import numpy as np
from encoders import *
project_root = os.path.abspath(os.path.join(os.getcwd(), ''))
sys.path.append(project_root)




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

def load_or_download_dataset(lang_config, split, datasets=None, num_samples=None, context_langs=None):
    """
    Checks if the specified datasets exist locally. If not, asks for confirmation to download and save them.
    Args:
        lang_config (str): Language configuration (e.g., "en_us").
        split (str): Split to load (e.g., "train", "test").
        datasets (list): List of dataset keys to load. Options: ['fleurs', 'seamless_align']. Default: ['fleurs']
        num_samples (int): Number of samples to load (slicing).
        context_langs (list): List of all languages being loaded (used for pair detection).
    """
    if datasets is None:
        datasets = ['fleurs']

    DATASET_MAP = {
        'fleurs': 'google/fleurs',
        'seamless_align': 'jhu-clsp/seamless-align-expressive'
    }

    loaded_datasets = []

    # Supported languages/pairs for Seamless Align Expressive
    # Repo structure uses specific folder names like deA-enA
    SEAMLESS_PAIRS = {'deA-enA', 'enA-esA', 'enA-frA', 'enA-itA', 'enA-zhA'}
    
    # Mapping simple pairs to repo folder names
    # Key: sorted tuple of (lang1, lang2), Value: repo_folder_name
    PAIR_MAPPING = {
        tuple(sorted(('de', 'en'))): 'deA-enA',
        tuple(sorted(('en', 'es'))): 'enA-esA',
        tuple(sorted(('en', 'fr'))): 'enA-frA',
        tuple(sorted(('en', 'it'))): 'enA-itA',
        tuple(sorted(('en', 'zh'))): 'enA-zhA',
    }
    
    # Also support single languages map to 'default' if no pair found, but prioritizing pairs.
    SEAMLESS_SINGLE_SUPPORTED = {'en', 'de', 'es', 'fr', 'it', 'zh'}

    # Use a local cache directory on the external drive to avoid filling up primary storage
    cache_dir = os.path.join(DATASETS_DIR, ".cache")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    for ds_key in datasets:
        if ds_key not in DATASET_MAP:
            print(f"Warning: Unknown dataset key '{ds_key}'. Skipping.")
            continue
        
        dataset_full_name = DATASET_MAP[ds_key]
        target_config = lang_config
        target_split = split
        use_shared_path = False # Flag for shared pair storage
        
        # We separate the config used for storage naming vs the config used for HuggingFace download
        hf_download_config = lang_config 
        data_files_arg = None

        # Specific handling for Seamless Align
        if ds_key == 'seamless_align':
            lang_code = lang_config[:2] # e.g. "en_us" -> "en"
            hf_download_config = "default" # Always use default for HF call
            
            # 1. Try to find a pairing from context_langs
            detected_pair = None
            if context_langs:
                # Collect all lang codes from context
                ctx_codes = set(l[:2] for l in context_langs)
                if lang_code in ctx_codes:
                    # Look for any matching pair in mapping involving this lang
                    # We pick the first valid pair we find that is fully present in context
                    for other_code in ctx_codes:
                        if other_code == lang_code: 
                            continue
                        
                        pair_key = tuple(sorted((lang_code, other_code)))
                        if pair_key in PAIR_MAPPING:
                            detected_pair = PAIR_MAPPING[pair_key]
                            break
            
            if detected_pair:
                target_config = detected_pair
                print(f"Seamless Align: Detected pair '{detected_pair}' for language '{lang_config}'.")
                use_shared_path = True
                
                # Construct data_files to restrict download
                # We interpret data_files as full URLs to ensure specific file fetching
                base_url = "https://huggingface.co/datasets/jhu-clsp/seamless-align-expressive/resolve/main"
                data_files_arg = [
                    f"{base_url}/data/{detected_pair}/src.tar.gz",
                    f"{base_url}/data/{detected_pair}/tgt.tar.gz"
                ]
            else:
                # Fallback to single language check
                if lang_code not in SEAMLESS_SINGLE_SUPPORTED:
                    print(f"Warning: Language '{lang_config}' is not supported by {ds_key} (and no supported pair found). Skipping.")
                    continue
                
                # If no pair found but supported, we fallback to default naming for storage too
                target_config = "default"

        # Directory structure: 
        # For pairs: datasets/seamless_align/{pair_config}/{split}  (Shared)
        # For others: datasets/{dataset_key}/{lang_config}/{split}
        if use_shared_path:
             local_path = os.path.join(DATASETS_DIR, ds_key, target_config, split)
        else:
             local_path = os.path.join(DATASETS_DIR, ds_key, lang_config, split)

        if os.path.exists(local_path):
            print(f"Loading {dataset_full_name} ({target_config}) from local storage: {local_path}...")
            ds = load_from_disk(local_path)
            # Apply slicing if requested
            if num_samples is not None and len(ds) > num_samples and ds_key != 'seamless_align':
                print(f"Slicing loaded dataset to {num_samples} samples...")
                ds = ds.select(range(num_samples))
            loaded_datasets.append(ds)
            continue
        
        print(f"Dataset {dataset_full_name} ({target_config}) not found locally at {local_path}.")
        
        # Estimate size
        try:
            builder = load_dataset_builder(dataset_full_name, hf_download_config, trust_remote_code=True, cache_dir=cache_dir)
            download_size = builder.info.download_size
            dataset_size = builder.info.dataset_size
            size_info = f"Download size: {download_size / 1024**3:.2f} GB, Generated size: {dataset_size / 1024**3:.2f} GB" if download_size and dataset_size else "Size info unavailable (typically ~1.5 GB per language config)"
        except Exception as e:
            size_info = f"Could not determine size: {e}. (Typically ~1.5 GB per language config)"

        print(f"This will download and save the dataset to {local_path}.")
        print(f"Download Cache: {cache_dir}")
        print(size_info)
        
        response = input(f"Do you want to proceed with the download of {ds_key}? (Y/n): ").strip().lower()
        if response not in ('', 'y', 'yes'):
            print(f"Download of {ds_key} aborted by user.")
            continue

        print(f"Downloading {dataset_full_name} (config: {hf_download_config})... This may take a while.")
        config = DownloadConfig(resume_download=True, max_retries=10)
        try:
            # For Seamless, if it has no splits, split=None might return DatasetDict or single Dataset
            dataset = load_dataset(dataset_full_name, hf_download_config, split=target_split, streaming=False, trust_remote_code=True, download_config=config, cache_dir=cache_dir, data_files=data_files_arg)
            
            print(f"Saving to {local_path}...")
            dataset.save_to_disk(local_path)
            print("Save complete.")
            
            # Apply slicing if requested (after saving full)
            # Skip slicing for Seamless Align because we need to filter src/tgt first (which happens later)
            if num_samples is not None and len(dataset) > num_samples and ds_key != 'seamless_align':
                 dataset = dataset.select(range(num_samples))
                 
            loaded_datasets.append(dataset)
        except Exception as e:
            print(f"Failed to download/save {dataset_full_name}: {e}")

    if not loaded_datasets:
        return None

    if len(loaded_datasets) == 1:
        return loaded_datasets[0]

    return concatenate_datasets(loaded_datasets)
    
def load_data(start_idx=0, num_samples=10000, encoding=None, lang=None, split="train", dataset=None):
    # Define columns to keep in the final harmonized schema
    columns_to_keep = ['id', 'audio', 'transcription', 'language', 'gender']

    # Dataset 1: Fleurs
    # Pass 'lang' as context_langs to help identifying pairs
    raw_data_en = load_or_download_dataset("en_us", split, datasets=dataset, num_samples=num_samples, context_langs=lang) if 'en' in lang else None
    raw_data_ar = load_or_download_dataset("ar_eg", split, datasets=dataset, num_samples=num_samples, context_langs=lang) if 'ar' in lang else None
    raw_data_tr = load_or_download_dataset("tr_tr", split, datasets=dataset, num_samples=num_samples, context_langs=lang) if 'tr' in lang else None
    raw_data_de = load_or_download_dataset("de_de", split, datasets=dataset, num_samples=num_samples, context_langs=lang) if 'de' in lang else None

    # ******************************SEAMLESS ALIGN PROCESSING******************************
    # Seamless datasets are loaded as raw files (src/..., tgt/...). 
    # We need to split them into language-specific datasets and map columns.
    
    def process_seamless(ds, lang_code, config_name, num_samples=None):
        if ds is None: return None
        if 'mp3' not in ds.column_names: return ds # Not seamless or already processed
        
        # Determine if lang_code is src or tgt in config (e.g. deA-enA)
        # config is like "deA-enA"
        try:
            parts = config_name.split('-')
            src_lang = parts[0][:2] # deA -> de
            tgt_lang = parts[1][:2] # enA -> en
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
                    # src/XX... -> ID is k[4:]
                    src_indices[k[4:]] = idx
                elif k.startswith('tgt/'):
                    tgt_indices[k[4:]] = idx
            
            # Find intersection
            common_ids = sorted(list(set(src_indices.keys()) & set(tgt_indices.keys())))
            
            # Select indices for the requested side
            if prefix == 'src/':
                final_indices = [src_indices[uid] for uid in common_ids]
            else:
                final_indices = [tgt_indices[uid] for uid in common_ids]

            # Slice indices here to optimize mapping speed
            if num_samples is not None and len(final_indices) > num_samples:
                final_indices = final_indices[:num_samples]

            ds = ds.select(final_indices)
            
            # Map columns
            def map_seamless_cols(batch):
                # ID extraction must match the logic above
                # key is like src/00/219/939
                key = batch['__key__']
                clean_id = key.replace(prefix, '').replace('/', '')
                
                # Check if we can convert to int, if not hash it
                try:
                    int_id = int(clean_id)
                except:
                    # simplistic hash to int
                    int_id = int(hash(clean_id) % 1e16)
                
                return {
                    'id': int_id,
                    'audio': batch['mp3'], 
                    'language': lang_code,
                    'gender': 'unknown', 
                    'transcription': '' 
                }

            ds = ds.map(map_seamless_cols, remove_columns=['mp3', '__key__', '__url__'])
        return ds

    # Process loaded datasets if they are Seamless
    if dataset and 'seamless_align' in dataset:

        def get_seamless_config(target_lang, context):
            # Same logic as load_or_download
            tgt2 = target_lang[:2]
            if not context: return "default"
            for c in context:
                c2 = c[:2]
                if c2 == tgt2: continue
                p1 = f"{tgt2}-{c2}"
                if p1 in load_or_download_dataset.__globals__.get('SEAMLESS_PAIRS', {'de-en'}):
                     pass 
            ctx_codes = set(l[:2] for l in context)
            if 'de' in ctx_codes and 'en' in ctx_codes: return "deA-enA"
            if 'en' in ctx_codes and 'es' in ctx_codes: return "enA-esA"
            if 'en' in ctx_codes and 'fr' in ctx_codes: return "enA-frA"
            if 'en' in ctx_codes and 'it' in ctx_codes: return "enA-itA"
            if 'en' in ctx_codes and 'zh' in ctx_codes: return "enA-zhA"
            return "default"

        pair_config = get_seamless_config("en", lang) # Config is same for all participating langs
        
        pair_config = get_seamless_config("en", lang) # Config is same for all participating langs
        
        raw_data_en = process_seamless(raw_data_en, 'en', pair_config, num_samples)
        raw_data_de = process_seamless(raw_data_de, 'de', pair_config, num_samples)
        raw_data_ar = process_seamless(raw_data_ar, 'ar', pair_config, num_samples) 
        raw_data_tr = process_seamless(raw_data_tr, 'tr', pair_config, num_samples)


    # ******************************ENGLISH TRANSFORMATIONS******************************

    # Apply transformations to fleurs_en_dataset
    en_transformed = raw_data_en.map(
        transform_internal,
        remove_columns=[col for col in raw_data_en.features if col not in columns_to_keep and col != 'audio'],
        num_proc=os.cpu_count() or 4
    ) if raw_data_en is not None else None

    # ******************************ARABIC TRANSFORMATIONS******************************

    # Apply transformations to fleurs_ar_dataset
    ar_transformed = raw_data_ar.map(
        transform_internal,
        remove_columns=[col for col in raw_data_ar.features if col not in columns_to_keep and col != 'audio'],
        num_proc=os.cpu_count() or 4
    ) if raw_data_ar is not None else None

    # ******************************TURKISH TRANSFORMATIONS******************************

    # Apply transformations to fleurs_tr_dataset
    tr_transformed = raw_data_tr.map(
        transform_internal,
        remove_columns=[col for col in raw_data_tr.features if col not in columns_to_keep and col != 'audio'],
        num_proc=os.cpu_count() or 4
    ) if raw_data_tr is not None else None

    # ******************************GERMAN TRANSFORMATIONS******************************

    # Apply transformations to fleurs_de_dataset
    de_transformed = raw_data_de.map(
        transform_internal,
        remove_columns=[col for col in raw_data_de.features if col not in columns_to_keep and col != 'audio'],
        num_proc=os.cpu_count() or 4
    ) if raw_data_de is not None else None

    # Combine datasets for each language
    combined_en = en_transformed if raw_data_en is not None else None
    combined_ar = ar_transformed if raw_data_ar is not None else None
    combined_tr = tr_transformed if raw_data_tr is not None else None
    combined_de = de_transformed if raw_data_de is not None else None

    # Convert to standard buffer-backed Datasets
    datasets = {
        "en": combined_en, 
        "ar": combined_ar, 
        "tr": combined_tr,
        "de": combined_de
    }

    # Optimized Deduplication and Filtering
    # Goal: Identify valid samples (unique, valid audio, shared across languages) WITHOUT rewriting the heavy dataset multiple times.
    
    valid_id_sets = []
    
    # 1. Parallel Validation Pass
    # We create a lightweight 'metadata' dataset for each language containing only ID and Validity.
    # This forces audio decoding (validation) but avoids writing the decoded audio back to disk.
    
    def check_validity_batch(batch):
        valid_mask = []
        for audio in batch['audio']:
            try:
                # Check for None or None array
                is_valid = audio is not None and audio.get('array') is not None
            except Exception:
                is_valid = False
            valid_mask.append(is_valid)
        return {'id': batch['id'], 'is_valid': valid_mask}

    for lang in datasets:
        if datasets[lang] is None: continue
        
        print(f"Validating {lang} (checking audio & uniqueness)...")
        
        # Run parallel map to get validity flags
        # We drop all columns except the new ones to minimize overhead
        meta_ds = datasets[lang].map(
            check_validity_batch,
            batched=True,
            remove_columns=datasets[lang].column_names,
            num_proc=os.cpu_count() or 4,
            desc=f"Validating {lang}"
        )
        
        # Now process metadata in memory (fast)
        ids = meta_ds['id']
        is_valid = meta_ds['is_valid']
        
        # Deduplicate and filter valid
        current_valid_ids = set()
        seen_ids = set()
        
        for i, uid in enumerate(ids):
            if uid in seen_ids:
                continue # Duplicate
            seen_ids.add(uid)
            
            if is_valid[i]:
                current_valid_ids.add(uid)
        
        print(f"  {len(current_valid_ids)} valid unique IDs found.")
        valid_id_sets.append(current_valid_ids)

    # 2. Intersection (Find common valid IDs)
    if not valid_id_sets:
        common_ids = set()
    else:
        common_ids = valid_id_sets[0]
        for s in valid_id_sets[1:]:
            common_ids.intersection_update(s)
            
    print(f"Final Count: {len(common_ids)} common valid samples.")

    for lang in datasets:
        if datasets[lang] is None: continue
        src_ids = datasets[lang]['id']
        id_to_idx = {} # Map ID -> First Index
        for idx, uid in enumerate(src_ids):
             if uid not in id_to_idx:
                 id_to_idx[uid] = idx
        
        # Collect indices for sorted common_ids
        sorted_common = sorted(list(common_ids))
        final_indices = [id_to_idx[uid] for uid in sorted_common if uid in id_to_idx]
        
        datasets[lang] = datasets[lang].select(final_indices)


    # Sort datasets by id
    for language in datasets:
        if datasets[language] is not None and len(datasets[language]) > 0:
             datasets[language] = datasets[language].sort("id")

    # Slice datasets after sorting to ensure determinstic and aligned output
    # Apply slicing manually using select/indices since they are Map-style datasets now
    def slice_dataset(ds, start, num):
        if ds is None:
            return None
        # Ensure we don't go out of bounds
        end = min(start + num, len(ds))
        if start >= len(ds):
            return ds.select([]) # Empty
        return ds.select(range(start, end))

    
    pass

    if encoding is not None:
        print(f"Applying encoding: {type(encoding).__name__}")
        for lang, ds in datasets.items():
            if ds is None:
                continue
            if len(ds) == 0:
                continue
                
            if isinstance(encoding, SpectrogramEncoder):
                # Spectrograms on CPU with multiprocessing
                # Note: 'audio' column type might change from Audio() to Array/Sequence
                # We trust map to handle schema inference or we might need to cast if it fails.
                datasets[lang] = ds.map(
                    apply_encoding_spectrogram, 
                    fn_kwargs={"encoder": encoding},
                    num_proc=8,
                    desc=f"Encoding {lang} with Spectrogram"
                )
            elif isinstance(encoding, (Wav2VecEncoder, VQGANEncoder)):
                # Neural models on GPU with batching
                datasets[lang] = ds.map(
                    apply_encoding_gpu,
                    fn_kwargs={"encoder": encoding},
                    batch_size=32,
                    desc=f"Encoding {lang} with {type(encoding).__name__}"
                )
            elif isinstance(encoding, SpeechT5Encoder):
                 datasets[lang] = ds.map(
                    apply_encoding_speecht5,
                    fn_kwargs={"encoder": encoding},
                    batched=True,
                    batch_size=32,
                    num_proc=os.cpu_count() or 4,
                    desc=f"Encoding {lang} with SpeechT5 Spectrograms"
                )
    
    return datasets

def play_audio(record):
  return Audio(data=record['audio']['array'], rate=record['audio']['sampling_rate'])