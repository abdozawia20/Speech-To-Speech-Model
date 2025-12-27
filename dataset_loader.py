from datasets import load_dataset, concatenate_datasets, DownloadConfig, IterableDataset, Dataset, load_from_disk, load_dataset_builder
from IPython.display import Audio
import sys
import os
import collections
import torch
import numpy as np
from encoders import *



def transform_fleurs_internal(batch):
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

DATASETS_DIR = "./datasets"

def load_or_download_fleurs(lang_config, split):
    """
    Checks if the dataset exists locally. If not, asks for confirmation to download and save it.
    """
    dataset_name = "google/fleurs"
    local_path = os.path.join(DATASETS_DIR, "fleurs", lang_config, split)

    if os.path.exists(local_path):
        print(f"Loading {dataset_name} ({lang_config}) from local storage: {local_path}...")
        return load_from_disk(local_path)
    
    print(f"Dataset {dataset_name} ({lang_config}) not found locally at {local_path}.")
    
    # Estimate size
    try:
        builder = load_dataset_builder(dataset_name, lang_config, trust_remote_code=True)
        # Note: This size is often an estimate or might be for the full dataset, not just the split.
        # But it gives an idea.
        download_size = builder.info.download_size
        dataset_size = builder.info.dataset_size
        size_info = f"Download size: {download_size / 1024**3:.2f} GB, Generated size: {dataset_size / 1024**3:.2f} GB" if download_size and dataset_size else "Size info unavailable (typically ~1.5 GB per language config)"
    except Exception as e:
        size_info = f"Could not determine size: {e}. (Typically ~1.5 GB per language config)"

    print(f"This will download and save the dataset to {local_path}.")
    print(size_info)
    
    response = input("Do you want to proceed with the download? (Y/n): ").strip().lower()
    if response not in ('', 'y', 'yes'):
        print("Download aborted by user.")
        return None

    print(f"Downloading {dataset_name} ({lang_config})... This may take a while.")
    # Download as Map-style dataset (streaming=False)
    config = DownloadConfig(resume_download=True, max_retries=10)
    dataset = load_dataset(dataset_name, lang_config, split=split, streaming=False, trust_remote_code=True, download_config=config)
    
    print(f"Saving to {local_path}...")
    dataset.save_to_disk(local_path)
    print("Save complete.")
    
    return dataset
    
def load_data(start_idx=0, num_samples=10000, encoding=None, lang=None, split="train"):
    # Define columns to keep in the final harmonized schema
    columns_to_keep = ['id', 'audio', 'transcription', 'language', 'gender']

    # Dataset 1: Fleurs
    fleurs_en = load_or_download_fleurs("en_us", split) if 'en' in lang else None
    fleurs_ar = load_or_download_fleurs("ar_eg", split) if 'ar' in lang else None
    fleurs_tr = load_or_download_fleurs("tr_tr", split) if 'tr' in lang else None

    # Dataset 2: Voxpopuli (Only Supports English)
    # TODO: enable voxpopuli later
    # voxpopuli_en = load_dataset("facebook/voxpopuli", "en", split=split, streaming=True, trust_remote_code=True, download_config=config)




    # ******************************ENGLISH TRANSFORMATIONS******************************

    # Apply transformations to fleurs_en_dataset
    fleurs_en_transformed = fleurs_en.map(
        transform_fleurs_internal,
        remove_columns=[col for col in fleurs_en.features if col not in columns_to_keep and col != 'audio']
    ) if fleurs_en is not None else None

    # Apply transformations to voxpopuli_en_dataset
    # voxpopuli_en_transformed = voxpopuli_en.map(
    #     transform_voxpopuli_internal,
    #     remove_columns=[col for col in voxpopuli_en.features if col not in columns_to_keep and col != 'audio']
    # )

    # ******************************ARABIC TRANSFORMATIONS******************************

    # Apply transformations to fleurs_ar_dataset
    fleurs_ar_transformed = fleurs_ar.map(
        transform_fleurs_internal,
        remove_columns=[col for col in fleurs_ar.features if col not in columns_to_keep and col != 'audio']
    ) if fleurs_ar is not None else None

    # ******************************TURKISH TRANSFORMATIONS******************************

    # Apply transformations to fleurs_tr_dataset
    fleurs_tr_transformed = fleurs_tr.map(
        transform_fleurs_internal,
        remove_columns=[col for col in fleurs_tr.features if col not in columns_to_keep and col != 'audio']
    ) if fleurs_tr is not None else None

    # Combine datasets for each language
    combined_en = fleurs_en_transformed if fleurs_en is not None else None
    combined_ar = concatenate_datasets([fleurs_ar_transformed]) if fleurs_ar is not None else None
    combined_tr = concatenate_datasets([fleurs_tr_transformed]) if fleurs_tr is not None else None

    # Convert to standard buffer-backed Datasets
    # Note: Since they are already Map-style (from disk), we might not need buffer_to_dataset anymore.
    
    datasets = {
        "en": combined_en, 
        "ar": combined_ar, 
        "tr": combined_tr
    }

    # Deduplicate datasets by ID to ensure unique samples
    for lang in datasets:
        if datasets[lang] is not None:
             ids = datasets[lang]['id']
             unique_indices = []
             seen = set()
             for idx, val in enumerate(ids):
                 if val not in seen:
                     seen.add(val)
                     unique_indices.append(idx)
             
             if len(unique_indices) < len(datasets[lang]):
                 print(f"Deduplicating {lang}: {len(datasets[lang])} -> {len(unique_indices)} unique IDs")
                 datasets[lang] = datasets[lang].select(unique_indices)

    # Filter for parallel data (intersection of IDs) if multiple languages are loaded
    active_datasets = [ds for ds in datasets.values() if ds is not None]
    if len(active_datasets) > 1:
        # Get set of IDs from the first dataset
        common_ids = set(active_datasets[0]['id'])
        
        # Intersect with all other datasets
        for ds in active_datasets[1:]:
            common_ids.intersection_update(set(ds['id']))
            
        print(f"Found {len(common_ids)} common IDs across {len(active_datasets)} languages.")
        
        # Filter all datasets to keep only common IDs
        for key in datasets:
             if datasets[key] is not None:
                 datasets[key] = datasets[key].filter(lambda x: x['id'] in common_ids)

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

    for lang in datasets:
        datasets[lang] = slice_dataset(datasets[lang], start_idx, num_samples)

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
                    batched=True,
                    batch_size=32,
                    desc=f"Encoding {lang} with {type(encoding).__name__}"
                )
    
    return datasets

def play_audio(record):
  return Audio(data=record['audio']['array'], rate=record['audio']['sampling_rate'])