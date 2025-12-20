from datasets import load_dataset, concatenate_datasets, DownloadConfig, IterableDataset, Dataset
from IPython.display import Audio
import sys
import collections
import torch
import numpy as np
from encoders import *

# Memory limit in bytes (10GB)
MEMORY_LIMIT = 10 * 1024 * 1024 * 1024

def get_item_size(item):
    """Estimate size of the item."""
    size = 0
    if isinstance(item, dict):
        for key, value in item.items():
            if hasattr(value, 'nbytes'):
                    size += value.nbytes
            elif isinstance(value, (str, bytes)):
                    size += sys.getsizeof(value)
            # primitive estimation for other types if needed
    else:
        size = sys.getsizeof(item)
    return size

def buffer_to_dataset(iterable_ds, max_memory_bytes=MEMORY_LIMIT):
    """
    Consumes an iterable dataset into a list up to max_memory_bytes and converts it 
    to a standard Hugging Face Dataset (Map-style).
    """
    buffer = []
    current_memory = 0
    
    print("Buffering data into memory...")
    iterator = iter(iterable_ds)
    
    try:
        while current_memory < max_memory_bytes:
            item = next(iterator)
            item_size = get_item_size(item)
            
            # Check if adding this item would exceed limit (soft check)
            if current_memory + item_size > max_memory_bytes:
                print(f"Memory limit reached ({current_memory / 1024**3:.2f} GB). Stopping buffering.")
                break
                
            buffer.append(item)
            current_memory += item_size
    except StopIteration:
        pass
        
    print(f"Buffered {len(buffer)} samples ({current_memory / 1024**2:.2f} MB). Converting to Dataset...")
    
    if not buffer:
        print("Warning: Buffer is empty!")
        return Dataset.from_dict({})
        
    return Dataset.from_list(buffer)

def transform_fleurs_internal(batch):
    batch['id'] = str(batch['id'])
    batch['gender'] = str(batch['gender'])
    batch['language'] = str(batch['language'])
    return batch

def transform_voxpopuli_internal(batch):
    batch['id'] = batch['audio_id']
    batch['transcription'] = batch['normalized_text']
    batch['language'] = str(batch['language'])
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

def load_data(start_idx=0, num_samples=10000, encoding=None, lang=None, split="train"):
    # Define columns to keep in the final harmonized schema
    columns_to_keep = ['id', 'audio', 'transcription', 'language', 'gender']

    # Configure download settings to be more robust against network timeouts
    config = DownloadConfig(resume_download=True, max_retries=10)

    # Dataset 1: Fleurs
    # Streaming mode: remove split slicing, use streaming=True
    fleurs_en = load_dataset("google/fleurs", "en_us", split=split, streaming=True, trust_remote_code=True, download_config=config) if 'en' in lang else None
    fleurs_ar = load_dataset("google/fleurs", "ar_eg", split=split, streaming=True, trust_remote_code=True, download_config=config) if 'ar' in lang else None
    fleurs_tr = load_dataset("google/fleurs", "tr_tr", split=split, streaming=True, trust_remote_code=True, download_config=config) if 'tr' in lang else None

    # Dataset 2: Voxpopuli (Only Supports English)
    # TODO: enable voxpopuli later
    # voxpopuli_en = load_dataset("facebook/voxpopuli", "en", split=split, streaming=True, trust_remote_code=True, download_config=config)

    # Apply slicing manually using skip/take
    fleurs_en = fleurs_en.skip(start_idx).take(num_samples) if 'en' in lang else None
    fleurs_ar = fleurs_ar.skip(start_idx).take(num_samples) if 'ar' in lang else None
    fleurs_tr = fleurs_tr.skip(start_idx).take(num_samples) if 'tr' in lang else None
    # voxpopuli_en = voxpopuli_en.skip(start_idx).take(num_samples)


    # ******************************ENGLISH TRANSFORMATIONS******************************

    # Apply transformations to fleurs_en_dataset
    fleurs_en_transformed = fleurs_en.map(
        transform_fleurs_internal,
        remove_columns=[col for col in fleurs_en.features if col not in columns_to_keep and col != 'audio']
    ) if 'en' in lang else None

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
    ) if 'ar' in lang else None

    # ******************************TURKISH TRANSFORMATIONS******************************

    # Apply transformations to fleurs_tr_dataset
    fleurs_tr_transformed = fleurs_tr.map(
        transform_fleurs_internal,
        remove_columns=[col for col in fleurs_tr.features if col not in columns_to_keep and col != 'audio']
    ) if 'tr' in lang else None

    # Combine datasets for each language
    # Interleave is used for streaming datasets instead of concatenate
    # Note: concatenate_datasets works for IterableDataset in recent versions (it essentially chains them), 
    # but interleave might be better if we want mixed loading. However, the original code used concatenate.
    # We will use concatenate_datasets which supports IterableDataset
    
    # combined_en = concatenate_datasets([fleurs_en_transformed, voxpopuli_en_transformed])
    combined_en = fleurs_en_transformed if 'en' in lang else None
    combined_ar = concatenate_datasets([fleurs_ar_transformed]) if 'ar' in lang else None
    combined_tr = concatenate_datasets([fleurs_tr_transformed]) if 'tr' in lang else None

    # Convert to standard buffer-backed Datasets
    datasets = {
        "en": buffer_to_dataset(combined_en) if 'en' in lang else None, 
        "ar": buffer_to_dataset(combined_ar) if 'ar' in lang else None, 
        "tr": buffer_to_dataset(combined_tr) if 'tr' in lang else None
    }

    if encoding is not None:
        print(f"Applying encoding: {type(encoding).__name__}")
        for lang, ds in datasets.items():
            if ds is None:
                continue
            if len(ds) == 0:
                continue
                
            if isinstance(encoding, SpectogramEncoder):
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