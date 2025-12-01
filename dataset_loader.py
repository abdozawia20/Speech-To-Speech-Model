from datasets import load_dataset, concatenate_datasets, DownloadConfig
from IPython.display import Audio

def transform_fleurs_internal(batch):
    batch['id'] = str(batch['id'])
    batch['gender'] = str(batch['gender'])
    batch['language'] = str(batch['language'])
    return batch

def transform_voxpopuli_internal(batch):
    batch['id'] = batch['audio_id']
    batch['transcription'] = batch['normalized_text']
    batch['language'] = str(batch['language'])
    return batch

def load_train_data():
    # Define columns to keep in the final harmonized schema
    columns_to_keep = ['id', 'audio', 'transcription', 'language', 'gender']

    # Configure download settings to be more robust against network timeouts
    config = DownloadConfig(resume_download=True, max_retries=10)

    # Dataset 1: Fleurs
    fleurs_en = load_dataset("google/fleurs", "en_us", split="train", trust_remote_code=True, streaming=True, download_config=config)
    fleurs_ar = load_dataset("google/fleurs", "ar_eg", split="train", trust_remote_code=True, streaming=True, download_config=config)
    fleurs_tr = load_dataset("google/fleurs", "tr_tr", split="train", trust_remote_code=True, streaming=True, download_config=config)

    # Dataset 2: Voxpopuli (Only Supports English)
    voxpopuli_en = load_dataset("facebook/voxpopuli", "en", split="train", trust_remote_code=True, streaming=True, download_config=config)

    # ******************************ENGLISH TRANSFORMATIONS******************************

    # Apply transformations to fleurs_en_dataset
    fleurs_en_transformed = fleurs_en.map(
        transform_fleurs_internal,
        remove_columns=[col for col in fleurs_en.features if col not in columns_to_keep and col != 'audio'],
        batched=False
    )

    # Apply transformations to voxpopuli_en_dataset
    voxpopuli_en_transformed = voxpopuli_en.map(
        transform_voxpopuli_internal,
        remove_columns=[col for col in voxpopuli_en.features if col not in columns_to_keep and col != 'audio'],
        batched=False
    )

    # ******************************ARABIC TRANSFORMATIONS******************************

    # Apply transformations to fleurs_ar_dataset
    fleurs_ar_transformed = fleurs_ar.map(
        transform_fleurs_internal,
        remove_columns=[col for col in fleurs_ar.features if col not in columns_to_keep and col != 'audio'],
        batched=False
    )

    # ******************************TURKISH TRANSFORMATIONS******************************

    # Apply transformations to fleurs_tr_dataset
    fleurs_tr_transformed = fleurs_tr.map(
        transform_fleurs_internal,
        remove_columns=[col for col in fleurs_tr.features if col not in columns_to_keep and col != 'audio'],
        batched=False
    )

    # Combine datasets for each language
    combined_en = concatenate_datasets([fleurs_en_transformed, voxpopuli_en_transformed])
    combined_ar = concatenate_datasets([fleurs_ar_transformed])
    combined_tr = concatenate_datasets([fleurs_tr_transformed])

    return {"en": combined_en, "ar": combined_ar, "tr": combined_tr}

def play_audio(record):
  return Audio(data=record['audio']['array'], rate=record['audio']['sampling_rate'])