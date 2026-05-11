import os
import torch
import torchaudio
import numpy as np
import dataset_loader

# Fix for speechbrain dependency on torchaudio.list_audio_backends
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
dataset_loader.NUM_PROC = 1
SOURCE_LANG = "en"
TARGET_LANG = "de"
OUTPUT_DIR = dataset_loader.DATASETS_DIR

# WavLM model for SOURCE encoding (English → continuous hidden states)
WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"

# SpeechT5 processor for TARGET encoding (German → 80-bin log-mel spectrogram)
SPEECHT5_MODEL_NAME = "microsoft/speecht5_vc"

# ---------------------------------------------------------------------------
# Global state for worker processes
# ---------------------------------------------------------------------------
_wavlm_proc_cache = None       # Wav2Vec2FeatureExtractor for WavLM
_wavlm_model_cache = None      # Frozen WavLMModel backbone
_speecht5_proc_cache = None    # SpeechT5Processor (mel-spectrogram extractor)
_spk_model_cache = None        # SpeechBrain X-Vector model


# ---------------------------------------------------------------------------
# SOURCE modality: WavLM hidden states  (English → [Seq_Len, 768])
# ---------------------------------------------------------------------------

def _get_wavlm_model():
    """
    Lazy-load and cache the WavLM feature extractor and frozen WavLMModel.
    Moves the model to GPU if available for faster preprocessing.
    """
    global _wavlm_proc_cache, _wavlm_model_cache
    if _wavlm_model_cache is None:
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading WavLM ({WAVLM_MODEL_NAME}) on {device} in worker (PID: {os.getpid()})...")
        
        _wavlm_proc_cache = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_NAME)
        _wavlm_model_cache = WavLMModel.from_pretrained(WAVLM_MODEL_NAME).to(device)
        
        # Freeze all WavLM parameters
        for p in _wavlm_model_cache.parameters():
            p.requires_grad_(False)
        _wavlm_model_cache.eval()
        
    return _wavlm_proc_cache, _wavlm_model_cache


def _get_spk_model():
    """Lazy-load the SpeechBrain x-vector model for speaker embedding extraction."""
    global _spk_model_cache
    if _spk_model_cache is None:
        from speechbrain.inference.speaker import EncoderClassifier
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading X-Vector model on {device}...")
        _spk_model_cache = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-xvect-voxceleb",
            savedir="tmp_spkrec",
            run_opts={"device": str(device)}
        )
    return _spk_model_cache


def process_source_wavlm(batch):
    """
    MODALITY BRIDGE — Source (English audio → WavLM continuous hidden states).

    WavLM replaces SpeechT5's built-in CNN feature extractor entirely.
    Output shape per sample: (Seq_Len, 768)
    """
    proc, model = _get_wavlm_model()
    device = next(model.parameters()).device

    encoded_list = []
    with torch.no_grad():
        for audio_item in batch["audio"]:
            audio_array = np.array(audio_item["array"], dtype=np.float32)
            sr = audio_item["sampling_rate"]

            # Resample to 16 kHz
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

            # Normalise and convert to model input on the correct device
            inputs = proc(audio_array, sampling_rate=16000, return_tensors="pt", padding=False)
            input_values = inputs.input_values.to(device)
            
            out = model(input_values)                    # forward through frozen WavLM
            hidden = out.last_hidden_state.squeeze(0)    # (Seq_Len, 768)
            encoded_list.append(hidden.cpu().numpy())    # move back to CPU for storage

    return {"input_values": encoded_list}


# ---------------------------------------------------------------------------
# TARGET modality: 80-bin log-mel spectrogram  (German audio → [T, 80])
# ---------------------------------------------------------------------------

def _get_speecht5_processor():
    """Lazy-load and cache the official SpeechT5Processor."""
    global _speecht5_proc_cache
    if _speecht5_proc_cache is None:
        from transformers import SpeechT5Processor
        print(f"Loading SpeechT5 Processor in worker (PID: {os.getpid()})...")
        _speecht5_proc_cache = SpeechT5Processor.from_pretrained(SPEECHT5_MODEL_NAME)
    return _speecht5_proc_cache


# ---------------------------------------------------------------------------
# SOURCE modality: WavLM hidden states  (English → [Seq_Len, 768])
# ---------------------------------------------------------------------------


def process_target_mel(batch):
    """
    MODALITY BRIDGE — Target (German audio → 80-bin log-mel spectrogram + Speaker Embedding).

    Extracts:
      - 'labels': Official normalized 80-bin log-mel spectrogram (T, 80)
      - 'speaker_embeddings': 512-dim x-vector for the specific sample
    """
    import torch
    import numpy as np
    processor = _get_speecht5_processor()
    spk_model = _get_spk_model()
    device = spk_model.device

    mel_list = []
    spk_embeds = []
    
    for audio_item in batch["audio"]:
        audio_array = np.array(audio_item["array"], dtype=np.float32).flatten()
        sr = audio_item["sampling_rate"]

        # 1. Official Mel Spectrogram (T, 80)
        # The processor handles resampling, mel-extraction, and normalization.
        inputs = processor(audio_target=audio_array, sampling_rate=sr, return_tensors="pt")
        # Squeeze batch dim -> (T, 80)
        log_mel = inputs.input_values[0]
        mel_list.append(log_mel.numpy())

        # 2. Speaker Embedding (512,)
        # For Speaker Embeddings, we still need 16kHz
        if sr != 16000:
            import librosa
            audio_array_16k = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        else:
            audio_array_16k = audio_array

        waveform = torch.from_numpy(audio_array_16k)
        with torch.no_grad():
            input_wav = waveform.unsqueeze(0).to(device)
            emb = spk_model.encode_batch(input_wav)
            emb = torch.nn.functional.normalize(emb, dim=2).squeeze().cpu()
            spk_embeds.append(emb.numpy())

    return {"labels": mel_list, "speaker_embeddings": spk_embeds}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess_and_save_wavlm(use_perfected=False, num_samples=None):
    """
    Hybrid preprocessing pipeline.

    SOURCE modality  (English):
        WavLM frozen backbone → continuous hidden states (Seq_Len, 768)
        Saved under key: 'input_values'

    TARGET modality  (German):
        SpeechT5Processor feature extractor → 80-bin log-mel spectrogram (T, 80)
        Saved under key: 'labels'

    Both modalities are merged into a single Arrow dataset with columns:
        ['input_values', 'labels', 'speaker_embeddings']
    """
    suffix = "_perfected" if use_perfected else ""
    print(f"--- Hybrid WavLM/SpeechT5 Preprocessing {SOURCE_LANG} -> {TARGET_LANG}{suffix} ---")
    print(f"  Source modality : WavLM hidden states ({WAVLM_MODEL_NAME})")
    print(f"  Target modality : 80-bin mel-spectrogram + x-vector")

    num_proc = 1
    print(f"Using {num_proc} CPU worker(s) (conservative for model-based encoding).")

    # ------------------------------------------------------------------
    # 1. Load Raw Data
    # ------------------------------------------------------------------
    if use_perfected:
        from datasets import load_from_disk
        print(f"Loading perfected datasets from disk...")
        en_path = os.path.join(dataset_loader.DATASETS_DIR, "speech_t5_perfected", "en")
        de_path = os.path.join(dataset_loader.DATASETS_DIR, "speech_t5_perfected", "de")
        
        source_ds = load_from_disk(en_path)
        target_ds = load_from_disk(de_path)
        
        if num_samples:
            source_ds = source_ds.select(range(min(num_samples, len(source_ds))))
            target_ds = target_ds.select(range(min(num_samples, len(target_ds))))
    else:
        print("Loading raw FLEURS datasets...")
        datasets = dataset_loader.load_data(
            lang=[SOURCE_LANG, TARGET_LANG],
            split="train",
            dataset=["fleurs"],
            num_samples=num_samples or 20000,
        )
        source_ds = datasets.get(SOURCE_LANG)
        target_ds = datasets.get(TARGET_LANG)

    if source_ds is None or target_ds is None:
        print(f"ERROR: Could not load one or both datasets ({SOURCE_LANG}, {TARGET_LANG}).")
        return

    # Align lengths
    min_len = min(len(source_ds), len(target_ds))
    source_ds = source_ds.select(range(min_len))
    target_ds = target_ds.select(range(min_len))
    print(f"Initial aligned pairs: {min_len}")

    # ------------------------------------------------------------------
    # 2. SOURCE: Encode English audio → WavLM hidden states (input_values)
    # ------------------------------------------------------------------
    print("Encoding SOURCE (EN) with frozen WavLM → hidden states...")
    source_ds = source_ds.map(
        process_source_wavlm,
        batched=True,
        batch_size=64,
        num_proc=num_proc,
        desc="WavLM Encoding Source Audio",
        remove_columns=[c for c in source_ds.column_names if c != "id"], # Keep ID for verification
    )

    # ------------------------------------------------------------------
    # 3. TARGET: Encode German audio → 80-bin mel-spectrogram + X-Vector
    # ------------------------------------------------------------------
    print("Encoding TARGET (DE) → 80-bin mel-spectrogram + X-Vector...")
    target_ds = target_ds.map(
        process_target_mel,
        batched=True,
        batch_size=16,
        num_proc=num_proc,
        desc="Target Mel + X-Vector Encoding",
        remove_columns=[c for c in target_ds.column_names if c != "id"],
    )

    # ------------------------------------------------------------------
    # 4. Merge into a single paired dataset
    # ------------------------------------------------------------------
    print("Merging source and target features...")
    # Keep ID from source, remove from target
    target_ds = target_ds.remove_columns(["id"])
    
    from datasets import concatenate_datasets
    paired_ds = concatenate_datasets([source_ds, target_ds], axis=1)

    # ------------------------------------------------------------------
    # 5. Save to Disk
    # ------------------------------------------------------------------
    version = "v8_perfected" if use_perfected else "v8"
    out_path = os.path.join(
        OUTPUT_DIR,
        f"processed_wavlm_{SOURCE_LANG}_{TARGET_LANG}_{version}",
    )
    print(f"Saving paired dataset to {out_path}...")
    paired_ds.save_to_disk(out_path)

    print(f"\nSUCCESS! Hybrid preprocessing complete.")
    print(f"  Samples      : {len(paired_ds)}")
    print(f"  Columns      : {paired_ds.column_names}")
    print(f"  Output path  : {out_path}")
    return out_path


def verify_preprocessing(dataset_path, num_verify=3):
    """
    Verifies preprocessing by reconstructing audio from the stored mel-spectrograms.
    Uses the pre-trained SpeechT5 Hifi-GAN vocoder for high-quality reconstruction.
    """
    from datasets import load_from_disk
    from transformers import SpeechT5HifiGan
    import soundfile as sf
    import torch
    
    print(f"\n--- Verifying Preprocessing with Hifi-GAN ({num_verify} samples) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        print(f"Loading SpeechT5 Hifi-GAN vocoder on {device}...")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        vocoder.eval()
    except Exception as e:
        print(f"Error loading Hifi-GAN: {e}. Falling back to Griffin-Lim (lower quality)...")
        return _verify_preprocessing_griffin_lim(dataset_path, num_verify)

    ds = load_from_disk(dataset_path)
    indices = range(min(num_verify, len(ds)))
    
    os.makedirs("verification_audio", exist_ok=True)
    
    for i in indices:
        sample = ds[i]
        sample_id = sample.get("id", i)
        
        # 1. Prepare Target (DE) mel-spectrogram
        mel_array = np.array(sample["labels"], dtype=np.float32)
        mel = torch.from_numpy(mel_array).to(device) # (T, 80)
        
        # Vocoder expects (Batch, Seq_Len, 80)
        with torch.no_grad():
            waveform = vocoder(mel.unsqueeze(0))
            waveform = waveform.squeeze().cpu().numpy()
        
        out_file = f"verification_audio/sample_{sample_id}_de_hifigan.wav"
        sf.write(out_file, waveform, 16000)
        print(f"  Saved high-quality reconstructed target: {out_file}")

    print("\nHigh-quality verification audio saved in 'verification_audio/'.")
    print("Listen to these samples to confirm the features are ready for training.")


def _verify_preprocessing_griffin_lim(dataset_path, num_verify=3):
    """Fallback Griffin-Lim reconstruction."""
    from datasets import load_from_disk
    import librosa
    import soundfile as sf
    
    ds = load_from_disk(dataset_path)
    indices = range(min(num_verify, len(ds)))
    
    for i in indices:
        sample = ds[i]
        sample_id = sample.get("id", i)
        mel_array = np.array(sample["labels"], dtype=np.float32)
        mel = torch.from_numpy(mel_array).T # (80, T)
        log_mel = (mel * 2.0) + (-5.0)
        mel_unlogged = torch.exp(log_mel)
        
        audio_recon = librosa.feature.inverse.mel_to_audio(
            mel_unlogged.numpy(),
            sr=16000,
            n_fft=1024,
            hop_length=256,
            win_length=1024
        )
        
        out_file = f"verification_audio/sample_{sample_id}_de_griffin.wav"
        sf.write(out_file, audio_recon, 16000)
        print(f"  Saved fallback reconstructed target: {out_file}")


if __name__ == "__main__":
    path = preprocess_and_save_wavlm(use_perfected=False, num_samples=10000)
    if path:
        verify_preprocessing(path, num_verify=3)
