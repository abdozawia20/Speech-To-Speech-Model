import os
import sys
import torch
import numpy as np
from faster_whisper import WhisperModel

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import dataset_loader

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
dataset_loader.NUM_PROC = 1 # Recommended for model mapping steps
SOURCE_LANG = "en"
TARGET_LANG = "de"
MAX_DURATION_SECONDS = 8.0
OUTPUT_DIR = dataset_loader.DATASETS_DIR

# WavLM model for SOURCE encoding (English → continuous hidden states)
WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"

# SpeechT5 model name for reference
SPEECHT5_MODEL_NAME = "microsoft/speecht5_vc"

# ---------------------------------------------------------------------------
# Global state for worker processes
# ---------------------------------------------------------------------------
_wavlm_proc_cache = None       # Wav2Vec2FeatureExtractor for WavLM
_wavlm_model_cache = None      # Frozen WavLMModel backbone
_MEL_TRANSFORM = None


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
# TARGET modality: Mel Spectrogram and Waveform (German audio)
# ---------------------------------------------------------------------------

def _get_mel_transform():
    """
    Lazy-load and cache the torchaudio MelSpectrogram transform using
    SpeechT5's exact parameters.
    """
    global _MEL_TRANSFORM
    if _MEL_TRANSFORM is None:
        import torchaudio
        _MEL_TRANSFORM = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=80.0,
            f_max=7600.0,
            n_mels=80,
            mel_scale="htk",
        )
    return _MEL_TRANSFORM


def process_target_vocoder(batch):
    """
    Extracts:
      - 'labels': 80-bin log-mel spectrogram (T, 80)
      - 'target_waveform': Raw 16kHz audio waveform
    """
    import torch
    mel_transform = _get_mel_transform()

    mel_list = []
    waveform_list = []
    for audio_item in batch["audio"]:
        audio_array = np.array(audio_item["array"], dtype=np.float32)
        sr = audio_item["sampling_rate"]

        # Resample to 16 kHz if needed
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

        waveform = torch.from_numpy(audio_array)  # (N,)

        # mel_transform expects (N,) or (1, N) and returns (80, T)
        mel = mel_transform(waveform)              # (80, T)

        # Apply log compression
        log_mel = torch.log(torch.clamp(mel, min=1e-5))  # (80, T)

        # Transpose to (T, 80)
        log_mel = log_mel.T                        # (T, 80)

        mel_list.append(log_mel.numpy())
        waveform_list.append(audio_array)

    return {"labels": mel_list, "target_waveform": waveform_list}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess_vocoder_data(num_samples=None):
    """
    Prepares a paired dataset for HiFi-GAN vocoder fine-tuning.
    
    Columns:
      - input_values: WavLM hidden states (from English audio)
      - labels: 80-bin log-mel spectrograms (from German audio)
      - target_waveform: 16kHz raw waveform (from German audio)
    """
    print(f"--- Vocoder Preprocessing {SOURCE_LANG} -> {TARGET_LANG} ---")
    print(f"  Source modality : WavLM hidden states ({WAVLM_MODEL_NAME})")
    print(f"  Target modality : 80-bin mel-spectrogram + 16kHz Waveform")
    print(f"  Max Duration    : {MAX_DURATION_SECONDS}s")

    num_proc = 2
    
    # 1. Load Raw Data
    print("Loading raw datasets...")
    datasets = dataset_loader.load_data(
        lang=[SOURCE_LANG, TARGET_LANG],
        split="train",
        dataset=["seamless_align"],
        num_samples=num_samples if num_samples else 15000,
    )

    source_ds = datasets[SOURCE_LANG]
    target_ds = datasets[TARGET_LANG]

    # Align lengths
    min_len = min(len(source_ds), len(target_ds))
    source_ds = source_ds.select(range(min_len))
    target_ds = target_ds.select(range(min_len))
    print(f"Initial aligned pairs: {min_len}")

    # 4. SOURCE: Encode English audio → WavLM hidden states (input_values)
    print("Encoding SOURCE (EN) with frozen WavLM...")
    source_ds = source_ds.map(
        process_source_wavlm,
        batched=True,
        batch_size=64,
        num_proc=num_proc,
        desc="WavLM Encoding Source Audio",
        remove_columns=["audio"],
    )

    # 5. TARGET: Encode German audio → Mel + Waveform
    print("Encoding TARGET (DE) → 80-bin mel-spectrogram + Waveform...")
    target_ds = target_ds.map(
        process_target_vocoder,
        batched=True,
        batch_size=16,
        num_proc=num_proc,
        desc="Vocoder Encoding Target Audio",
        remove_columns=["audio"],
    )

    # 6. Merge
    print("Merging source hidden states and target vocoder data...")
    target_ds = target_ds.remove_columns([c for c in target_ds.column_names if c in source_ds.column_names])
    
    from datasets import concatenate_datasets
    paired_ds = concatenate_datasets([source_ds, target_ds], axis=1)

    # 7. Save
    out_path = os.path.join(
        OUTPUT_DIR,
        f"processed_vocoder_{SOURCE_LANG}_{TARGET_LANG}_v1",
    )
    print(f"Saving paired dataset to {out_path}...")
    paired_ds.save_to_disk(out_path)

    print(f"\nSUCCESS! Vocoder preprocessing complete.")
    print(f"  Samples      : {len(paired_ds)}")
    print(f"  Columns      : {paired_ds.column_names}")
    print(f"  Output path  : {out_path}")

    # --- Cache predicted mel spectrograms from frozen SpeechT5 ---
    # Path to where your fine-tuned WavLM+SpeechT5 weights are saved
    generate_and_cache_predicted_mel(
        out_path,
        out_path
    )


# ---------------------------------------------------------------------------
# Phase 2: Cache predicted mel spectrograms from frozen SpeechT5
# ---------------------------------------------------------------------------

def generate_and_cache_predicted_mel(paired_ds_path, output_ds_path, model_path, device="cuda"):
    """
    Loads the preprocessed dataset, runs the frozen SpeechT5+WavLM
    architecture sequentially to get the 'predicted_mel', and saves
    it to the output dataset path.
    """
    import torch
    from datasets import load_from_disk
    from transformers.models.speecht5.modeling_speecht5 import shift_spectrograms_right
    import numpy as np

    print(f"\n--- Starting Phase 2: Caching Predicted Mel Spectrograms ---")
    ds = load_from_disk(paired_ds_path)

    # 1. Load the customized model and embedding
    print(f"Loading custom model from {model_path}...")
    from model import SpeechT5WavLM  # Assumes your model class is importable
    custom_model = SpeechT5WavLM()
    custom_model.load(model_path)
    custom_model.model.eval()

    speaker_embedding = custom_model.target_embeddings.to(device)

    predicted_mels = []

    # 2. Iterate sequentially (Batch size 8 prevents OOM)
    print("Running forward pass to generate predicted spectrograms...")
    with torch.no_grad():
        for i in range(0, len(ds), 8):
            batch = ds[i:i+8]

            # Reconstruct tensors with explicit float32 dtype
            input_values = [torch.tensor(np.array(x).reshape(-1, 768), dtype=torch.float32) for x in batch["input_values"]]
            labels = [torch.tensor(np.array(x).reshape(-1, 80), dtype=torch.float32) for x in batch["labels"]]

            # Pad sequences
            input_padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True).to(device)
            labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100.0).to(device)
            attn_mask = (input_padded.abs().sum(dim=-1) != 0).long()

            spk_embeds = speaker_embedding.unsqueeze(0).expand(input_padded.shape[0], -1)

            # WavLM Injection
            hidden_states = torch.nn.functional.layer_norm(input_padded, [input_padded.shape[-1]])
            prenet = custom_model.model.speecht5.encoder.prenet
            hidden_states = hidden_states + prenet.pos_conv_embed(hidden_states)

            padding_mask = attn_mask.ne(1).long()
            pos_sin = prenet.pos_sin_embed(padding_mask) if hasattr(prenet, "pos_sin_embed") else prenet.pos_sinusoidal_embed(padding_mask)
            hidden_states = hidden_states + pos_sin

            encoder_obj = custom_model.model.speecht5.encoder
            if hasattr(encoder_obj, "wrapped_encoder"):
                encoder_obj = encoder_obj.wrapped_encoder
            encoder_out = encoder_obj(hidden_states=hidden_states, attention_mask=attn_mask, return_dict=True)

            decoder_input_values, decoder_attention_mask = shift_spectrograms_right(
                labels_padded, custom_model.model.config.reduction_factor, None
            )

            outputs = custom_model.model.speecht5(
                encoder_outputs=encoder_out,
                attention_mask=attn_mask,
                decoder_input_values=decoder_input_values,
                decoder_attention_mask=decoder_attention_mask,
                speaker_embeddings=spk_embeds,
                use_cache=False,
                return_dict=True,
            )

            _, outputs_after_postnet, _ = custom_model.model.speech_decoder_postnet(outputs.last_hidden_state)

            # Extract unpadded mels and move to CPU numpy for saving
            for j in range(len(batch["labels"])):
                T_tgt = labels[j].shape[0]
                pred_mel = outputs_after_postnet[j, :T_tgt, :].cpu().numpy()
                predicted_mels.append(pred_mel)

            if i % 80 == 0:
                print(f"Processed {i}/{len(ds)} samples...")

    # 3. Save new column to dataset
    print(f"Appending 'predicted_mel' to dataset and saving to {output_ds_path}...")
    ds = ds.add_column("predicted_mel", predicted_mels)
    ds.save_to_disk(output_ds_path)
    print("Caching complete!")


if __name__ == "__main__":
    # For testing, use a small number of samples if requested or default to 15000
    # test_mode = "--test" in sys.argv
    # num_samples = 10 if test_mode else 15000
    # preprocess_vocoder_data(num_samples=num_samples)
    
    # For upgrading the local dataset directly
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    input_ds = os.path.join(project_root, "datasets", "processed_vocoder_en_de_v1")
    output_ds = os.path.join(project_root, "datasets", "processed_vocoder_en_de_v2")
    model_path = os.path.join(os.path.dirname(__file__), "speecht5_wavlm_en_de_v3")
    
    print(f"Upgrading dataset:\n  From: {input_ds}\n  To:   {output_ds}")
    generate_and_cache_predicted_mel(input_ds, output_ds, model_path)