import os
import sys
import torch
import torchaudio
import numpy as np

# Fix for speechbrain dependency on torchaudio.list_audio_backends
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

import dataset_loader

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
dataset_loader.NUM_PROC = 1
SOURCE_LANG = "en"
TARGET_LANG = "de"
MAX_DURATION_SECONDS = 8.0
OUTPUT_DIR = dataset_loader.DATASETS_DIR

# WavLM model for SOURCE encoding
WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"

# SpeechT5 model name for reference
SPEECHT5_MODEL_NAME = "microsoft/speecht5_vc"

# ---------------------------------------------------------------------------
# Global state for worker processes
# ---------------------------------------------------------------------------
_wavlm_proc_cache = None
_wavlm_model_cache = None
_spk_model_cache = None
_MEL_TRANSFORM = None


def _get_wavlm_model():
    global _wavlm_proc_cache, _wavlm_model_cache
    if _wavlm_model_cache is None:
        from transformers import Wav2Vec2FeatureExtractor, WavLMModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading WavLM ({WAVLM_MODEL_NAME}) on {device}...")
        _wavlm_proc_cache = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL_NAME)
        _wavlm_model_cache = WavLMModel.from_pretrained(WAVLM_MODEL_NAME).to(device)
        for p in _wavlm_model_cache.parameters():
            p.requires_grad_(False)
        _wavlm_model_cache.eval()
    return _wavlm_proc_cache, _wavlm_model_cache


def _get_spk_model():
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


def _get_mel_transform():
    global _MEL_TRANSFORM
    if _MEL_TRANSFORM is None:
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


def process_source_wavlm(batch):
    proc, model = _get_wavlm_model()
    device = next(model.parameters()).device
    encoded_list = []
    with torch.no_grad():
        for audio_item in batch["audio"]:
            audio_array = np.array(audio_item["array"], dtype=np.float32)
            sr = audio_item["sampling_rate"]
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
            inputs = proc(audio_array, sampling_rate=16000, return_tensors="pt", padding=False)
            input_values = inputs.input_values.to(device)
            out = model(input_values)
            hidden = out.last_hidden_state.squeeze(0)
            encoded_list.append(hidden.cpu().numpy())
    return {"input_values": encoded_list}


def process_target_vocoder(batch):
    """Extracts normalized labels, waveforms, and per-sample speaker embeddings."""
    import torch
    mel_transform = _get_mel_transform()
    spk_model = _get_spk_model()
    device = spk_model.device

    mel_list = []
    waveform_list = []
    spk_embeds = []

    for audio_item in batch["audio"]:
        audio_array = np.array(audio_item["array"], dtype=np.float32)
        sr = audio_item["sampling_rate"]
        if sr != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

        waveform = torch.from_numpy(audio_array)

        # 1. Mel + Normalization
        mel = mel_transform(waveform)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        log_mel = (log_mel - (-5.0)) / 2.0
        log_mel = log_mel.T
        mel_list.append(log_mel.numpy())

        # 2. Waveform (original for vocoder ground-truth)
        waveform_list.append(audio_array)

        # 3. Per-sample Speaker Embedding
        with torch.no_grad():
            input_wav = waveform.unsqueeze(0).to(device)
            emb = spk_model.encode_batch(input_wav)
            emb = torch.nn.functional.normalize(emb, dim=2).squeeze().cpu()
            spk_embeds.append(emb.numpy())

    return {"labels": mel_list, "target_waveform": waveform_list, "speaker_embeddings": spk_embeds}


def preprocess_vocoder_data(num_samples=None, trained_model_path=None):
    """
    Prepares a paired dataset for HiFi-GAN vocoder fine-tuning.
    """
    print(f"--- Vocoder Preprocessing {SOURCE_LANG} -> {TARGET_LANG} (v4 Compatible) ---")
    
    # 1. Load Raw Data
    datasets = dataset_loader.load_data(
        lang=[SOURCE_LANG, TARGET_LANG],
        split="train",
        dataset=["fleurs"],
        num_samples=num_samples if num_samples else 15000,
    )
    source_ds = datasets[SOURCE_LANG]
    target_ds = datasets[TARGET_LANG]

    min_len = min(len(source_ds), len(target_ds))
    source_ds = source_ds.select(range(min_len))
    target_ds = target_ds.select(range(min_len))

    # 4. SOURCE: Encode WavLM
    source_ds = source_ds.map(process_source_wavlm, batched=True, batch_size=32, desc="WavLM Encoding")

    # 5. TARGET: Encode Mel + Waveform + Speaker
    target_ds = target_ds.map(process_target_vocoder, batched=True, batch_size=16, desc="Target Vocoder Encoding")

    # 6. Merge
    target_ds = target_ds.remove_columns([c for c in target_ds.column_names if c in source_ds.column_names])
    from datasets import concatenate_datasets
    paired_ds = concatenate_datasets([source_ds, target_ds], axis=1)

    # 7. Save intermediate
    out_path = os.path.join(OUTPUT_DIR, f"processed_vocoder_{SOURCE_LANG}_{TARGET_LANG}_v4")
    paired_ds.save_to_disk(out_path)

    # 8. Phase 2: Cache predicted mel
    if trained_model_path and os.path.exists(trained_model_path):
        final_out_path = out_path + "_final"
        generate_and_cache_predicted_mel(out_path, final_out_path, model_path=trained_model_path)
    else:
        print("WARNING: Skipping Phase 2 (Predicted Mel) because trained_model_path was not provided or not found.")


def generate_and_cache_predicted_mel(paired_ds_path, output_ds_path, model_path, device="cuda"):
    """
    Runs the fine-tuned model (with v4 adapter) to generate 'predicted_mel'.
    """
    import torch
    from datasets import load_from_disk
    from transformers.models.speecht5.modeling_speecht5 import shift_spectrograms_right
    from models.SpeechT5WavLM.model import SpeechT5WavLM

    print(f"\n--- Starting Phase 2: Caching Predicted Mel (v4 Logic) ---")
    ds = load_from_disk(paired_ds_path)

    custom_model = SpeechT5WavLM()
    custom_model.load(model_path)
    custom_model.eval()

    predicted_mels = []

    with torch.no_grad():
        for i in range(0, len(ds), 8):
            batch = ds[i:i+8]
            input_values = [torch.tensor(np.array(x).reshape(-1, 768), dtype=torch.float32) for x in batch["input_values"]]
            labels = [torch.tensor(np.array(x).reshape(-1, 80), dtype=torch.float32) for x in batch["labels"]]
            spk_embeds = [torch.tensor(np.array(x), dtype=torch.float32) for x in batch["speaker_embeddings"]]

            input_padded = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True).to(device)
            labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100.0).to(device)
            spk_padded = torch.stack(spk_embeds).to(device)
            attn_mask = (input_padded.abs().sum(dim=-1) != 0).long()

            # Hybrid v4 forward path: Adapter + Encoder + Decoder
            encoder_out = custom_model._encode_wavlm_states(input_padded, attn_mask)
            
            decoder_input_values, decoder_attention_mask = shift_spectrograms_right(
                labels_padded, custom_model.model.config.reduction_factor, None
            )

            outputs = custom_model.model.speecht5(
                encoder_outputs=encoder_out,
                attention_mask=attn_mask,
                decoder_input_values=decoder_input_values,
                decoder_attention_mask=decoder_attention_mask,
                speaker_embeddings=spk_padded,
                use_cache=False,
                return_dict=True,
            )

            _, outputs_after_postnet, _ = custom_model.model.speech_decoder_postnet(outputs.last_hidden_state)

            for j in range(len(batch["labels"])):
                T_tgt = labels[j].shape[0]
                pred_mel = outputs_after_postnet[j, :T_tgt, :].cpu().numpy()
                predicted_mels.append(pred_mel)

            if i % 80 == 0:
                print(f"Processed {i}/{len(ds)} samples...")

    from datasets import Dataset, concatenate_datasets
    def mel_generator():
        for mel in predicted_mels:
            yield {"predicted_mel": mel}
    mel_ds = Dataset.from_generator(mel_generator)
    ds = concatenate_datasets([ds, mel_ds], axis=1)
    ds.save_to_disk(output_ds_path)
    print(f"Caching complete! Final dataset: {output_ds_path}")


if __name__ == "__main__":
    # Point this to your latest checkpoint (where loss is ~3.1)
    # E.g. "checkpoint_epoch_20" or "speecht5_wavlm_interrupted"
    if len(sys.argv) < 2:
        print("Usage: python preprocess_vocoder_data.py <path_to_trained_model_dir>")
        sys.exit(1)
        
    model_checkpoint = sys.argv[1]
    preprocess_vocoder_data(num_samples=None, trained_model_path=model_checkpoint)
