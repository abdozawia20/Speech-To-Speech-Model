"""
test_encoders_diagnostic.py
============================
Three-way validation script for both Wav2VecSpeechT5Encoder and WavLMSpeechT5Encoder.

For each encoder it produces:
  1. Reconstruction from SAVED hidden states (preprocess_SpeechT5_wav2vec output)
  2. Reconstruction from ON-THE-FLY encoded states
  3. Native generate_speech baseline (Wav2VecSpeechT5Encoder only)

All outputs are written to validation_results/ and summarised at the end.
"""

import os
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
from datasets import load_from_disk
from dataset_loader import load_data
from encoders import Wav2VecSpeechT5Encoder, WavLMSpeechT5Encoder
from transformers import SpeechT5HifiGan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_file(path: str, label: str) -> None:
    """Print file status with a quality hint based on size."""
    if os.path.exists(path):
        size = os.path.getsize(path)
        quality = "OK" if size > 50_000 else "WARN – suspiciously small"
        print(f"  [{quality}] {label}: {size:,} bytes")
    else:
        print(f"  [MISSING] {label}")


def _load_speaker_embedding(encoder, audio: np.ndarray) -> torch.Tensor:
    print("  Extracting real speaker embedding from source audio...")
    emb = encoder.get_speaker_embedding(audio)
    print(f"  Speaker embedding shape: {emb.shape}, norm: {emb.norm().item():.4f}")
    return emb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_dir = "validation_results"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("[Step 1] Loading preprocessed dataset...")
    preprocessed_ds = load_from_disk("preprocessed_speecht5_wav2vec")

    print("[Step 1] Loading original raw audio for comparison...")
    original_datasets = load_data(dataset=["fleurs"], lang=["en", "de"], num_samples=100)
    en_orig_ds = original_datasets["en"]
    de_orig_ds = original_datasets["de"]

    # Select first sample pair
    prep_sample = preprocessed_ds[0]
    sample_id = prep_sample["id"]
    en_orig = en_orig_ds[0]
    de_orig = de_orig_ds[0]

    assert str(en_orig["id"]) == str(sample_id), (
        f"ID mismatch: {en_orig['id']} != {sample_id}"
    )
    assert str(de_orig["id"]) == str(sample_id), (
        f"ID mismatch: {de_orig['id']} != {sample_id}"
    )
    print(f"  Sample ID: {sample_id}")

    orig_en_audio = en_orig["audio"]["array"]
    orig_de_audio = de_orig["audio"]["array"]

    # ------------------------------------------------------------------
    # 2. Save original audio files for reference
    # ------------------------------------------------------------------
    print("[Step 2] Writing original audio files...")
    sf.write(os.path.join(output_dir, "original_en.wav"), orig_en_audio, 16000)
    sf.write(os.path.join(output_dir, "original_de.wav"), orig_de_audio, 16000)

    # ------------------------------------------------------------------
    # 3. Wav2VecSpeechT5Encoder validation
    # ------------------------------------------------------------------
    print("\n[Step 3] --- Wav2VecSpeechT5Encoder ---")
    w2v_encoder = Wav2VecSpeechT5Encoder(load_decoder=True)
    spk_emb = _load_speaker_embedding(w2v_encoder, orig_en_audio)

    # 3a. Reconstruct from SAVED hidden states
    print("  [3a] Reconstructing from SAVED hidden states (stop token disabled)...")
    en_hs_saved = np.array(prep_sample["en_hidden_states"], dtype=np.float32)
    if en_hs_saved.ndim == 1:
        en_hs_saved = en_hs_saved.reshape(-1, 768)
    en_hs_saved_t = torch.from_numpy(en_hs_saved)
    # disable_stop_token=True: for long inputs speecht5_vc's cross-attention only
    # covers ~50% of encoder frames before the stop token fires (empirically verified).
    # Running to maxlenratio produces audio that covers the full source duration.
    recon_saved = w2v_encoder.decode(en_hs_saved_t, spk_emb, disable_stop_token=True)
    out_path = os.path.join(output_dir, "w2v_reconstructed_from_saved.wav")
    sf.write(out_path, recon_saved, 16000)

    # 3b. Reconstruct ON-THE-FLY
    print("  [3b] Encoding & reconstructing ON-THE-FLY (stop token disabled)...")
    en_hs_new = w2v_encoder.encode(orig_en_audio)
    recon_new = w2v_encoder.decode(en_hs_new, spk_emb, disable_stop_token=True)
    out_path_new = os.path.join(output_dir, "w2v_reconstructed_on_the_fly.wav")
    sf.write(out_path_new, recon_new, 16000)

    # 3c. Native generate_speech (ground-truth baseline)
    print("  [3c] Native generate_speech baseline...")
    with torch.no_grad():
        inputs = w2v_encoder.processor(
            audio=orig_en_audio, sampling_rate=16000, return_tensors="pt"
        ).to(w2v_encoder.device)
        recon_native = w2v_encoder.model.generate_speech(
            inputs.input_values,
            spk_emb.to(w2v_encoder.device),
            vocoder=w2v_encoder.vocoder,
            # Explicitly use matching defaults so the comparison is fair
            threshold=0.5,
            minlenratio=0.0,
            maxlenratio=20.0,
        )
        recon_native = recon_native.cpu().numpy()
    sf.write(os.path.join(output_dir, "w2v_reconstructed_native.wav"), recon_native, 16000)

    # ------------------------------------------------------------------
    # 4. WavLMSpeechT5Encoder validation
    # ------------------------------------------------------------------
    print("\n[Step 4] --- WavLMSpeechT5Encoder ---")
    wavlm_encoder = WavLMSpeechT5Encoder(load_decoder=True)
    spk_emb_wavlm = _load_speaker_embedding(wavlm_encoder, orig_en_audio)

    # 4a. Reconstruct ON-THE-FLY (WavLM has no pre-saved hidden states)
    print("  [4a] Encoding & reconstructing ON-THE-FLY (stop token disabled)...")
    wavlm_hs = wavlm_encoder.encode(orig_en_audio)
    print(f"  WavLM hidden states shape: {wavlm_hs.shape}")
    recon_wavlm = wavlm_encoder.decode(wavlm_hs, spk_emb_wavlm, disable_stop_token=True)
    sf.write(
        os.path.join(output_dir, "wavlm_reconstructed_on_the_fly.wav"),
        recon_wavlm,
        16000,
    )

    # ------------------------------------------------------------------
    # 5. Reconstruct German Mel Spectrogram via HiFi-GAN (target side)
    # ------------------------------------------------------------------
    print("\n[Step 5] Reconstructing German audio from saved Mel Spectrogram...")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    vocoder.eval()

    de_mel = np.array(prep_sample["de_mel_spectrogram"], dtype=np.float32)
    if de_mel.ndim == 1:
        de_mel = de_mel.reshape(-1, 80)
    # Ensure (T, 80) — transpose if stored as (80, T)
    if de_mel.shape[0] == 80 and de_mel.shape[1] != 80:
        de_mel = de_mel.T
    de_mel_tensor = torch.from_numpy(de_mel).unsqueeze(0)  # (1, T, 80)
    print(f"  de_mel shape: {de_mel_tensor.shape}")

    with torch.no_grad():
        recon_de = vocoder(de_mel_tensor).cpu().numpy().squeeze()
    if recon_de.ndim == 0:
        recon_de = np.array([recon_de])
    sf.write(os.path.join(output_dir, "de_reconstructed_from_mel.wav"), recon_de, 16000)

    # ------------------------------------------------------------------
    # 6. Spectrogram visualisation
    # ------------------------------------------------------------------
    print("\n[Step 6] Generating spectrogram visualisations...")

    S_orig = librosa.feature.melspectrogram(
        y=orig_de_audio, sr=16000, n_mels=80, fmax=8000
    )
    S_dB_orig = librosa.power_to_db(S_orig, ref=np.max)

    # Ensure (80, T) for specshow
    S_prep = de_mel.T if de_mel.shape[1] == 80 else de_mel

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    img0 = librosa.display.specshow(
        S_dB_orig, x_axis="time", y_axis="mel", sr=16000, fmax=8000, ax=axes[0]
    )
    fig.colorbar(img0, ax=axes[0], format="%+2.0f dB")
    axes[0].set_title("Original German Mel Spectrogram (librosa)")

    img1 = librosa.display.specshow(
        S_prep, x_axis="time", y_axis="mel", sr=16000, fmax=8000, ax=axes[1]
    )
    fig.colorbar(img1, ax=axes[1])
    axes[1].set_title("Preprocessed German Mel Spectrogram (SpeechT5Processor)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "spectrogram_comparison.png"), dpi=150)
    plt.close()

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("VERIFICATION SUMMARY")
    print("=" * 72)

    wav_checks = [
        ("original_en.wav",                   "Original English"),
        ("original_de.wav",                   "Original German"),
        ("w2v_reconstructed_from_saved.wav",  "W2V → SpeechT5 (saved states)"),
        ("w2v_reconstructed_on_the_fly.wav",  "W2V → SpeechT5 (on-the-fly)"),
        ("w2v_reconstructed_native.wav",      "Native generate_speech baseline"),
        ("wavlm_reconstructed_on_the_fly.wav","WavLM → SpeechT5 (on-the-fly)"),
        ("de_reconstructed_from_mel.wav",     "German Mel → HiFi-GAN"),
    ]
    other_checks = [
        ("spectrogram_comparison.png",        "Spectrogram comparison"),
    ]

    # Find reference duration from original_en for ratio comparison
    ref_dur = None
    all_ok = True

    print(f"  {'Label':<40s} {'Duration':>9s}  {'Ratio':>7s}  {'Size':>12s}")
    print(f"  {'-'*40} {'-'*9}  {'-'*7}  {'-'*12}")

    for filename, label in wav_checks:
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            data, sr = sf.read(path)
            dur = len(data) / sr
            size = os.path.getsize(path)
            if ref_dur is None:
                ref_dur = dur
                ratio_str = " (ref)"
            else:
                ratio = dur / ref_dur
                ratio_str = f"{ratio:6.1%}"
                if filename.endswith("en.wav") or "reconstruct" in filename:
                    # Exclude the native baseline — it uses the model's built-in stop
                    # token and is intentionally shorter (comparison reference only).
                    if "native" not in filename and ratio < 0.90:
                        all_ok = False
                        ratio_str += " ⚠"
            print(f"  {label:<40s} {dur:8.2f}s  {ratio_str:>7s}  {size:>11,} B")
        else:
            print(f"  {label:<40s} {'MISSING':>9s}")
            all_ok = False

    for filename, label in other_checks:
        path = os.path.join(output_dir, filename)
        status = f"{os.path.getsize(path):,} bytes" if os.path.exists(path) else "MISSING"
        print(f"  {label:<40s} {'':9s}  {'':7s}  {status:>12s}")
        if not os.path.exists(path):
            all_ok = False

    print("=" * 72)
    if all_ok:
        print("All checks passed. Reconstructed durations are within 80% of original.")
    else:
        print("WARNING: Some reconstructions are too short — check stop-token threshold.")
    print(f"Output directory: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    main()
