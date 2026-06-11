import sys
import os
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np
import torch

# Resolve paths relative to this script's directory for maximum robustness
SCRIPT_DIR = Path(__file__).resolve().parent

# Add models/omni_phi and the project root to sys.path to allow imports from any CWD
omni_phi_dir = str(SCRIPT_DIR)
project_root = os.path.abspath(os.path.join(omni_phi_dir, "..", ".."))
for path in [omni_phi_dir, project_root]:
    if path not in sys.path:
        sys.path.append(path)

from model import OmniPhiS2ST

BATCH_TOKEN_LIMIT = 200   # tokens per chunk @ 150 tok/s (75 frames/s × 2 codebooks) ≈ 1.3 s audio
MAX_BATCHES       = 10    # hard cap: 10 chunks × 1.3 s ≈ 13 s of output max


def translate_speech(
    source_wav_path: str,
    output_wav_path: str = "output.wav",
    model: OmniPhiS2ST = None,
    max_new_tokens: int = 200,
    lang_prefix: str = "de",
):
    """
    Load a fine-tuned OmniPhiS2ST checkpoint and translate an audio file.

    Args:
        source_wav_path:  Path to the source English WAV file.
        output_wav_path:  Path to write the translated WAV file.
        model:            Pre-loaded OmniPhiS2ST instance. If None, the
                          checkpoint for `lang_prefix` is loaded automatically.
        max_new_tokens:   Upper limit on generated audio tokens.
                          At 1.5 kbps (75 frames/s × 2 codebooks):
                          200 tok ≈ 1.3 s, 800 tok ≈ 5.3 s.
        lang_prefix:      Target language ISO prefix (e.g. 'de', 'fr', 'it').
                          Determines which checkpoint dir to load and which
                          prompt is used. Must match a key in lang_config.LANG_CONFIG.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load fine-tuned model if not provided
    if model is None:
        checkpoint_dir = SCRIPT_DIR / f"checkpoints_en_{lang_prefix}"
        if not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()):
            raise FileNotFoundError(
                f"[inference] Checkpoint directory '{checkpoint_dir}' is empty or missing.\n"
                f"            Train the model first: python train.py --lang_tgt {lang_prefix}"
            )
        print(f"[inference] Loading OmniPhiS2ST from {checkpoint_dir}...")
        model = OmniPhiS2ST(
            phi4_model_id=str(checkpoint_dir),
            device=device,
            lang_prefix=lang_prefix,
        )
    model.eval()

    # Load source audio
    if not os.path.exists(source_wav_path):
        raise FileNotFoundError(f"Source audio file not found at: {source_wav_path}")

    audio, sr = sf.read(source_wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono
    audio = audio.astype(np.float32)

    # Run end-to-end generation
    print(f"[inference] Translating {source_wav_path} (sampling rate: {sr} Hz) → {lang_prefix}...")
    output_audio = model.generate_speech(audio, source_sr=sr, max_new_tokens=max_new_tokens)

    # Save result at 24kHz (EnCodec's native output rate)
    sf.write(output_wav_path, output_audio, samplerate=24000)
    print(f"[inference] Done. Output saved to {output_wav_path}")
    return output_audio


def translate_speech_batched(
    model: OmniPhiS2ST,
    source_audio: np.ndarray,
    source_sr: int = 16000,
    max_batches: int = MAX_BATCHES,
):
    """
    Translates using repeated generate_speech calls capped at BATCH_TOKEN_LIMIT tokens each,
    concatenating chunk waveforms into the final output.

    NOTE: Each chunk is an independent forward pass on the same source audio.
    This is useful for very long audio where a single generate call would OOM.
    For sentence-length inputs (FLEURS), use model.generate_speech() directly
    with max_new_tokens=800 instead.
    """
    all_waveforms = []

    print(f"[inference] Starting batched inference (token limit: {BATCH_TOKEN_LIMIT}/chunk, max {max_batches} chunks)...")
    for batch_idx in range(max_batches):
        print(f"[inference] Generating chunk {batch_idx + 1}/{max_batches}...")
        # Generate next batch of tokens
        chunk_waveform = model.generate_speech(
            source_audio,
            source_sr=source_sr,
            max_new_tokens=BATCH_TOKEN_LIMIT,
        )
        all_waveforms.append(chunk_waveform)

        # Stop condition: model produced an EOS or very short output
        # less than 0.1s at 24kHz (2400 samples) indicates termination
        if len(chunk_waveform) < 2400:
            print(f"[inference] Termination condition met at chunk {batch_idx + 1} (short chunk = EOS).")
            break
    else:
        print(f"[inference] Reached max_batches limit ({max_batches}). Stopping.")

    if len(all_waveforms) == 0:
        return np.zeros(2400, dtype=np.float32)

    return np.concatenate(all_waveforms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run OmniPhiS2ST inference on a WAV file."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to the source English WAV file.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to write the translated WAV file. "
             "Defaults to translated_<lang_prefix>.wav in the current directory.",
    )
    parser.add_argument(
        "--lang_tgt", default="de",
        help="Target language ISO prefix, e.g. 'de', 'fr', 'it' (default: 'de').",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=800,
        help="Maximum number of audio tokens to generate (default: 800 ≈ 5.3 s).",
    )
    args = parser.parse_args()

    output_path = args.output or f"translated_{args.lang_tgt}.wav"
    translate_speech(
        source_wav_path=args.input,
        output_wav_path=output_path,
        lang_prefix=args.lang_tgt,
        max_new_tokens=args.max_new_tokens,
    )
