import sys
import os
from pathlib import Path
import soundfile as sf
import numpy as np
import torch

# Resolve paths relative to this script's directory for maximum robustness
SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"

# Add models/omni_phi and the project root to sys.path to allow imports from any CWD
omni_phi_dir = str(SCRIPT_DIR)
project_root = os.path.abspath(os.path.join(omni_phi_dir, "..", ".."))
for path in [omni_phi_dir, project_root]:
    if path not in sys.path:
        sys.path.append(path)

from model import OmniPhiS2ST

BATCH_TOKEN_LIMIT = 200   # tokens per generation call (~5 seconds of audio at 1.5 kbps)

def translate_speech(source_wav_path: str, output_wav_path: str = "output.wav", model: OmniPhiS2ST = None):
    """
    Load a fine-tuned OmniPhiS2ST checkpoint and translate an audio file.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load fine-tuned model if not provided
    if model is None:
        print(f"[inference] Loading OmniPhiS2ST model from {CHECKPOINT_DIR}...")
        model = OmniPhiS2ST(phi4_model_id=str(CHECKPOINT_DIR), device=device)
    model.eval()

    # Load source audio
    if not os.path.exists(source_wav_path):
        raise FileNotFoundError(f"Source audio file not found at: {source_wav_path}")
        
    audio, sr = sf.read(source_wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono
    audio = audio.astype(np.float32)

    # Run end-to-end generation
    print(f"[inference] Translating {source_wav_path} (sampling rate: {sr} Hz)...")
    output_audio = model.generate_speech(audio, source_sr=sr)

    # Save result at 24kHz (EnCodec's native output rate)
    sf.write(output_wav_path, output_audio, samplerate=24000)
    print(f"[inference] Done. Output saved to {output_wav_path}")
    return output_audio

def translate_speech_batched(model: OmniPhiS2ST, source_audio: np.ndarray, source_sr: int = 16000):
    """
    Translates in batches of BATCH_TOKEN_LIMIT tokens.
    Queues each batch and concatenates the final waveforms.
    """
    all_waveforms = []
    remaining_audio = source_audio

    print(f"[inference] Starting batched inference (token limit: {BATCH_TOKEN_LIMIT} per call)...")
    while True:
        # Generate next batch of tokens
        chunk_waveform = model.generate_speech(
            remaining_audio,
            source_sr=source_sr,
            max_new_tokens=BATCH_TOKEN_LIMIT,
        )
        all_waveforms.append(chunk_waveform)

        # Stop condition: model produced an EOS or very short output
        # less than 0.1s at 24kHz (2400 samples) indicates termination
        if len(chunk_waveform) < 2400:
            print("[inference] Termination condition met (short chunk generated).")
            break

    if len(all_waveforms) == 0:
        return np.zeros(2400, dtype=np.float32)

    return np.concatenate(all_waveforms)

if __name__ == "__main__":
    # Example usage:
    # translate_speech("test_english.wav", "translated_german.wav")
    pass
