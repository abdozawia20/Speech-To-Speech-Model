import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import librosa
from tqdm import tqdm

# Add project root to path so we can import from encoders and dataset_loader
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset_loader import load_data
from encoders import VQGANEncoder

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Target Tokenization & Preprocessing for Omni-Phi")
    parser.add_argument("--dataset", type=str, default="seamless_align", choices=["fleurs", "seamless_align"],
                        help="Dataset to load ('fleurs' or 'seamless_align')")
    parser.add_argument("--lang_src", type=str, default="en", help="Source language code (default: en)")
    parser.add_argument("--lang_tgt", type=str, default="de", help="Target language code (default: de)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to preprocess (default: train)")
    parser.add_argument("--num_samples", type=int, default=15000, help="Maximum number of samples to load")
    parser.add_argument("--bandwidth", type=float, default=1.5, help="EnCodec bandwidth in kbps (default: 1.5 for 2 codebooks)")
    parser.add_argument("--token_offset", type=int, default=100000, help="Vocabulary offset for target tokens (default: 100000)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for preprocessed JSONL files")
    return parser.parse_args()

def audio_to_tokens(audio_array: np.ndarray, vqgan: VQGANEncoder, bandwidth: float = 1.5, token_offset: int = 100_000, orig_sr: int = 16000) -> list[int]:
    """
    Converts a raw 16kHz numpy array to a flat list of offset EnCodec token IDs.

    Returns:
        List of ints, length = Frames * num_codebooks.
        Each int is in the range [token_offset, token_offset + 1023].
    """
    # 1. EnCodec expects 24kHz; resample target audio from orig_sr to 24000 Hz
    if orig_sr != 24000:
        audio_24k = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=24000)
    else:
        audio_24k = audio_array

    # 2. Extract inputs via VQGANEncoder's processor
    inputs = vqgan.processor(raw_audio=audio_24k, sampling_rate=24000, return_tensors="pt")
    inputs = {k: v.to(vqgan.device) for k, v in inputs.items()}

    # 3. Get codebooks
    with torch.no_grad():
        encoded = vqgan.model.encode(**inputs, bandwidth=bandwidth)

    # audio_codes shape: [Batch=1, Channels=1, Codebooks, Frames]
    codes = encoded.audio_codes[0]  # Shape: [Channels=1, Codebooks, Frames]
    if codes.ndim == 3:
        codes = codes[0]  # Shape: [Codebooks, Frames]

    codes = codes.cpu().numpy()
    num_codebooks, num_frames = codes.shape

    # 4. Interleave: [CB0_F0, CB1_F0, CB0_F1, CB1_F1, ...] and add offset
    flat_tokens = []
    for frame_idx in range(num_frames):
        for cb_idx in range(num_codebooks):
            # EnCodec token is [0, 1023] -> map to [token_offset, token_offset + 1023]
            flat_tokens.append(int(codes[cb_idx, frame_idx]) + token_offset)

    return flat_tokens

def main():
    args = parse_args()
    
    # Establish output directory
    if args.output_dir is None:
        out_dir = Path(os.path.dirname(__file__)) / "data" / "preprocessed"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Omni-Phi Preprocessing Phase 1 ===")
    print(f"Dataset      : {args.dataset}")
    print(f"Languages    : {args.lang_src} (source) -> {args.lang_tgt} (target)")
    print(f"Split        : {args.split}")
    print(f"Max Samples  : {args.num_samples}")
    print(f"Bandwidth    : {args.bandwidth} kbps")
    print(f"Token Offset : {args.token_offset}")
    print(f"Output Dir   : {out_dir}")
    print(f"======================================")

    # ── 1. Load Aligned Audio Pairs ──────────────────────────────────────────
    print(f"Loading aligned pairs using dataset_loader...")
    datasets = load_data(
        dataset=[args.dataset],
        lang=[args.lang_src, args.lang_tgt],
        split=args.split,
        num_samples=args.num_samples,
    )
    
    src_ds = datasets.get(args.lang_src)
    tgt_ds = datasets.get(args.lang_tgt)
    
    if src_ds is None or tgt_ds is None or len(src_ds) == 0:
        print(f"ERROR: No aligned data loaded for languages {args.lang_src} and {args.lang_tgt}!")
        return

    # Guarantee dataset lengths are perfectly matched and sorted by ID
    assert len(src_ds) == len(tgt_ds), f"Mismatch in dataset sizes: {len(src_ds)} vs {len(tgt_ds)}"
    num_pairs = len(src_ds)
    print(f"Successfully loaded {num_pairs} aligned audio pairs.")

    # ── 2. Configure VQGANEncoder ───────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Instantiating VQGANEncoder on {device}...")
    vqgan = VQGANEncoder(model_name="facebook/encodec_24khz")
    vqgan.model.eval().to(device)
    for p in vqgan.model.parameters():
        p.requires_grad = False
    print("VQGANEncoder frozen.")

    # ── 3. Processing loop & Streaming to Disk ──────────────────────────────
    output_file = out_dir / f"{args.split}.jsonl"
    print(f"Tokenizing, preprocessing, and saving records directly to {output_file}...")
    
    saved_count = 0
    with open(output_file, "w") as f:
        for i in tqdm(range(num_pairs), desc="Preprocessing"):
            src_entry = src_ds[i]
            tgt_entry = tgt_ds[i]

            # Verify alignment
            assert src_entry["id"] == tgt_entry["id"], f"ID mismatch at index {i}: {src_entry['id']} != {tgt_entry['id']}"

            # Source Audio: 16kHz raw float32 array
            src_audio = np.array(src_entry["audio"]["array"], dtype=np.float32)
            src_sr = src_entry["audio"]["sampling_rate"]

            # If source isn't 16000Hz, resample it (though dataset_loader should ensure 16000Hz)
            if src_sr != 16000:
                src_audio = librosa.resample(src_audio, orig_sr=src_sr, target_sr=16000)

            # Target Audio: Convert to interleaved offset EnCodec tokens
            tgt_audio = np.array(tgt_entry["audio"]["array"], dtype=np.float32)
            tgt_sr = tgt_entry["audio"]["sampling_rate"]

            # Run tokenization
            try:
                target_tokens = audio_to_tokens(
                    audio_array=tgt_audio,
                    vqgan=vqgan,
                    bandwidth=args.bandwidth,
                    token_offset=args.token_offset,
                    orig_sr=tgt_sr
                )
                
                record = {
                    "id": src_entry["id"],
                    "source_audio": src_audio.tolist(),   # serializable list
                    "target_tokens": target_tokens,
                }
                f.write(json.dumps(record) + "\n")
                saved_count += 1
            except Exception as e:
                print(f"\nWarning: Skipped sample {src_entry['id']} due to error: {e}")
                continue
            
    print(f"SUCCESS: Preprocessing Phase 1 complete. Saved {saved_count} records. File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
