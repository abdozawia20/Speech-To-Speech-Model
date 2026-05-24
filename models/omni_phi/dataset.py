import json
import os
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import AutoProcessor

ANSWER_SUFFIX = "<|end|><|endoftext|>"
IGNORE_INDEX  = -100
INSTRUCTION   = "Translate this to spoken german:"

class OmniPhiDataset(Dataset):
    """
    Loads preprocessed (source_audio, target_tokens) pairs.
    Formats them into the exact input structure expected by
    Phi-4-multimodal-instruct for causal language model training.

    Performance: On first run, processor feature extraction (WavLM mel
    features) is computed once per sample and cached to .pt files next to
    the JSONL.  Subsequent epochs/runs skip all CPU feature extraction and
    just torch.load() the pre-computed tensors — making the DataLoader
    essentially I/O-bound rather than CPU-bound.
    """

    def __init__(self, jsonl_path: str, processor: AutoProcessor, training: bool = True):
        self.jsonl_path = jsonl_path
        self.processor  = processor
        self.training   = training
        self.offsets    = []

        # Cache directory lives alongside the JSONL file
        split = "train" if training else "eval"
        self.cache_dir = Path(jsonl_path).parent / f".cache_{split}"
        self.cache_dir.mkdir(exist_ok=True)

        # Pre-compute the static prompt text once (avoids repeated tokenizer calls)
        user_message = {
            "role": "user",
            "content": f"<|audio_1|>\n{INSTRUCTION}",
        }
        self._prompt_text = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        self._answer_ids = self.processor.tokenizer(
            ANSWER_SUFFIX, return_tensors="pt"
        ).input_ids  # [1, suffix_len] — computed once

        print(f"[OmniPhiDataset] Scanning byte offsets from {jsonl_path} ...")
        with open(jsonl_path, "rb") as f:
            offset = 0
            for line in f:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        print(f"[OmniPhiDataset] Scanned {len(self.offsets)} records.")

        # Pre-warm cache on construction (single-threaded, runs once)
        self._build_cache()

    def _build_cache(self):
        """Run processor once per sample and persist tensors to disk."""
        missing = [i for i in range(len(self.offsets))
                   if not (self.cache_dir / f"{i}.pt").exists()]
        if not missing:
            print(f"[OmniPhiDataset] Cache already complete ({len(self.offsets)} samples). Skipping.")
            return

        print(f"[OmniPhiDataset] Building feature cache for {len(missing)} samples "
              f"(this runs once — future epochs will load from disk)...")
        for count, idx in enumerate(missing, 1):
            self._compute_and_cache(idx)
            if count % 100 == 0 or count == len(missing):
                print(f"  Cached {count}/{len(missing)} samples...", flush=True)
        print(f"[OmniPhiDataset] Cache complete. Saved to {self.cache_dir}")

    def _compute_and_cache(self, idx: int):
        """Run AutoProcessor on one sample and save the result."""
        offset = self.offsets[idx]
        with open(self.jsonl_path, "rb") as f:
            f.seek(offset)
            line = f.readline()
        record = json.loads(line.decode("utf-8"))

        src_audio  = np.array(record["source_audio"], dtype=np.float32)
        target_ids = record["target_tokens"]

        inputs = self.processor(
            text=self._prompt_text,
            audios=[(src_audio, 16000)],
            return_tensors="pt",
        )

        target_tensor   = torch.tensor(target_ids, dtype=torch.long).unsqueeze(0)
        full_answer_ids = torch.cat([target_tensor, self._answer_ids], dim=1)

        torch.save({
            "input_ids":          inputs.input_ids,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes":  inputs.audio_embed_sizes,
            "full_answer_ids":    full_answer_ids,
        }, self.cache_dir / f"{idx}.pt")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        # Load from cache — fast torch.load() instead of processor feature extraction
        cached = torch.load(self.cache_dir / f"{idx}.pt", weights_only=True)

        input_ids_prompt = cached["input_ids"]          # [1, prompt_len]
        input_audio_embeds = cached["input_audio_embeds"]
        audio_embed_sizes  = cached["audio_embed_sizes"]
        full_answer_ids    = cached["full_answer_ids"]  # [1, N + suffix_len]

        if self.training:
            input_ids = torch.cat([input_ids_prompt, full_answer_ids], dim=1)
            labels    = torch.full_like(input_ids, IGNORE_INDEX)
            labels[:, -full_answer_ids.shape[1]:] = full_answer_ids
        else:
            input_ids = input_ids_prompt
            labels    = full_answer_ids

        return {
            "input_ids":          input_ids,
            "labels":             labels,
            "input_audio_embeds": input_audio_embeds,
            "audio_embed_sizes":  audio_embed_sizes,
        }
