import json
import numpy as np
import torch
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
    """

    def __init__(self, jsonl_path: str, processor: AutoProcessor, training: bool = True):
        self.jsonl_path = jsonl_path
        self.processor  = processor
        self.training   = training
        self.offsets    = []

        print(f"[OmniPhiDataset] Scanning byte offsets from {jsonl_path} to enable low-RAM lazy loading...")
        with open(jsonl_path, "rb") as f:
            offset = 0
            for line in f:
                if line.strip():
                    self.offsets.append(offset)
                offset += len(line)
        print(f"[OmniPhiDataset] Scanned {len(self.offsets)} records.")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        offset = self.offsets[idx]
        with open(self.jsonl_path, "rb") as f:
            f.seek(offset)
            line = f.readline()
        record = json.loads(line.decode("utf-8"))

        # 1. Source audio: raw 16kHz float32 array
        src_audio = np.array(record["source_audio"], dtype=np.float32)

        # 2. Target token IDs (already offset by +100,000 in preprocess_omni.py)
        target_ids = record["target_tokens"]

        # 3. Build the user prompt (matches official Phi-4 chat template)
        user_message = {
            "role": "user",
            "content": f"<|audio_1|>\n{INSTRUCTION}",
        }
        prompt_text = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )

        # 4. Run the AutoProcessor: encodes the prompt text + source audio together.
        #    This produces input_ids, input_audio_embeds, and audio_embed_sizes.
        inputs = self.processor(
            text=prompt_text,
            audios=[(src_audio, 16000)],
            return_tensors="pt",
        )

        # 5. Build the answer token sequence and append the EOS suffix
        answer_text = ANSWER_SUFFIX
        answer_ids  = self.processor.tokenizer(
            answer_text, return_tensors="pt"
        ).input_ids  # shape: [1, suffix_len]

        # Prepend target audio tokens to the answer
        target_tensor = torch.tensor(target_ids, dtype=torch.long).unsqueeze(0)  # [1, N]
        full_answer_ids = torch.cat([target_tensor, answer_ids], dim=1)          # [1, N + suffix_len]

        if self.training:
            # Concatenate prompt tokens + answer tokens into one sequence
            input_ids = torch.cat([inputs.input_ids, full_answer_ids], dim=1)

            # Build labels: mask the prompt portion with IGNORE_INDEX (-100)
            # so the model only computes loss on the target audio tokens + suffix
            labels = torch.full_like(input_ids, IGNORE_INDEX)
            labels[:, -full_answer_ids.shape[1]:] = full_answer_ids
        else:
            # Eval mode: input is prompt only; labels are the ground-truth tokens
            input_ids = inputs.input_ids
            labels    = full_answer_ids

        return {
            "input_ids":          input_ids,
            "labels":             labels,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes":  inputs.audio_embed_sizes,
        }
