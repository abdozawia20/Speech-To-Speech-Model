import torch
from transformers import BatchFeature

IGNORE_INDEX = -100

def omni_phi_collate_fn(batch):
    """
    Pads a list of dataset items into a single batched BatchFeature.
    Labels for the prompt region are -100 (ignored by Cross-Entropy loss).

    Perf notes:
    - input_audio_embeds are cast to bfloat16 here (processor outputs float32).
      This halves memory bandwidth for both CPU padding and the subsequent
      CPU→GPU transfer (which the model's forward() does with non_blocking=True).
    - pin_memory should be True in the DataLoader for fastest CPU→GPU transfer.
    """
    input_ids_list          = [item["input_ids"][0] for item in batch]
    labels_list             = [item["labels"][0]    for item in batch]
    input_audio_embeds_list = [item["input_audio_embeds"] for item in batch]
    audio_embed_sizes_list  = [item["audio_embed_sizes"]  for item in batch]

    def pad_sequence(sequences, padding_value, padding_side="right"):
        max_len    = max(s.size(0) for s in sequences)
        batch_size = len(sequences)
        out = sequences[0].new_full((batch_size, max_len), padding_value)
        for i, seq in enumerate(sequences):
            if padding_side == "left":
                out[i, -seq.size(0):] = seq
            else:
                out[i, :seq.size(0)] = seq
        return out

    input_ids = pad_sequence(input_ids_list, padding_value=0,            padding_side="left")
    labels    = pad_sequence(labels_list,    padding_value=IGNORE_INDEX,  padding_side="left")

    attention_mask = (input_ids != 0).long()

    # Cast audio embeds to bfloat16 before padding — the processor outputs float32
    # but Phi-4 uses bfloat16, so we convert here to halve the memory footprint
    # of the padded tensor and the subsequent CPU→GPU transfer.
    input_audio_embeds_list = [e.to(dtype=torch.bfloat16) for e in input_audio_embeds_list]

    # Pad and stack audio embeddings to support varying audio lengths in the batch
    max_audio_len = max(e.size(1) for e in input_audio_embeds_list)
    embed_dim = input_audio_embeds_list[0].size(2)
    padded_embeds = []
    for e in input_audio_embeds_list:
        curr_len = e.size(1)
        if curr_len < max_audio_len:
            padding = e.new_zeros((1, max_audio_len - curr_len, embed_dim))
            padded_e = torch.cat([e, padding], dim=1)
        else:
            padded_e = e
        padded_embeds.append(padded_e)
    
    input_audio_embeds = torch.cat(padded_embeds, dim=0)
    audio_embed_sizes  = torch.cat(audio_embed_sizes_list)

    # Audio attention mask (True = real embedding, False = padding)
    audio_attention_mask = None
    if len(input_audio_embeds_list) > 1:
        max_audio_len = max(e.size(1) for e in input_audio_embeds_list)
        mask_list = [
            e.new_full((e.size(1),), True, dtype=torch.bool)
            for e in input_audio_embeds_list
        ]
        audio_attention_mask = pad_sequence(mask_list, padding_value=False, padding_side="right")

    return BatchFeature({
        "input_ids":           input_ids,
        "labels":              labels,
        "attention_mask":      attention_mask,
        "input_audio_embeds":  input_audio_embeds,
        "audio_embed_sizes":   audio_embed_sizes,
        "audio_attention_mask": audio_attention_mask,
        "input_mode":          2,  # speech mode
    })
