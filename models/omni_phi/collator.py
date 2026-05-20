import torch
from transformers import BatchFeature

IGNORE_INDEX = -100

def omni_phi_collate_fn(batch):
    """
    Pads a list of dataset items into a single batched BatchFeature.
    Labels for the prompt region are -100 (ignored by Cross-Entropy loss).
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

    # Concatenate audio embeddings along the batch dimension
    input_audio_embeds = torch.cat(input_audio_embeds_list, dim=0)
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
