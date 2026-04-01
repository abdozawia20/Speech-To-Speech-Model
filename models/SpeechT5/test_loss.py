import torch
import sys
import os

from model import SpeechT5, SpeechT5Dataset, speecht5_collate_fn
from torch.utils.data import DataLoader

model = SpeechT5(encoder_type='wav2vec')
model.load("speecht5_interrupted")

# Mock dataset for quick testing
class DummyDataset:
    def __init__(self, size=2, len=153):
        self.size = size
        self.data = [{'audio': {'array': torch.randn(200, 768).numpy()}}] * size
        self.labels = [{'audio': {'array': torch.randn(len, 80).numpy()}}] * size
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.data[idx]

src_ds = DummyDataset(2, 100)
tgt_ds = DummyDataset(2, 457)

train_dataset = SpeechT5Dataset(
    src_ds,
    tgt_ds,
    model.processor,
    torch.randn(512),
    is_preprocessed=True,
    encoder_type='wav2vec'
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    collate_fn=speecht5_collate_fn
)

for input_values, attention_mask, labels, speaker_embeddings in train_loader:
    print("Labels size:", labels.shape)
    
    from transformers.models.speecht5.modeling_speecht5 import shift_spectrograms_right, SpeechT5SpectrogramLoss
    decoder_input_values, decoder_attention_mask = shift_spectrograms_right(
        labels, model.model.config.reduction_factor, None
    )

    encoder_out = model._encode_wav2vec_states(
        input_values.to(model.device), attention_mask.to(model.device)
    )
    outputs = model.model.speecht5(
        encoder_outputs=encoder_out,
        attention_mask=attention_mask.to(model.device),
        decoder_input_values=decoder_input_values.to(model.device),
        decoder_attention_mask=decoder_attention_mask,
        speaker_embeddings=speaker_embeddings.to(model.device),
        use_cache=False,
        output_attentions=True,
    )

    outputs_before_postnet, outputs_after_postnet, logits = model.model.speech_decoder_postnet(outputs[0])
    
    criterion = SpeechT5SpectrogramLoss(model.model.config)
    loss = criterion(
        attention_mask.to(model.device),
        outputs_before_postnet,
        outputs_after_postnet,
        logits,
        labels.to(model.device),
        cross_attentions=outputs.cross_attentions
    )
    print("Loss:", loss.item())
    break

