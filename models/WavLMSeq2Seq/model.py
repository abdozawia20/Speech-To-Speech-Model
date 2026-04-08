import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import WavLMModel, Wav2Vec2Processor
from tqdm import tqdm
import math
import signal
import sys

class WavLMSeq2SeqDataset(Dataset):
    def __init__(self, source_ds, target_ds):
        self.source_ds = source_ds
        self.target_ds = target_ds

    def __len__(self):
        return len(self.source_ds)

    def __getitem__(self, idx):
        src_item = self.source_ds[int(idx)]
        tgt_item = self.target_ds[int(idx)]

        def get_val(item):
            val = item['audio']
            if isinstance(val, dict) and 'array' in val:
                return val['array']
            return val

        # Handle raw audio sequences
        src_val = torch.tensor(get_val(src_item), dtype=torch.float32)
        tgt_val = torch.tensor(get_val(tgt_item), dtype=torch.float32)

        if src_val.ndim > 1: src_val = src_val.flatten()
        if tgt_val.ndim > 1: tgt_val = tgt_val.flatten()

        return {"input_values": src_val, "target_values": tgt_val}

def wavlm_collate_fn(batch):
    inputs = [item["input_values"] for item in batch]
    targets = [item["target_values"] for item in batch]

    # Pad sequences to match lengths in a batch
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

    # Attention masks (1 for valid, 0 for padding)
    input_mask = (inputs_padded != 0).long()
    target_mask = (targets_padded != 0).long()

    return inputs_padded, input_mask, targets_padded, target_mask


class WavLMTranslator(nn.Module):
    def __init__(self, encoder_name="microsoft/wavlm-base", d_model=768, num_decoder_layers=6):
        super(WavLMTranslator, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing WavLMTranslator on {self.device}")

        # Processor & Pretrained Encoder
        print(f"Loading WavLM backbone: {encoder_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(encoder_name)
        self.encoder = WavLMModel.from_pretrained(encoder_name)

        # Freeze encoder to save memory (Feature Extraction Mode)
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=8, 
            dim_feedforward=3072, 
            dropout=0.1, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output linear projection
        self.vector_out = nn.Linear(d_model, d_model)
        # Probability of stopping (end of speech)
        self.stop_token_out = nn.Linear(d_model, 1)

        self.to(self.device)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_audio, input_mask, target_audio, target_mask):
        """
        Training forward pass utilizing teacher forcing.
        Input audio and target audio are raw waveforms (padded).
        """
        # 1. Encode Source Audio -> WavLM Vectors
        with torch.no_grad():
            enc_out = self.encoder(input_values=input_audio, attention_mask=input_mask)
            memory = enc_out.last_hidden_state # (Batch, Src_Seq, 768)

        # 2. Encode Target Audio -> Target Vectors (Teacher Forcing Input)
        with torch.no_grad():
            tgt_out = self.encoder(input_values=target_audio, attention_mask=target_mask)
            target_vectors = tgt_out.last_hidden_state # (Batch, Tgt_Seq, 768)
        
        batch_size, tgt_seq_len, _ = target_vectors.size()

        # Target input for the decoder (shifted right by 1 frame intuitively or just masking future)
        # We use a standard causal mask to prevent peeking into the future
        tgt_causal_mask = self.generate_square_subsequent_mask(tgt_seq_len)

        # 3. Decode
        # memory_key_padding_mask requires boolean where True means "ignore" (i.e. padding = True)
        # However, Wav2Vec doesn't strictly provide frame-level masks easily without subsampling logic.
        # For simplicity in this vector-to-vector S2S, we'll run without padding masks or compute them if needed.
        
        decoded = self.decoder(
            tgt=target_vectors, 
            memory=memory, 
            tgt_mask=tgt_causal_mask
        )

        predicted_vectors = self.vector_out(decoded)
        predicted_stop = self.stop_token_out(decoded).squeeze(-1) # (Batch, Tgt_Seq)

        return predicted_vectors, predicted_stop, target_vectors

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "model.pth"))
        self.processor.save_pretrained(os.path.join(path, "processor"))
        print(f"Model saved to {path}")

    def load(self, path):
        loaded_state = torch.load(os.path.join(path, "model.pth"), map_location=self.device)
        self.load_state_dict(loaded_state)
        # Processor is assumed to be generic WavLM base
        self.eval()
        print(f"Model loaded from {path}")

    def interrupt_handler(self, sig, frame):
        """Handles manual interruption signal."""
        print("\nInterruption detected! Saving checkpoint...")
        self.save('interrupted_wavlm_checkpoint')
        sys.exit(0)

    def train_model(self, loader, epochs=10, learning_rate=1e-4):
        self.train()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=learning_rate)
        
        mse_loss_fn = nn.MSELoss()
        bce_loss_fn = nn.BCEWithLogitsLoss()

        signal.signal(signal.SIGINT, self.interrupt_handler)

        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                epoch_loss = 0.0
                pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")

                for input_audio, input_mask, target_audio, target_mask in pbar:
                    input_audio = input_audio.to(self.device).squeeze(1) if input_audio.dim() > 2 else input_audio.to(self.device)
                    input_mask = input_mask.to(self.device).squeeze(1) if input_mask.dim() > 2 else input_mask.to(self.device)
                    target_audio = target_audio.to(self.device).squeeze(1) if target_audio.dim() > 2 else target_audio.to(self.device)
                    target_mask = target_mask.to(self.device).squeeze(1) if target_mask.dim() > 2 else target_mask.to(self.device)

                    optimizer.zero_grad()
                    
                    try:
                        # Target audio must be normalized
                        input_inputs = self.processor(input_audio.cpu().numpy(), sampling_rate=16000, return_tensors='pt', padding=True)
                        target_inputs = self.processor(target_audio.cpu().numpy(), sampling_rate=16000, return_tensors='pt', padding=True)
                        
                        in_wav = input_inputs.input_values.to(self.device)
                        in_mask = input_inputs.attention_mask.to(self.device)
                        tgt_wav = target_inputs.input_values.to(self.device)
                        tgt_mask = target_inputs.attention_mask.to(self.device)

                        predicted_vectors, predicted_stop, target_vectors = self.forward(
                            in_wav, in_mask, tgt_wav, tgt_mask
                        )

                        # For teacher forcing predicting the next frame, 
                        # we should ideally shift the targets. 
                        # pred: [0, T-1] should match target: [1, T]
                        # Since we want to keep it simple, we compare identically length-aligned chunks
                        # but shift inside the loss
                        
                        pred_vec = predicted_vectors[:, :-1, :]
                        true_vec = target_vectors[:, 1:, :]
                        
                        loss_mse = mse_loss_fn(pred_vec, true_vec)
                        
                        # Stop token dummy loss (1 when it's the last frame)
                        stop_targets = torch.zeros_like(predicted_stop)
                        stop_targets[:, -1] = 1.0 # Only last frame is 1
                        
                        loss_stop = bce_loss_fn(predicted_stop, stop_targets)
                        
                        loss = loss_mse + loss_stop
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                    
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        continue
                
                print(f"Epoch {epoch+1} Completed. Avg Loss: {epoch_loss/len(loader):.4f}")
        
        except KeyboardInterrupt:
            print(f"\nTraining interrupted by user. Saving checkpoint at epoch {getattr(self, 'current_epoch', 0) + 1}...")
            self.save('interrupted_wavlm_checkpoint')
            return self

        print("Training finished.")
        return self
