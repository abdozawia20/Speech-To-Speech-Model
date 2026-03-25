import torch
import gc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
from ..dataset_loader import *
from datasets import load_from_disk
from ..encoders import *
import signal
import sys
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import math

class AlignedSpeechDataset(Dataset):
    def __init__(self, en_ds, tr_ds, fixed_size=(256, 256)):
        self.en_ds = en_ds
        self.tr_ds = tr_ds
        self.fixed_size = fixed_size
        
    def __len__(self):
        return len(self.en_ds)

    def __getitem__(self, idx):
        # Retrieve wav2vec embeddings (numpy arrays) from 'audio' field
        src_w2v = self.en_ds[idx]['audio']
        tgt_w2v = self.tr_ds[idx]['audio']
        
        # Helper to process raw wav2vec to resized tensor
        def process(w2v):
            if isinstance(w2v, list):
                w2v = np.array(w2v)
            
            t_w2v = torch.tensor(w2v, dtype=torch.float32).unsqueeze(0)
            
            # Interpolate expects (Batch, Channels, H, W) -> (1, 1, H, W)
            # We resize to fixed_size
            t_resized = F.interpolate(t_w2v.unsqueeze(0), size=self.fixed_size, mode='bilinear', align_corners=False).squeeze(0)
            return t_resized

        src = process(src_w2v)
        tgt = process(tgt_w2v)
        
        return src, tgt

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class UNetWave2Vec(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetWave2Vec, self).__init__()

        # Encoder 1: Input -> 64 (Stride 1)
        self.enc1 = ResBlock(in_channels, 64, stride=1)

        # Encoder 2: 64 -> 128 (Stride 2)
        self.enc2 = ResBlock(64, 128, stride=2)

        # Encoder 3: 128 -> 256 (Stride 2)
        self.enc3 = ResBlock(128, 256, stride=2)

        # Bottleneck: 256 -> 512 (Stride 2)
        self.bottleneck = ResBlock(256, 512, stride=2)

        # Decoder 3
        # Upsample 512 -> 256
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_up3 = nn.BatchNorm2d(256)
        # Fuse with Enc3 (256): 256 + 256 = 512 -> 256
        self.dec3 = ResBlock(512, 256, stride=1)

        # Decoder 2
        # Upsample 256 -> 128
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_up2 = nn.BatchNorm2d(128)
        # Fuse with Enc2 (128): 128 + 128 = 256 -> 128
        self.dec2 = ResBlock(256, 128, stride=1)

        # Decoder 1
        # Upsample 128 -> 64
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_up1 = nn.BatchNorm2d(64)
        # Fuse with Enc1 (64): 64 + 64 = 128 -> 64
        self.dec1 = ResBlock(128, 64, stride=1)

        # Final Output Layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def save(self, path, optimizer=None, epoch=None):
        """Saves weights, optimizer state, and epoch."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'epoch': epoch
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load(self, path, optimizer=None):
        """Loads weights and optimizer state."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
            
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and checkpoint['optimizer_state_dict']:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
        return checkpoint

    def interrupt_handler(self, sig, frame):
        """Handles manual interruption signal."""
        print("\nInterruption detected! Saving checkpoint...")
        self.save('interrupted_checkpoint.pth', 
                  optimizer=getattr(self, 'optimizer', None), 
                  epoch=getattr(self, 'current_epoch', None))
        sys.exit(0)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder with Skip Connections
        
        # Level 3
        d3 = self.up3(b)
        d3 = self.bn_up3(d3)
        d3 = nn.ReLU(inplace=True)(d3)
        # Concatenate: (Batch, 256, H/4, W/4) + (Batch, 256, H/4, W/4)
        d3 = torch.cat((d3, e3), dim=1) 
        d3 = self.dec3(d3)

        # Level 2
        d2 = self.up2(d3)
        d2 = self.bn_up2(d2)
        d2 = nn.ReLU(inplace=True)(d2)
        # Concatenate: (Batch, 128, H/2, W/2) + (Batch, 128, H/2, W/2)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        # Level 1
        d1 = self.up1(d2)
        d1 = self.bn_up1(d1)
        d1 = nn.ReLU(inplace=True)(d1)
        # Concatenate: (Batch, 64, H, W) + (Batch, 64, H, W)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        # Output
        out = self.final_conv(d1)
        return out

    def train_model(self, epochs=5, batch_size=4, learning_rate=1e-3, num_workers=0, data_dir='./processed_data'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        self.to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.optimizer = optimizer
        
        # LR Scheduler: Decays every 15/100 epochs
        step_size = max(1, int(epochs * 0.15))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

        print(f"Starting training... (LR Step Size: {step_size})")
        self.train()

        # Register interrupt handler
        signal.signal(signal.SIGINT, self.interrupt_handler)

        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                print(f"--- Starting Epoch {epoch+1}/{epochs} ---")
                epoch_loss = 0.0
                num_chunks = 0

                en_dir = os.path.join(data_dir, 'en')
                if not os.path.exists(en_dir):
                    print(f"Error: Data directory not found at {en_dir}")
                    return self
                    
                chunk_names = [d for d in os.listdir(en_dir) if d.startswith('chunk_')]
                chunk_names.sort(key=lambda x: int(x.split('_')[1]))
                
                for chunk_name in chunk_names:
                    print(f"Processing {chunk_name}...")
                    
                    en_path = os.path.join(en_dir, chunk_name)
                    try:
                        en_ds = load_from_disk(en_path)
                    except Exception as e:
                        print(f"Failed to load {en_path}: {e}")
                        continue

                    tr_path = os.path.join(data_dir, 'tr', chunk_name)
                    if not os.path.exists(tr_path):
                        print(f"Corresponding Turkish chunk not found at {tr_path}. Skipping.")
                        continue
                        
                    try:
                        tr_ds = load_from_disk(tr_path)
                    except Exception as e:
                        print(f"Failed to load {tr_path}: {e}")
                        continue

                    def get_unique_indices(ds, key='id'):
                        seen = set()
                        indices = []
                        for i, val in enumerate(ds[key]):
                            if val not in seen:
                                seen.add(val)
                                indices.append(i)
                        return indices

                    if len(en_ds) > 0:
                        en_ds = en_ds.select(get_unique_indices(en_ds))
                    if len(tr_ds) > 0:
                        tr_ds = tr_ds.select(get_unique_indices(tr_ds))

                    en_ids = set(en_ds['id'])
                    tr_ids = set(tr_ds['id'])
                    common_ids = en_ids.intersection(tr_ids)
                    
                    if len(common_ids) == 0:
                        print("No common samples found in this chunk. Skipping.")
                        continue

                    en_ds = en_ds.filter(lambda x: x['id'] in common_ids)
                    tr_ds = tr_ds.filter(lambda x: x['id'] in common_ids)
                    
                    en_ds = en_ds.sort("id")
                    tr_ds = tr_ds.sort("id")

                    try:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                        dataset = AlignedSpeechDataset(en_ds, tr_ds)
                        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                                num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)

                        chunk_loss = 0.0
                        for i, (inputs, targets) in enumerate(dataloader):
                            inputs, targets = inputs.to(device), targets.to(device)

                            optimizer.zero_grad()
                            outputs = self(inputs)
                            loss = criterion(outputs, targets)
                            loss.backward()
                            optimizer.step()

                            chunk_loss += loss.item()
                        
                        if len(dataloader) > 0:
                            avg_chunk_loss = chunk_loss / len(dataloader)
                            epoch_loss += avg_chunk_loss
                            num_chunks += 1
                            print(f"  Chunk {chunk_name} Loss: {avg_chunk_loss:.4f}")
                    except Exception as e:
                        print(f"Error during standard training on chunk {chunk_name}: {e}")
                        continue
            
                if num_chunks > 0:
                    avg_epoch_loss = epoch_loss / num_chunks
                    print(f"Epoch [{epoch+1}/{epochs}] Completed. Average Loss: {avg_epoch_loss:.4f}")
                else:
                    print(f"Epoch [{epoch+1}/{epochs}] Completed. No data processed.")
                
                scheduler.step()
                print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving checkpoint...")
            self.save(os.path.join(data_dir, f'interrupted_model_epoch_{epoch+1}.pth'), optimizer, epoch)
            return self

        print("Training finished.")
        return self
