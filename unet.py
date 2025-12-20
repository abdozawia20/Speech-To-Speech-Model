import torch
import gc
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import dataset_loader
from datasets import load_from_disk
import encoders
import signal
import sys
import os


class AlignedSpeechDataset(Dataset):
    def __init__(self, en_ds, tr_ds, fixed_size=(256, 256)):
        self.en_ds = en_ds
        self.tr_ds = tr_ds
        self.fixed_size = fixed_size
        
    def __len__(self):
        return len(self.en_ds)
        
    def __getitem__(self, idx):
        # Retrieve encoded spectrograms (numpy arrays) from 'audio' field
        src_spec = self.en_ds[idx]['audio']
        tgt_spec = self.tr_ds[idx]['audio']
        
        # Helper to process raw spectrogram to resized tensor
        def process(spec):
            # Convert to tensor: (1, F, T)
            # Ensure proper type
            if isinstance(spec, list):
                spec = np.array(spec)
            t_spec = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)
            
            # Interpolate expects (Batch, Channels, H, W) -> (1, 1, F, T)
            # We resize to fixed_size
            t_resized = F.interpolate(t_spec.unsqueeze(0), size=self.fixed_size, mode='bilinear', align_corners=False).squeeze(0)
            return t_resized

        src = process(src_spec)
        tgt = process(tgt_spec)
        
        return src, tgt

def train_model(epochs=5, batch_size=4, learning_rate=1e-3, num_workers=0, data_dir='./processed_data'):
    
    # Initialize model, optimizer, criterion ONCE
    model = UNetSpectrogramTranslator()
    
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # LR Scheduler: Decays every 15/100 epochs
    step_size = max(1, int(epochs * 0.15))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    print(f"Starting training... (LR Step Size: {step_size})")
    model.train()

    for epoch in range(epochs):
        print(f"--- Starting Epoch {epoch+1}/{epochs} ---")
        epoch_loss = 0.0
        num_chunks = 0

        # Discover chunks in the English directory
        en_dir = os.path.join(data_dir, 'en')
        if not os.path.exists(en_dir):
            print(f"Error: Data directory not found at {en_dir}")
            return model
            
        # List all chunk directories and sort NUMERICALLY
        chunk_names = [d for d in os.listdir(en_dir) if d.startswith('chunk_')]
        # Sort by start_idx: chunk_{start}_{end}
        chunk_names.sort(key=lambda x: int(x.split('_')[1]))
        
        for chunk_name in chunk_names:
            print(f"Processing {chunk_name}...")
            
            # Load English dataset
            en_path = os.path.join(en_dir, chunk_name)
            try:
                en_ds = load_from_disk(en_path)
            except Exception as e:
                print(f"Failed to load {en_path}: {e}")
                continue

            # Attempt to load corresponding Turkish dataset
            # Assuming same chunk naming convention or we look for overlapping ID range
            # Here we assume the user ran preprocess for both languages with same chunking
            tr_path = os.path.join(data_dir, 'tr', chunk_name)
            if not os.path.exists(tr_path):
                print(f"Corresponding Turkish chunk not found at {tr_path}. Skipping.")
                continue
                
            try:
                tr_ds = load_from_disk(tr_path)
            except Exception as e:
                print(f"Failed to load {tr_path}: {e}")
                continue

            # Helper to deduplicate by ID
            def get_unique_indices(ds, key='id'):
                seen = set()
                indices = []
                # ds[key] returns values list efficiently
                for i, val in enumerate(ds[key]):
                    if val not in seen:
                        seen.add(val)
                        indices.append(i)
                return indices

            # print("Deduplicating datasets...")
            if len(en_ds) > 0:
                en_ds = en_ds.select(get_unique_indices(en_ds))
            if len(tr_ds) > 0:
                tr_ds = tr_ds.select(get_unique_indices(tr_ds))

            # print("Aligning datasets...")
            # 1. Filter to common IDs
            en_ids = set(en_ds['id'])
            tr_ids = set(tr_ds['id'])
            common_ids = en_ids.intersection(tr_ids)
            
            # print(f"Found {len(common_ids)} common samples in this chunk.")
            
            if len(common_ids) == 0:
                print("No common samples found in this chunk. Skipping.")
                continue

            en_ds = en_ds.filter(lambda x: x['id'] in common_ids)
            tr_ds = tr_ds.filter(lambda x: x['id'] in common_ids)
            
            # 2. Sort by ID to ensure alignment
            en_ds = en_ds.sort("id")
            tr_ds = tr_ds.sort("id")

            # Data is ALREADY encoded as spectrograms from preprocess_data.py
            # So we skip the encoding step!

            # print("Creating DataLoader...")
            try:
                dataset = AlignedSpeechDataset(en_ds, tr_ds)
                # pinned_memory=True helps with CUDA transfer
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                                        num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)

                chunk_loss = 0.0
                for i, (inputs, targets) in enumerate(dataloader):
                    inputs, targets = inputs.to(device), targets.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    chunk_loss += loss.item()
                
                if len(dataloader) > 0:
                    avg_chunk_loss = chunk_loss / len(dataloader)
                    # print(f"Chunk Loss: {avg_chunk_loss:.4f}")
                    epoch_loss += avg_chunk_loss
                    num_chunks += 1
            except Exception as e:
                print(f"Error during training on chunk {chunk_name}: {e}")
                continue
        
        if num_chunks > 0:
            avg_epoch_loss = epoch_loss / num_chunks
            print(f"Epoch [{epoch+1}/{epochs}] Completed. Average Loss: {avg_epoch_loss:.4f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] Completed. No data processed.")
        
        # Step the scheduler
        scheduler.step()
        print(f"Current Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    print("Training finished.")
    return model

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

class UNetSpectrogramTranslator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetSpectrogramTranslator, self).__init__()

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

    def resume_training(self, checkpoint_path, dataloader, epochs=5, learning_rate=1e-3):
        """Resumes training from a checkpoint or starts fresh."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        start_epoch = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = self.load(checkpoint_path, self.optimizer)
            start_epoch = checkpoint.get('epoch', -1) + 1
            print(f"Resuming training from epoch {start_epoch}...")
        else:
            print("Starting new training session...")
            
        # Register interrupt handler
        signal.signal(signal.SIGINT, self.interrupt_handler)
        
        self.train()
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch # Store for interrupt handler
            running_loss = 0.0
            
            for i, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            avg_loss = running_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
        print("Training finished.")

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
        # Concatenate: (Batch, 256, F/4, T/4) + (Batch, 256, F/4, T/4)
        d3 = torch.cat((d3, e3), dim=1) 
        d3 = self.dec3(d3)

        # Level 2
        d2 = self.up2(d3)
        d2 = self.bn_up2(d2)
        d2 = nn.ReLU(inplace=True)(d2)
        # Concatenate: (Batch, 128, F/2, T/2) + (Batch, 128, F/2, T/2)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)

        # Level 1
        d1 = self.up1(d2)
        d1 = self.bn_up1(d1)
        d1 = nn.ReLU(inplace=True)(d1)
        # Concatenate: (Batch, 64, F, T) + (Batch, 64, F, T)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)

        # Output
        out = self.final_conv(d1)
        return out
