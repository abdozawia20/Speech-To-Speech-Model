import torch
import gc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import signal
import sys
import os

# Add the project root to sys.path so direct imports (dataset_loader, encoders)
# work regardless of how/where this module is loaded — mirrors the style used
# by preprocess_speecht5.py.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dataset_loader import *
from datasets import load_from_disk
from encoders import *

from torch.utils.data import Dataset, DataLoader, TensorDataset

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import math

class AlignedSpeechDataset(Dataset):
    """
    Dataset that loads pre-encoded 80-bin log-mel spectrograms from disk
    (produced by preprocess_unet.py) and resizes the time axis to a fixed
    width so every batch is uniformly shaped.

    Each sample's 'audio' field is a (80, T) float32 numpy array in dB scale.
    After processing the tensor shape is (1, 80, time_frames) where
    time_frames == fixed_size[1].
    """
    def __init__(self, en_ds, tr_ds, fixed_size=(80, 256)):
        self.en_ds = en_ds
        self.tr_ds = tr_ds
        self.fixed_size = fixed_size  # (n_mels, time_frames) = (80, 256)

    def __len__(self):
        return len(self.en_ds)

    def __getitem__(self, idx):
        # Retrieve pre-encoded mel-spectrograms (numpy arrays) from 'audio' field.
        # Shape on disk: (80, T)  — 80 mel bins, variable time length T.
        src_mel = self.en_ds[idx]['audio']
        tgt_mel = self.tr_ds[idx]['audio']

        def process(mel):
            if isinstance(mel, list):
                mel = np.array(mel)
            # mel shape: (80, T)  ->  tensor (1, 80, T)
            t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
            # Resize only the time axis to fixed_size[1]; keep n_mels=80 as-is.
            # F.interpolate expects (N, C, H, W); we treat (1, 1, 80, T).
            t_resized = F.interpolate(
                t.unsqueeze(0),          # (1, 1, 80, T)
                size=self.fixed_size,    # (80, time_frames)
                mode='bilinear',
                align_corners=False
            ).squeeze(0)                 # (1, 80, 256)
            # Normalize log-mel dB range [-80, 0] -> [0, 1].
            # This makes the loss landscape smoother and gradients more stable.
            t_resized = (t_resized + 80.0) / 80.0
            t_resized = t_resized.clamp(0.0, 1.0)
            return t_resized

        src = process(src_mel)
        tgt = process(tgt_mel)

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


class VisionTransformerBottleneck(nn.Module):
    """
    A Transformer bottleneck that flattens the spatial/temporal dimensions
    and applies self-attention. This allows the model to map features globally
    (e.g., routing an English word early in the sequence to a German word
    later in the sequence).
    """
    def __init__(self, channels=512, height=10, width=32, num_layers=4, num_heads=8):
        super().__init__()
        num_patches = height * width  # 10 * 32 = 320 patches
        
        # Learnable positional embeddings so the transformer knows "when" and "where" a feature is
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, channels) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        b, c, h, w = x.shape  # (Batch, 512, 10, 32)
        
        # Flatten spatial dims to sequence: (Batch, 320, 512)
        x_flat = x.view(b, c, h * w).transpose(1, 2)
        
        # Add position embeddings
        x_flat = x_flat + self.pos_embed
        
        # Apply global self-attention
        out = self.transformer(x_flat)
        
        # Reshape back to image grid: (Batch, 512, 10, 32)
        out = out.transpose(1, 2).view(b, c, h, w)
        return out


class SpectrogramDiscriminator(nn.Module):
    """
    Conditional PatchGAN discriminator for mel spectrograms (LSGAN variant).

    Takes a (source EN, target DE) pair and classifies each *patch* of the
    target as real or fake, conditioned on the source. Conditioning on the
    source prevents mode collapse: the discriminator must judge whether
    the DE output actually corresponds to the given EN input.

    Input:  src (B, 1, 80, 256) + tgt (B, 1, 80, 256)  →  cat  →  (B, 2, 80, 256)
    Output: (B, 1, H', W')  patch-level real/fake scores (no sigmoid — LSGAN uses MSE)
    """
    def __init__(self, in_channels=1, ndf=64):
        super().__init__()

        def block(in_c, out_c, stride=2, bn=True):
            layers = [nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=not bn)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        inp = in_channels * 2  # concat(src, tgt)
        self.model = nn.Sequential(
            *block(inp,    ndf,    stride=2, bn=False),  # (B, 64,  40, 128)
            *block(ndf,    ndf*2,  stride=2),             # (B, 128, 20,  64)
            *block(ndf*2,  ndf*4,  stride=2),             # (B, 256, 10,  32)
            *block(ndf*4,  ndf*8,  stride=1),             # (B, 512,  9,  31)
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=1),  # (B, 1, 8, 30)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        """DCGAN-style weight init for stable GAN training."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)

    def forward(self, src, tgt):
        return self.model(torch.cat([src, tgt], dim=1))


class UNetMelSpectrogram(nn.Module):
    """
    U-Net model that operates on 80-bin log-mel spectrograms.

    Input / output shape: (Batch, 1, 80, 256)  — 1 channel, 80 mel bands, 256 time frames.
    The encoder downsamples 80->40->20->10 (height) and 256->128->64->32 (width).
    All spatial sizes are divisible by the maximum stride product (2^3 = 8), so
    no padding hacks are needed in the skip connections.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetMelSpectrogram, self).__init__()

        # Encoder 1: Input -> 64 (Stride 1)
        self.enc1 = ResBlock(in_channels, 64, stride=1)

        # Encoder 2: 64 -> 128 (Stride 2)
        self.enc2 = ResBlock(64, 128, stride=2)

        # Encoder 3: 128 -> 256 (Stride 2)
        self.enc3 = ResBlock(128, 256, stride=2)

        # Bottleneck (Conv + Transformer)
        # Spatial size here is (10, 32)
        self.bottleneck_conv = ResBlock(256, 512, stride=2)
        self.bottleneck_attn = VisionTransformerBottleneck(channels=512, height=10, width=32, num_layers=4, num_heads=8)

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
        """Saves generator weights, discriminator weights, optimizer state, and epoch."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'discriminator_state_dict': self.discriminator.state_dict() if hasattr(self, 'discriminator') else None,
            'epoch': epoch,
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")

    def load(self, path, optimizer=None):
        """Loads generator weights, discriminator weights, and optimizer state."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(path, weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if hasattr(self, 'discriminator') and checkpoint.get('discriminator_state_dict'):
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
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
        b = self.bottleneck_conv(e3)
        b = self.bottleneck_attn(b)

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
        return torch.sigmoid(out)

    def train_model(self, epochs=10, batch_size=8, learning_rate=3e-4,
                    lambda_rec=10.0, num_workers=0,
                    data_dir='./datasets/processed_spectrogram_unet_en_de_v1'):
        """
        GAN training loop (LSGAN).

        Generator (UNet) loss  = λ_rec * (MSE + L1) + adversarial loss
        Discriminator loss     = 0.5 * (MSE(D(real),1) + MSE(D(fake),0))

        Args:
            lambda_rec:  Weight for the reconstruction term. Higher = closer
                         pixel-level match; lower = more GAN-driven texture.
                         Default 10 keeps output anchored while still learning
                         sharp spectral structure from the discriminator.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        self.to(device)

        # ── Losses ────────────────────────────────────────────────────────────
        criterion_mse = nn.MSELoss()
        criterion_l1  = nn.L1Loss()
        criterion_adv = nn.MSELoss()  # LSGAN uses MSE, not BCE

        # ── Discriminator ─────────────────────────────────────────────────────
        self.discriminator = SpectrogramDiscriminator(in_channels=1).to(device)

        # ── Optimizers (GAN convention: Adam with beta1=0.5) ──────────────────
        optimizer_G = optim.Adam(self.parameters(),                lr=learning_rate,       betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.discriminator.parameters(),  lr=learning_rate * 0.5, betas=(0.5, 0.999))
        self.optimizer = optimizer_G

        print(f"Starting GAN training... (λ_rec={lambda_rec})")
        self.train()
        self.discriminator.train()
        signal.signal(signal.SIGINT, self.interrupt_handler)

        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                print(f"--- Starting Epoch {epoch+1}/{epochs} ---")

                en_dir = os.path.join(data_dir, 'en')
                de_dir = os.path.join(data_dir, 'de')
                if not os.path.exists(en_dir) or not os.path.exists(de_dir):
                    print(f"Error: Dataset not found at {data_dir}")
                    print("  -> Run 'python preprocess_unet.py' first.")
                    return self

                # ── Load dataset once ──────────────────────────────────────────
                if epoch == 0:
                    print(f"Loading datasets from {data_dir} ...")
                    try:
                        en_ds = load_from_disk(en_dir)
                        de_ds = load_from_disk(de_dir)
                    except Exception as e:
                        print(f"Failed to load dataset: {e}")
                        return self
                    print(f"  Loaded {len(en_ds)} EN and {len(de_ds)} DE samples.")

                    dataset   = AlignedSpeechDataset(en_ds, de_ds)
                    dataloader = DataLoader(
                        dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                    )
                    num_batches = len(dataloader)

                    # OneCycleLR for generator (handles warmup automatically)
                    scheduler_G = optim.lr_scheduler.OneCycleLR(
                        optimizer_G, max_lr=learning_rate,
                        total_steps=num_batches * epochs,
                        pct_start=0.3, anneal_strategy='cos',
                        div_factor=25, final_div_factor=1e4,
                    )
                    # Cosine decay for discriminator (no warmup needed)
                    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
                        optimizer_D, T_max=num_batches * epochs, eta_min=1e-6,
                    )
                    print(f"  Batches/epoch: {num_batches} | Warmup: {int(0.3*num_batches*epochs)} steps")

                try:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    g_loss_sum = d_loss_sum = 0.0

                    for i, (inputs, targets) in enumerate(dataloader):
                        inputs, targets = inputs.to(device), targets.to(device)

                        # ── Step 1: Train Discriminator ────────────────────────
                        optimizer_D.zero_grad()

                        with torch.no_grad():
                            fake = self(inputs)             # generated DE spectrogram

                        real_score = self.discriminator(inputs, targets)
                        fake_score = self.discriminator(inputs, fake)

                        # LSGAN: D wants real→1, fake→0
                        loss_D = 0.5 * (
                            criterion_adv(real_score, torch.ones_like(real_score)) +
                            criterion_adv(fake_score, torch.zeros_like(fake_score))
                        )
                        loss_D.backward()
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                        optimizer_D.step()
                        scheduler_D.step()

                        # ── Step 2: Train Generator ────────────────────────────
                        optimizer_G.zero_grad()

                        fake       = self(inputs)           # re-generate (graph needed)
                        fake_score = self.discriminator(inputs, fake)

                        # LSGAN: G wants fake→1 (fool D)
                        loss_G_adv = 0.5 * criterion_adv(fake_score, torch.ones_like(fake_score))
                        # Reconstruction: keep output anchored to the target
                        loss_G_rec = criterion_mse(fake, targets) + criterion_l1(fake, targets)
                        loss_G     = loss_G_adv + lambda_rec * loss_G_rec

                        loss_G.backward()
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        optimizer_G.step()
                        scheduler_G.step()

                        g_loss_sum += loss_G.item()
                        d_loss_sum += loss_D.item()

                        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
                            lr_G = optimizer_G.param_groups[0]['lr']
                            print(
                                f"Epoch {epoch+1}, Step [{i+1}/{num_batches}], "
                                f"LR_G: {lr_G:.6f}, "
                                f"G_Loss: {g_loss_sum/(i+1):.4f}, "
                                f"D_Loss: {d_loss_sum/(i+1):.4f}",
                                flush=True,
                            )

                    if num_batches > 0:
                        print(
                            f"Epoch [{epoch+1}/{epochs}] Completed. "
                            f"Avg G_Loss: {g_loss_sum/num_batches:.4f} | "
                            f"Avg D_Loss: {d_loss_sum/num_batches:.4f}"
                        )
                    else:
                        print(f"Epoch [{epoch+1}/{epochs}] Completed. No data processed.")

                except Exception as e:
                    print(f"Error during training on epoch {epoch+1}: {e}")
                    raise

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving checkpoint...")
            self.save(os.path.join(data_dir, f'interrupted_model_epoch_{epoch+1}.pth'), optimizer_G, epoch)
            return self

        save_path = os.path.join(data_dir, f'gan_model_final_epoch_{epochs}.pth')
        self.save(save_path, optimizer_G, epochs)
        print(f"Training finished. Final model saved to {save_path}")
        return self
