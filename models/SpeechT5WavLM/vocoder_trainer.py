import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm

# =============================================================================
# Discriminator Architectures (HiFi-GAN)
# =============================================================================

LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class DiscriminatorP(torch.nn.Module):
    """Period Discriminator for HiFi-GAN."""
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []
        
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        
        return x, fmap

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class DiscriminatorS(torch.nn.Module):
    """Scale Discriminator for HiFi-GAN."""
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

# =============================================================================
# Discriminator Losses
# =============================================================================

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

# =============================================================================
# Dataset and Dataloader
# =============================================================================

class SpeechT5VocoderDataset(Dataset):
    """
    Dataset for HiFi-GAN Vocoder fine-tuning.
    Reads: predicted_mel (cached), labels (Mel), target_waveform (Audio).
    """
    def __init__(self, ds_path, speaker_embeddings, max_wav_value=32768.0):
        self.ds = load_from_disk(ds_path)
        
        # --- THE FIX: Force zero-copy PyTorch tensors ---
        self.ds.set_format(type="torch", columns=["predicted_mel", "labels", "target_waveform"])
        
        self.speaker_embeddings = speaker_embeddings
        self.max_wav_value = max_wav_value

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[int(idx)]

        # Because of set_format("torch"), these are ALREADY PyTorch tensors!
        # No more slow CPU np.array() conversions.
        predicted_mel = row["predicted_mel"].float()
        target_features = row["labels"].float()
        waveform = row["target_waveform"].float().unsqueeze(0) # (1, T)

        # -------------------------------------------------------------
        # Keep your existing shape alignment logic just in case:
        # -------------------------------------------------------------
        if predicted_mel.ndim == 1:
            predicted_mel = predicted_mel.reshape(-1, 80)

        if target_features.ndim == 1:
            if target_features.numel() % 80 == 0:
                target_features = target_features.reshape(-1, 80)
        elif target_features.ndim == 3:
            target_features = target_features.squeeze(0)

        if target_features.shape[0] == 80 and target_features.shape[1] != 80:
            target_features = target_features.transpose(0, 1)

        if target_features.shape[0] % 2 != 0:
            target_features = target_features[:-1, :]

        return {
            "predicted_mel": predicted_mel,
            "labels": target_features,
            "target_waveform": waveform
        }

def vocoder_collate_fn(batch):
    predicted_mels = [item["predicted_mel"] for item in batch]
    labels = [item["labels"] for item in batch]
    waveforms = [item["target_waveform"].squeeze(0) for item in batch]

    predicted_mels_padded = torch.nn.utils.rnn.pad_sequence(predicted_mels, batch_first=True, padding_value=-100.0)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100.0)
    
    # Pad waveforms
    waveforms_padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
    waveforms_padded = waveforms_padded.unsqueeze(1) # (B, 1, T)

    return predicted_mels_padded, labels_padded, waveforms_padded

# =============================================================================
# Trainer
# =============================================================================

class VocoderTrainer:
    def __init__(self, model, vocoder, device, target_embeddings):
        self.model = model  # Kept for compatibility but no longer used in training loop
        self.vocoder = vocoder
        self.device = device
        self.target_embeddings = target_embeddings

        self.mpd = MultiPeriodDiscriminator().to(device)
        self.msd = MultiScaleDiscriminator().to(device)

    def train(self, preprocessed_path, epochs, learning_rate, batch_size, save_callback=None):
        print("Initializing Vocoder Trainer...")
        
        # Datasets
        train_dataset = SpeechT5VocoderDataset(preprocessed_path, self.target_embeddings)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=vocoder_collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        # Optimizers
        optim_g = torch.optim.AdamW(self.vocoder.parameters(), lr=learning_rate, betas=(0.8, 0.99))
        optim_d = torch.optim.AdamW(
            list(self.mpd.parameters()) + list(self.msd.parameters()), 
            lr=learning_rate, betas=(0.8, 0.99)
        )

        l1_criterion = torch.nn.L1Loss()

        self.vocoder.train()
        self.mpd.train()
        self.msd.train()

        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for step, (predicted_mel, labels, target_waveform) in enumerate(pbar):
                
                predicted_mel = predicted_mel.to(self.device)
                labels = labels.to(self.device)
                target_waveform = target_waveform.to(self.device)

                # =================================================================
                # 1. Forward pass through frozen SpeechT5 is completely REMOVED!
                #    We replace it with zeroing out padded frames in the cached mel
                # =================================================================
                # Zero out the padding (-100.0) so it doesn't break the vocoder
                predicted_mel_clean = predicted_mel.clone()
                predicted_mel_clean[predicted_mel == -100.0] = 0.0

                # =================================================================
                # 2. Generator (Vocoder) Forward
                # =================================================================
                # SpeechT5HifiGan expects (B, T, 80) and handles internal transpose
                y_hat = self.vocoder(predicted_mel_clean) # (B, T_wav)
                if y_hat.ndim == 2:
                    y_hat = y_hat.unsqueeze(1) # (B, 1, T_wav)
                
                # Match lengths of generated audio and target audio
                min_len = min(y_hat.shape[-1], target_waveform.shape[-1])
                y_hat = y_hat[:, :, :min_len]
                y = target_waveform[:, :, :min_len]

                # =================================================================
                # 3. Train Discriminator
                # =================================================================
                optim_d.zero_grad()
                
                # MPD
                y_d_rs, y_d_gs, _, _ = self.mpd(y, y_hat.detach())
                loss_disc_f, _, _ = discriminator_loss(y_d_rs, y_d_gs)
                
                # MSD
                y_ds_rs, y_ds_gs, _, _ = self.msd(y, y_hat.detach())
                loss_disc_s, _, _ = discriminator_loss(y_ds_rs, y_ds_gs)
                
                loss_disc_all = loss_disc_f + loss_disc_s
                loss_disc_all.backward()
                optim_d.step()

                # =================================================================
                # 4. Train Generator
                # =================================================================
                optim_g.zero_grad()

                # L1 Mel-Spectrogram Loss
                min_mel_len = min(predicted_mel_clean.shape[1], labels.shape[1])
                predicted_mel_clean_loss = predicted_mel_clean[:, :min_mel_len, :]
                labels_clean = labels[:, :min_mel_len, :].clone()
                labels_clean[labels_clean == -100.0] = 0.0
                
                # Compare the precomputed predicted mel against target mel
                mel_loss = l1_criterion(predicted_mel_clean_loss, labels_clean)

                # Adversarial Loss
                _, y_d_gs, fmap_rs, fmap_gs = self.mpd(y, y_hat)
                loss_gen_f, _ = generator_loss(y_d_gs)
                loss_fm_f = feature_loss(fmap_rs, fmap_gs)

                _, y_ds_gs, fmap_rs_s, fmap_gs_s = self.msd(y, y_hat)
                loss_gen_s, _ = generator_loss(y_ds_gs)
                loss_fm_s = feature_loss(fmap_rs_s, fmap_gs_s)

                loss_gen_all = loss_gen_f + loss_gen_s + loss_fm_f + loss_fm_s + 45.0 * mel_loss
                loss_gen_all.backward()
                optim_g.step()

                # Update progress bar
                pbar.set_postfix({
                    "D_Loss": f"{loss_disc_all.item():.4f}",
                    "G_Loss": f"{loss_gen_all.item():.4f}"
                })

            if save_callback is not None:
                save_callback(epoch + 1)

        print("Vocoder fine-tuning complete!")
