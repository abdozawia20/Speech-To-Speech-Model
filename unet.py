import torch
import torch.nn as nn

class UNetSpectrogramTranslator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetSpectrogramTranslator, self).__init__()

        # Encoder 1: No downsampling (stays at input resolution)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Encoder 2: Downsample by 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Encoder 3: Downsample by 2 (Total /4)
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Bottleneck: Downsample by 2 (Total /8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder 3: Upsample to match Enc3
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_up3 = nn.BatchNorm2d(256)
        # Fuse with Enc3 (256 + 256 -> 256)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Decoder 2: Upsample to match Enc2
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_up2 = nn.BatchNorm2d(128)
        # Fuse with Enc2 (128 + 128 -> 128)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Decoder 1: Upsample to match Enc1
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_up1 = nn.BatchNorm2d(64)
        # Fuse with Enc1 (64 + 64 -> 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final Output Layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

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
