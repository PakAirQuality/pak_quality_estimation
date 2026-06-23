"""
U-Net Architecture (v2)
=======================

Small plain U-Net with 4-level conv encoder for gridded PM2.5 estimation.

Input:  [B, C, 141, 175]  (C = 66: 35 features + 31 availability masks)
Output: [B, 1, 281, 349]  (PM2.5 at 0.05° resolution, non-negative via Softplus)

The input is padded to 144x176 (divisible by 8) for clean encoder/decoder
halving, cropped back to 141x175, then bilinearly upsampled to 281x349
(0.05° output grid) for reduced station-pixel collisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    """
    Plain 4-level U-Net.

    Encoder:  66 -> 32 -> 64 -> 128
    Bottleneck: 128 -> 256
    Decoder:  256+128 -> 128, 128+64 -> 64, 64+32 -> 32
    Head:     32 -> 1, upsample to 0.05° grid, Softplus
    """

    # Input spatial size (0.1° grid)
    H_ORIG = 141
    W_ORIG = 175
    # Padded size (divisible by 8)
    H_PAD = 144
    W_PAD = 176
    # Output spatial size (0.05° grid)
    H_OUT = 281
    W_OUT = 349

    def __init__(self, in_channels: int = 66):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(128, 256)

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(256 + 128, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(128 + 64, 64)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(64 + 32, 32)

        # Head
        self.head = nn.Conv2d(32, 1, kernel_size=1)
        self.activation = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, 141, 175] input feature grid at 0.1°

        Returns:
            [B, 1, 281, 349] predicted PM2.5 map at 0.05° (non-negative)
        """
        # Pad to 144x176 with reflect padding
        x = F.pad(x, (0, self.W_PAD - self.W_ORIG, 0, self.H_PAD - self.H_ORIG), mode="reflect")

        # Encoder
        e1 = self.enc1(x)                     # [B, 32, 144, 176]
        e2 = self.enc2(self.pool(e1))          # [B, 64, 72, 88]
        e3 = self.enc3(self.pool(e2))          # [B, 128, 36, 44]

        # Bottleneck
        b = self.bottleneck(self.pool(e3))     # [B, 256, 18, 22]

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))   # [B, 128, 36, 44]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # [B, 64, 72, 88]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # [B, 32, 144, 176]

        # Head → crop to 0.1° grid → upsample to 0.05° grid
        out = self.head(d1)                                     # [B, 1, 144, 176]
        out = out[:, :, : self.H_ORIG, : self.W_ORIG]          # [B, 1, 141, 175]
        out = F.interpolate(out, size=(self.H_OUT, self.W_OUT),
                            mode="bilinear", align_corners=True) # [B, 1, 281, 349]
        out = self.activation(out)                               # non-negative

        return out
