"""
Stage B: Siamese U-Net Damage Assessment Model
Shared-weight dual-encoder with multi-scale feature difference and concatenation fusion
to classify damage into 4 categories:
  0: Undamaged
  1: Minor Damage
  2: Major Damage
  3: Destroyed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class SharedEncoderBranch(nn.Module):
    """
    Shared-weight encoder branch used for both pre-disaster and post-disaster tiles.
    """
    def __init__(self, in_channels: int = 3, base_filters: int = 32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters, base_filters * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 2, base_filters * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 4, base_filters * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 8, base_filters * 16))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.inc(x)
        e2 = self.down1(e1)
        e3 = self.down2(e2)
        e4 = self.down3(e3)
        e5 = self.down4(e4)
        return e1, e2, e3, e4, e5


class SiameseDamageUNet(nn.Module):
    """
    Shared-weight Siamese U-Net for bi-temporal disaster damage segmentation.
    Inputs:
      - pre_img: (B, 3, H, W)
      - post_img: (B, 3, H, W)
    Fusion:
      Concatenation of [Pre, Post, |Post - Pre|] at each skip connection level.
    Output:
      Logits for 4 damage classes (B, 4, H, W)
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 4, base_filters: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.base_filters = base_filters

        # Shared encoder
        self.encoder = SharedEncoderBranch(in_channels, base_filters)

        # Bottleneck fusion: pre (16F) + post (16F) + diff (16F) = 48F -> 16F
        b_ch = base_filters * 16
        self.bottleneck_fusion = nn.Sequential(
            nn.Conv2d(b_ch * 3, b_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(b_ch),
            nn.ReLU(inplace=True),
        )

        # Decoder stages with 1x1 convolutions after upsampling to align channel dims
        # Bottleneck: 16F
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_filters * 16, base_filters * 8, kernel_size=1)
        )
        # Skip 4: (8F + 8F + 8F) = 24F. Total in dec1 = 8F + 24F = 32F -> 8F
        self.dec1 = DoubleConv(base_filters * 8 + (base_filters * 8) * 3, base_filters * 8)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_filters * 8, base_filters * 4, kernel_size=1)
        )
        # Skip 3: (4F + 4F + 4F) = 12F. Total in dec2 = 4F + 12F = 16F -> 4F
        self.dec2 = DoubleConv(base_filters * 4 + (base_filters * 4) * 3, base_filters * 4)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_filters * 4, base_filters * 2, kernel_size=1)
        )
        # Skip 2: (2F + 2F + 2F) = 6F. Total in dec3 = 2F + 6F = 8F -> 2F
        self.dec3 = DoubleConv(base_filters * 2 + (base_filters * 2) * 3, base_filters * 2)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(base_filters * 2, base_filters, kernel_size=1)
        )
        # Skip 1: (1F + 1F + 1F) = 3F. Total in dec4 = 1F + 3F = 4F -> 1F
        self.dec4 = DoubleConv(base_filters + (base_filters * 1) * 3, base_filters)

        # Final classification head
        self.classifier = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def _fuse_features(self, feat_pre: torch.Tensor, feat_post: torch.Tensor) -> torch.Tensor:
        """Combine pre-disaster, post-disaster, and absolute difference features"""
        diff = torch.abs(feat_post - feat_pre)
        return torch.cat([feat_pre, feat_post, diff], dim=1)

    def forward(self, pre_img: torch.Tensor, post_img: torch.Tensor) -> torch.Tensor:
        # 1. Forward both images through the shared-weight encoder
        pre1, pre2, pre3, pre4, pre5 = self.encoder(pre_img)
        post1, post2, post3, post4, post5 = self.encoder(post_img)

        # 2. Fuse bottleneck
        f5 = self._fuse_features(pre5, post5)
        d5 = self.bottleneck_fusion(f5)

        # 3. Decoder with fused multi-scale skips
        d4_up = self.up1(d5)
        f4 = self._fuse_features(pre4, post4)
        d4 = self.dec1(torch.cat([d4_up, f4], dim=1))

        d3_up = self.up2(d4)
        f3 = self._fuse_features(pre3, post3)
        d3 = self.dec2(torch.cat([d3_up, f3], dim=1))

        d2_up = self.up3(d3)
        f2 = self._fuse_features(pre2, post2)
        d2 = self.dec3(torch.cat([d2_up, f2], dim=1))

        d1_up = self.up4(d2)
        f1 = self._fuse_features(pre1, post1)
        d1 = self.dec4(torch.cat([d1_up, f1], dim=1))

        # 4. Classify each pixel into 4 damage tiers
        logits = self.classifier(d1)
        return logits
