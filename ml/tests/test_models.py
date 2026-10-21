"""
Unit Tests for U-Net Building Localizer and Siamese Damage Assessment Architectures
"""

import torch
import pytest
from ml.src.models.unet import BuildingLocalizationUNet
from ml.src.models.siamese_unet import SiameseDamageUNet
from ml.src.models.losses import CombinedDamageLoss


def test_unet_forward():
    model = BuildingLocalizationUNet(in_channels=3, out_channels=1, base_filters=16)
    model.eval()

    # Batch of 2 RGB tiles of size 128x128
    dummy_input = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (2, 1, 128, 128), f"Unexpected shape {output.shape}"


def test_siamese_damage_unet_forward():
    model = SiameseDamageUNet(in_channels=3, num_classes=4, base_filters=16)
    model.eval()

    # Pre- and post-disaster paired tiles
    pre_tile = torch.randn(2, 3, 128, 128)
    post_tile = torch.randn(2, 3, 128, 128)

    with torch.no_grad():
        output = model(pre_tile, post_tile)

    # Must produce 4 damage class logits per pixel
    assert output.shape == (2, 4, 128, 128), f"Unexpected shape {output.shape}"


def test_combined_damage_loss():
    loss_fn = CombinedDamageLoss()
    logits = torch.randn(2, 4, 64, 64)
    targets = torch.randint(0, 4, (2, 64, 64))

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)

