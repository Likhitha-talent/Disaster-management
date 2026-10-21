"""
Loss functions for building segmentation and class-imbalanced post-disaster damage assessment.
Combines Focal / Weighted Cross-Entropy with multi-class Soft Dice Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Soft Dice loss for multi-class or binary segmentation.
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W)
        # targets: (B, H, W) integer classes
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - torch.mean(dice_score)


class CombinedDamageLoss(nn.Module):
    """
    Combined Class-Weighted Cross Entropy + Dice Loss for 4-class damage assessment.
    Weights are tuned to heavily penalize misclassification of rare disaster damage classes:
    [0: Undamaged, 1: Minor Damage, 2: Major Damage, 3: Destroyed]
    """
    def __init__(self, class_weights: Optional[torch.Tensor] = None, dice_weight: float = 0.5):
        super().__init__()
        if class_weights is None:
            # Typical inverse frequency weighting for xBD dataset
            class_weights = torch.tensor([1.0, 2.5, 3.5, 4.0])
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return ce_loss + self.dice_weight * dice_loss

