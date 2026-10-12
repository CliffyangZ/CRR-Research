"""BCE + soft Dice loss for binary segmentation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, bce: float = 1.0, dice: float = 1.0):
    return bce * F.binary_cross_entropy_with_logits(logits, target) + dice * soft_dice_loss(logits, target)
