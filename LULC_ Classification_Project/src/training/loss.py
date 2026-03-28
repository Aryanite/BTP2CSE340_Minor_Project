"""
TransUNet-RS — Loss Functions
==============================
Combined CrossEntropy + Dice loss for semantic segmentation.

The hybrid loss balances pixel-wise classification accuracy (CE) with
region-level overlap (Dice), which is especially helpful for imbalanced
LULC classes.

Usage::

    criterion = CombinedLoss(num_classes=10, ce_weight=0.5, dice_weight=0.5)
    loss = criterion(logits, targets)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================== #
#  Dice Loss
# ===================================================================== #
class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes.
    smooth : float
        Smoothing factor to avoid division by zero.
    ignore_index : int
        Class index to ignore (e.g. -1 for unlabeled pixels).
    """

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1.0,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : Tensor  [B, C, H, W]  — raw model output (before softmax).
        targets : Tensor  [B, H, W]     — ground-truth class indices.

        Returns
        -------
        Tensor  scalar — mean Dice loss across classes.
        """
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # Create valid mask (exclude ignore_index)
        valid_mask = (targets != self.ignore_index)  # [B, H, W]

        # One-hot encode targets
        targets_clamped = targets.clone()
        targets_clamped[~valid_mask] = 0
        one_hot = F.one_hot(
            targets_clamped, self.num_classes
        ).permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Apply valid mask
        valid_mask_4d = valid_mask.unsqueeze(1).float()  # [B, 1, H, W]
        probs = probs * valid_mask_4d
        one_hot = one_hot * valid_mask_4d

        # Compute Dice per class
        dims = (0, 2, 3)  # reduce over batch and spatial dims
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (
            cardinality + self.smooth
        )
        dice_loss = 1.0 - dice_score.mean()

        return dice_loss


# ===================================================================== #
#  Combined CE + Dice Loss
# ===================================================================== #
class CombinedLoss(nn.Module):
    """Weighted combination of CrossEntropy and Dice losses.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes.
    ce_weight : float
        Weight for the CrossEntropy component.
    dice_weight : float
        Weight for the Dice component.
    label_smoothing : float
        Label smoothing for CrossEntropy.
    ignore_index : int
        Class index to ignore.
    class_weights : Tensor, optional
        Per-class weights for CrossEntropy (e.g. inverse frequency).
    """

    def __init__(
        self,
        num_classes: int = 10,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        label_smoothing: float = 0.1,
        ignore_index: int = -1,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : Tensor [B, C, H, W]
        targets : Tensor [B, H, W]

        Returns
        -------
        Tensor  scalar — combined loss.
        """
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


# ===================================================================== #
#  MixUp-aware Loss
# ===================================================================== #
def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute loss for MixUp-augmented batches.

    Parameters
    ----------
    criterion : nn.Module
        Base loss function (e.g. CombinedLoss).
    logits : Tensor [B, C, H, W]
    targets_a, targets_b : Tensor [B, H, W]
        The two sets of targets from MixUp.
    lam : float
        Mixing coefficient.

    Returns
    -------
    Tensor  scalar
    """
    return lam * criterion(logits, targets_a) + (1 - lam) * criterion(
        logits, targets_b
    )


# ===================================================================== #
#  Quick test
# ===================================================================== #
if __name__ == "__main__":
    B, C, H, W = 4, 10, 64, 64
    logits = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))

    loss_fn = CombinedLoss(num_classes=C)
    loss = loss_fn(logits, targets)
    print(f"Combined Loss: {loss.item():.4f}")
