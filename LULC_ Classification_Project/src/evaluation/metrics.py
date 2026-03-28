"""
TransUNet-RS — Evaluation Metrics
==================================
Computes standard remote-sensing segmentation metrics:
  - Overall Accuracy (OA)
  - Mean Intersection over Union (mIoU)
  - Per-class F1 Score (macro average)
  - Cohen's Kappa coefficient

Designed as an accumulator: call ``update()`` per batch, then ``compute()``
once at the end of an epoch.

Usage::

    metrics = SegmentationMetrics(num_classes=10)
    for preds, targets in dataloader:
        metrics.update(preds, targets)
    results = metrics.compute()
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulates a confusion matrix and computes segmentation metrics.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes.
    ignore_index : int
        Class index to ignore (e.g. -1 for unlabeled).
    """

    def __init__(self, num_classes: int = 10, ignore_index: int = -1) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.int64
        )

    def reset(self) -> None:
        """Reset the confusion matrix."""
        self.confusion_matrix.fill(0)

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Accumulate predictions into the confusion matrix.

        Parameters
        ----------
        preds   : Tensor [B, H, W]  — predicted class indices.
        targets : Tensor [B, H, W]  — ground-truth class indices.
        """
        preds_np = preds.detach().cpu().numpy().flatten()
        targets_np = targets.detach().cpu().numpy().flatten()

        # Filter out ignore_index
        valid = targets_np != self.ignore_index
        preds_np = preds_np[valid]
        targets_np = targets_np[valid]

        # Build confusion matrix
        for t, p in zip(targets_np, preds_np):
            if 0 <= t < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[t, p] += 1

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from the accumulated confusion matrix.

        Returns
        -------
        dict with keys: oa, miou, f1, kappa, per_class_iou, per_class_f1
        """
        cm = self.confusion_matrix.astype(np.float64)

        # ── Overall Accuracy ─────────────────────────────────────────
        total = cm.sum()
        oa = cm.trace() / max(total, 1)

        # ── Per-class IoU ────────────────────────────────────────────
        per_class_iou = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            denom = tp + fp + fn
            per_class_iou[c] = tp / max(denom, 1)

        # Mean IoU (only classes with > 0 support)
        valid_classes = cm.sum(axis=1) > 0
        miou = per_class_iou[valid_classes].mean() if valid_classes.any() else 0.0

        # ── Per-class F1 (Dice) ──────────────────────────────────────
        per_class_f1 = np.zeros(self.num_classes)
        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            per_class_f1[c] = (
                2 * precision * recall / max(precision + recall, 1e-8)
            )

        macro_f1 = per_class_f1[valid_classes].mean() if valid_classes.any() else 0.0

        # ── Cohen's Kappa ────────────────────────────────────────────
        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)
        expected = (row_sums * col_sums).sum() / max(total ** 2, 1)
        observed = oa
        kappa = (observed - expected) / max(1 - expected, 1e-8)

        return {
            "oa": float(oa),
            "miou": float(miou),
            "f1": float(macro_f1),
            "kappa": float(kappa),
            "per_class_iou": per_class_iou.tolist(),
            "per_class_f1": per_class_f1.tolist(),
        }

    def get_confusion_matrix(self) -> np.ndarray:
        """Return the raw confusion matrix."""
        return self.confusion_matrix.copy()


# ===================================================================== #
#  Quick test
# ===================================================================== #
if __name__ == "__main__":
    metrics = SegmentationMetrics(num_classes=5)

    # Simulate some predictions
    preds = torch.randint(0, 5, (4, 64, 64))
    targets = torch.randint(0, 5, (4, 64, 64))

    metrics.update(preds, targets)
    results = metrics.compute()

    print("Overall Accuracy:", f"{results['oa']:.4f}")
    print("Mean IoU:        ", f"{results['miou']:.4f}")
    print("Macro F1:        ", f"{results['f1']:.4f}")
    print("Cohen's Kappa:   ", f"{results['kappa']:.4f}")
    print("Per-class IoU:   ", [f"{v:.3f}" for v in results['per_class_iou']])
