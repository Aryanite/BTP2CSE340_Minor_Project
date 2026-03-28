"""
TransUNet-RS — Visualization Utilities
========================================
Functions for creating publication-quality plots:
  - Confusion matrix heatmap
  - Prediction overlay (image + mask + pred)
  - Per-class performance bar charts
  - Training curve plots

All functions return matplotlib Figure objects for flexibility.

Usage::

    from src.evaluation.visualize import plot_confusion_matrix, plot_prediction
    fig = plot_confusion_matrix(cm, class_names)
    fig.savefig("confusion_matrix.png", dpi=150)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import torch


# ===================================================================== #
#  Color palette for LULC classes
# ===================================================================== #
LULC_COLORS: List[Tuple[int, int, int]] = [
    (255, 255, 100),   # AnnualCrop       — light yellow
    (0, 128, 0),       # Forest           — dark green
    (124, 252, 0),     # HerbaceousVeg    — lime green
    (128, 128, 128),   # Highway          — gray
    (255, 0, 0),       # Industrial       — red
    (144, 238, 144),   # Pasture          — light green
    (255, 165, 0),     # PermanentCrop    — orange
    (255, 255, 255),   # Residential      — white
    (0, 0, 255),       # River            — blue
    (0, 191, 255),     # SeaLake          — deep sky blue
]

DEFAULT_CLASS_NAMES: List[str] = [
    "AnnualCrop", "Forest", "HerbVeg", "Highway", "Industrial",
    "Pasture", "PermCrop", "Residential", "River", "SeaLake",
]


# ===================================================================== #
#  Confusion Matrix
# ===================================================================== #
def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    confusion_matrix : np.ndarray  [C, C]
    class_names : list[str]
    normalize : bool
        If True, normalize rows to show percentages.
    figsize : tuple
    cmap : str
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES[: confusion_matrix.shape[0]]

    cm = confusion_matrix.copy().astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm = cm / row_sums
        fmt = ".2f"
        vmax = 1.0
    else:
        fmt = "d"
        cm = cm.astype(int)
        vmax = None

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        vmin=0,
        vmax=vmax,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ===================================================================== #
#  Prediction Overlay
# ===================================================================== #
def colorize_mask(
    mask: np.ndarray,
    colors: Optional[List[Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Convert a class-index mask to an RGB image.

    Parameters
    ----------
    mask : np.ndarray [H, W]  — integer class indices.
    colors : list of (R, G, B) tuples.

    Returns
    -------
    np.ndarray [H, W, 3]  — uint8 RGB.
    """
    if colors is None:
        colors = LULC_COLORS

    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(colors):
        rgb[mask == cls_idx] = color
    return rgb


def plot_prediction(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    class_names: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    alpha: float = 0.5,
    figsize: Tuple[int, int] = (18, 5),
) -> plt.Figure:
    """Show image, ground truth, and prediction side by side.

    Parameters
    ----------
    image : np.ndarray [H, W, 3]  — RGB input image (0-255).
    ground_truth : np.ndarray [H, W]  — true class indices.
    prediction : np.ndarray [H, W]  — predicted class indices.
    class_names : list[str]
    colors : list of (R, G, B)
    alpha : float — overlay transparency.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if colors is None:
        colors = LULC_COLORS
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    gt_rgb = colorize_mask(ground_truth, colors)
    pred_rgb = colorize_mask(prediction, colors)

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Input Image", fontsize=12)
    axes[0].axis("off")

    # Ground truth
    axes[1].imshow(gt_rgb)
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis("off")

    # Prediction
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction", fontsize=12)
    axes[2].axis("off")

    # Overlay
    overlay = image.copy()
    overlay = (overlay * (1 - alpha) + pred_rgb * alpha).astype(np.uint8)
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay", fontsize=12)
    axes[3].axis("off")

    plt.suptitle("Segmentation Result", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ===================================================================== #
#  Per-Class Performance Bar Chart
# ===================================================================== #
def plot_per_class_metrics(
    per_class_iou: List[float],
    per_class_f1: List[float],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Bar chart of per-class IoU and F1 scores.

    Parameters
    ----------
    per_class_iou : list[float]  — IoU per class.
    per_class_f1  : list[float]  — F1 per class.
    class_names : list[str]
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES[: len(per_class_iou)]

    n = len(class_names)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width / 2, per_class_iou, width, label="IoU", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, per_class_f1, width, label="F1", color="#DD8452")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Performance", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return fig


# ===================================================================== #
#  Training Curves
# ===================================================================== #
def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_mious: List[float],
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """Plot training/validation loss and mIoU curves.

    Parameters
    ----------
    train_losses : list[float]
    val_losses : list[float]
    val_mious : list[float]
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Loss
    ax1.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-", label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # mIoU
    ax2.plot(epochs, val_mious, "g-", label="Val mIoU", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("mIoU")
    ax2.set_title("Validation mIoU", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


# ===================================================================== #
#  Quick test
# ===================================================================== #
if __name__ == "__main__":
    # Test confusion matrix
    cm = np.random.randint(0, 100, (10, 10))
    np.fill_diagonal(cm, np.random.randint(200, 500, 10))
    fig = plot_confusion_matrix(cm)
    fig.savefig("test_confusion_matrix.png", dpi=100, bbox_inches="tight")
    print("Saved test_confusion_matrix.png")

    # Test per-class metrics
    iou = np.random.uniform(0.5, 0.95, 10).tolist()
    f1 = np.random.uniform(0.6, 0.97, 10).tolist()
    fig2 = plot_per_class_metrics(iou, f1)
    fig2.savefig("test_per_class.png", dpi=100, bbox_inches="tight")
    print("Saved test_per_class.png")
