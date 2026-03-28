"""
TransUNet-RS — Preprocessing & Augmentation
=============================================
Provides Albumentations-based transform pipelines for training and
validation, including spectral jitter & MixUp augmentation.

Usage::

    from src.dataset.preprocessing import get_train_transforms, get_val_transforms

    train_tf = get_train_transforms(image_size=256)
    val_tf   = get_val_transforms(image_size=256)
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


# ===================================================================== #
#  Normalization constants (ImageNet)
# ===================================================================== #
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ===================================================================== #
#  Transform Factories
# ===================================================================== #
def get_train_transforms(
    image_size: int = 256,
    mean: Optional[list] = None,
    std: Optional[list] = None,
    spectral_jitter: bool = True,
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
) -> "A.Compose":
    """Build training augmentation pipeline.

    Includes:
      - Random horizontal / vertical flip
      - Random 90° rotation
      - Spectral jitter (brightness + contrast + hue-saturation)
      - Random resized crop
      - Normalize → ToTensor

    Parameters
    ----------
    image_size : int
        Target spatial dimension (square).
    mean, std : list[float]
        Normalization parameters (default: ImageNet).
    spectral_jitter : bool
        Enable brightness/contrast/color augmentation.
    brightness_limit, contrast_limit : float
        Limits for spectral jitter.

    Returns
    -------
    albumentations.Compose
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError(
            "albumentations is required for augmentation. "
            "Install via: pip install albumentations"
        )

    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    transforms = [
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1),
            p=0.5,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]

    if spectral_jitter:
        transforms.extend([
            A.RandomBrightnessContrast(
                brightness_limit=brightness_limit,
                contrast_limit=contrast_limit,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3,
            ),
            A.GaussNoise(p=0.2),
        ])

    transforms.extend([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(transforms)


def get_val_transforms(
    image_size: int = 256,
    mean: Optional[list] = None,
    std: Optional[list] = None,
) -> "A.Compose":
    """Validation / test transforms — resize + normalize only."""
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required.")

    mean = mean or IMAGENET_MEAN
    std = std or IMAGENET_STD

    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


# ===================================================================== #
#  MixUp Augmentation (applied at batch level)
# ===================================================================== #
def mixup_batch(
    images: "torch.Tensor",
    masks: "torch.Tensor",
    alpha: float = 0.2,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", float]:
    """Apply MixUp augmentation to a batch.

    Parameters
    ----------
    images : Tensor  [B, C, H, W]
    masks  : Tensor  [B, H, W]  (integer class labels)
    alpha  : float
        Beta distribution parameter.

    Returns
    -------
    mixed_images : Tensor  [B, C, H, W]
    masks_a, masks_b : Tensor  [B, H, W]
    lam : float
        Mixing coefficient.
    """
    import torch

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    masks_a = masks
    masks_b = masks[index]

    return mixed_images, masks_a, masks_b, lam


# ===================================================================== #
#  CLI: Data Preprocessing Utility
# ===================================================================== #
def preprocess_dataset(
    input_dir: str,
    output_dir: str,
    image_size: int = 256,
) -> None:
    """Copy and optionally resize a raw dataset to the processed directory.

    This handles the EuroSAT folder structure:
        input_dir/<ClassName>/*.tif  →  output_dir/<ClassName>/*.tif

    Parameters
    ----------
    input_dir : str
        Path to raw data (e.g. ``data/raw/eurosat``).
    output_dir : str
        Path to processed output (e.g. ``data/processed``).
    image_size : int
        Target spatial dimension for resizing.
    """
    from PIL import Image

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"[WARNING] Input directory does not exist: {input_path}")
        print("Please download EuroSAT and place it under data/raw/eurosat/")
        return

    count = 0
    for class_dir in sorted(input_path.iterdir()):
        if not class_dir.is_dir():
            continue
        out_class_dir = output_path / class_dir.name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        for img_file in sorted(class_dir.iterdir()):
            if not img_file.suffix.lower() in (".tif", ".jpg", ".png"):
                continue
            try:
                img = Image.open(img_file).convert("RGB")
                img = img.resize((image_size, image_size), Image.BILINEAR)
                out_path = out_class_dir / img_file.name
                img.save(out_path)
                count += 1
            except Exception as e:
                print(f"  [SKIP] {img_file.name}: {e}")

    print(f"[DONE] Processed {count} images → {output_path}")


# ===================================================================== #
#  Entry point
# ===================================================================== #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TransUNet-RS Data Preprocessing"
    )
    parser.add_argument(
        "--input", type=str, default="data/raw/eurosat",
        help="Path to raw data directory",
    )
    parser.add_argument(
        "--output", type=str, default="data/processed",
        help="Path to processed output directory",
    )
    parser.add_argument(
        "--image-size", type=int, default=256,
        help="Target image size",
    )
    args = parser.parse_args()

    preprocess_dataset(args.input, args.output, args.image_size)
