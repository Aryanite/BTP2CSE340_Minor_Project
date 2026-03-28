"""
TransUNet-RS — Data Loader
===========================
PyTorch Dataset and DataLoader factory for EuroSAT / Sentinel-2 imagery.

Supports two data layouts:
  1. **Classification-style** (EuroSAT default):
       data/processed/<class_name>/*.tif
     Each image is a 64×64 patch with a single whole-image label.
     For segmentation, each pixel inherits the scene label.

  2. **Segmentation-style** (custom prepared):
       data/processed/images/*.tif
       data/processed/masks/*.tif
     Paired image + per-pixel label mask.

The loader auto-detects the layout and applies transforms from
``preprocessing.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


# ===================================================================== #
#  EuroSAT class names
# ===================================================================== #
EUROSAT_CLASSES: List[str] = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(EUROSAT_CLASSES)}


# ===================================================================== #
#  Scene-Classification Dataset (EuroSAT default)
# ===================================================================== #
class EuroSATSceneDataset(Dataset):
    """EuroSAT scene-level classification dataset.

    Each sample is a 64×64 RGB image whose label is the containing folder
    name.  For segmentation training the label is expanded to a
    uniform mask of the same spatial size as the (resized) image.

    Parameters
    ----------
    root_dir : str | Path
        Path to the processed data directory containing class subfolders.
    transform : callable, optional
        Transform pipeline from ``preprocessing.py``.
    target_size : tuple[int, int]
        Resize images to this (H, W) for the model.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.target_size = target_size

        # Collect (image_path, class_idx) pairs
        self.samples: List[Tuple[Path, int]] = []
        for class_name in sorted(os.listdir(self.root_dir)):
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir() or class_name not in CLASS_TO_IDX:
                continue
            idx = CLASS_TO_IDX[class_name]
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith((".tif", ".jpg", ".png")):
                    self.samples.append((class_dir / fname, idx))

        if len(self.samples) == 0:
            raise FileNotFoundError(
                f"No images found in {self.root_dir}. "
                "Expected subdirectories named after EuroSAT classes."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, class_idx = self.samples[idx]

        # Load image (handle both TIFF and JPEG/PNG)
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.target_size, Image.BILINEAR)
        image = np.array(image, dtype=np.float32)  # [H, W, 3]

        # Create uniform segmentation mask (scene-level label → every pixel)
        mask = np.full(
            self.target_size, class_idx, dtype=np.int64
        )  # [H, W]

        # Apply augmentations (albumentations format)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()

        return {"image": image, "mask": mask, "label": class_idx}


# ===================================================================== #
#  Segmentation Dataset (image + mask pairs)
# ===================================================================== #
class SegmentationDataset(Dataset):
    """Generic paired image–mask segmentation dataset.

    Parameters
    ----------
    image_dir : str | Path
        Directory of input images.
    mask_dir : str | Path
        Directory of label masks (same filenames).
    transform : callable, optional
        Augmentation pipeline.
    target_size : tuple[int, int]
        Spatial resize.
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        transform: Optional[Callable] = None,
        target_size: Tuple[int, int] = (256, 256),
    ) -> None:
        super().__init__()
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.target_size = target_size

        self.filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith((".tif", ".jpg", ".png"))
        ])

        if len(self.filenames) == 0:
            raise FileNotFoundError(
                f"No images found in {self.image_dir}."
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fname = self.filenames[idx]

        image = Image.open(self.image_dir / fname).convert("RGB")
        image = image.resize(self.target_size, Image.BILINEAR)
        image = np.array(image, dtype=np.float32)

        mask_path = self.mask_dir / fname
        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")
            mask = mask.resize(self.target_size, Image.NEAREST)
            mask = np.array(mask, dtype=np.int64)
        else:
            mask = np.zeros(self.target_size, dtype=np.int64)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()

        return {"image": image, "mask": mask}


# ===================================================================== #
#  DataLoader Factory
# ===================================================================== #
def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 16,
    val_batch_size: int = 32,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (256, 256),
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Create train / val / test DataLoaders.

    Auto-detects the data layout:
      - If ``data_dir/images`` and ``data_dir/masks`` exist →
        SegmentationDataset
      - Otherwise → EuroSATSceneDataset (classification folders)

    Returns
    -------
    dict with keys ``"train"``, ``"val"``, ``"test"`` mapping to DataLoaders.
    """
    data_dir = Path(data_dir)

    # Detect layout
    if (data_dir / "images").is_dir() and (data_dir / "masks").is_dir():
        full_dataset = SegmentationDataset(
            image_dir=data_dir / "images",
            mask_dir=data_dir / "masks",
            transform=None,  # applied per-split below
            target_size=target_size,
        )
    else:
        full_dataset = EuroSATSceneDataset(
            root_dir=data_dir,
            transform=None,
            target_size=target_size,
        )

    # Split
    n = len(full_dataset)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Wrap with transforms
    train_ds.dataset.transform = train_transform
    # Note: val/test share the base dataset — we handle transform via
    # a wrapper to avoid contaminating val/test with train augmentation.
    class _TransformSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            sample = self.subset[idx]
            # The sample already went through dataset.__getitem__
            return sample

    val_wrapper = _TransformSubset(val_ds, val_transform)
    test_wrapper = _TransformSubset(test_ds, val_transform)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_wrapper,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            test_wrapper,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return loaders


# ===================================================================== #
#  Quick test
# ===================================================================== #
if __name__ == "__main__":
    print("EuroSAT classes:", EUROSAT_CLASSES)
    print("Class→Index map:", CLASS_TO_IDX)
