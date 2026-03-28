"""
TransUNet-RS — Inference / Prediction Script
==============================================
Load a trained checkpoint and run inference on single images or entire
directories.  Produces colorized segmentation maps and optionally saves
prediction masks.

Usage::

    # Single image
    python -m src.inference.predict \\
        --checkpoint checkpoints/best_model.pth \\
        --input test_image.tif \\
        --output results/

    # Directory of images
    python -m src.inference.predict \\
        --checkpoint checkpoints/best_model.pth \\
        --input data/processed/test/ \\
        --output results/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.transunet_rs import TransUNetRS
from src.evaluation.visualize import colorize_mask, LULC_COLORS
from src.dataset.data_loader import EUROSAT_CLASSES


# ===================================================================== #
#  Predictor
# ===================================================================== #
class Predictor:
    """Load a TransUNet-RS checkpoint and run inference.

    Parameters
    ----------
    checkpoint_path : str
        Path to the model checkpoint (.pth).
    model_config_path : str
        Path to model_config.yaml.
    device : str
        Device to use ("cuda" or "cpu").
    image_size : int
        Input image size (spatial dimension).
    """

    # ImageNet normalization constants
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        checkpoint_path: str,
        model_config_path: str = "configs/model_config.yaml",
        device: str = "cuda",
        image_size: int = 256,
    ) -> None:
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        self.image_size = image_size

        # Build model
        if os.path.exists(model_config_path):
            self.model = TransUNetRS.from_config(model_config_path)
        else:
            self.model = TransUNetRS(num_classes=10, img_size=image_size)

        # Load weights
        if os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"[WARNING] Checkpoint not found: {checkpoint_path}")
            print("         Running with random weights (demo mode).")

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a single image for inference.

        Parameters
        ----------
        image : np.ndarray [H, W, 3]  — RGB uint8.

        Returns
        -------
        Tensor [1, 3, image_size, image_size]
        """
        # Resize
        img = Image.fromarray(image).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        img = np.array(img, dtype=np.float32) / 255.0

        # Normalize
        img = (img - self.MEAN) / self.STD

        # To tensor [1, C, H, W]
        tensor = (
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        )
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on a single image.

        Parameters
        ----------
        image : np.ndarray [H, W, 3]  — RGB uint8.

        Returns
        -------
        pred_mask : np.ndarray [H, W]  — predicted class indices.
        pred_rgb : np.ndarray [H, W, 3]  — colorized prediction mask.
        prob_map : np.ndarray [C, H, W]  — class probabilities.
        """
        original_h, original_w = image.shape[:2]
        tensor = self.preprocess(image)

        logits = self.model(tensor)  # [1, C, H, W]
        probs = F.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)  # [1, H, W]

        # Resize back to original dimensions
        pred_resized = F.interpolate(
            pred.unsqueeze(1).float(),
            size=(original_h, original_w),
            mode="nearest",
        ).squeeze().long()

        prob_resized = F.interpolate(
            probs,
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        pred_mask = pred_resized.cpu().numpy()
        prob_map = prob_resized.cpu().numpy()
        pred_rgb = colorize_mask(pred_mask)

        return pred_mask, pred_rgb, prob_map

    def predict_file(
        self, image_path: str, output_dir: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load an image file, predict, and optionally save results.

        Parameters
        ----------
        image_path : str
            Path to the input image.
        output_dir : str, optional
            If provided, save the colorized prediction here.

        Returns
        -------
        pred_mask, pred_rgb
        """
        image = np.array(Image.open(image_path).convert("RGB"))
        pred_mask, pred_rgb, _ = self.predict(image)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fname = Path(image_path).stem
            # Save colorized prediction
            pred_img = Image.fromarray(pred_rgb)
            pred_img.save(os.path.join(output_dir, f"{fname}_pred.png"))
            # Save raw mask
            mask_img = Image.fromarray(pred_mask.astype(np.uint8))
            mask_img.save(os.path.join(output_dir, f"{fname}_mask.png"))

        return pred_mask, pred_rgb


# ===================================================================== #
#  CLI
# ===================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(description="TransUNet-RS Inference")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model-config", type=str, default="configs/model_config.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input image file or directory",
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--image-size", type=int, default=256,
        help="Model input size",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)",
    )
    args = parser.parse_args()

    predictor = Predictor(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        device=args.device,
        image_size=args.image_size,
    )

    input_path = Path(args.input)

    if input_path.is_file():
        # Single image
        print(f"Processing: {input_path}")
        predictor.predict_file(str(input_path), args.output)
        print(f"Result saved to: {args.output}")

    elif input_path.is_dir():
        # Directory
        extensions = (".tif", ".jpg", ".jpeg", ".png")
        files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]
        print(f"Found {len(files)} images in {input_path}")

        for img_file in files:
            print(f"  Processing: {img_file.name}")
            predictor.predict_file(str(img_file), args.output)

        print(f"\nAll results saved to: {args.output}")
    else:
        print(f"[ERROR] Input not found: {input_path}")


if __name__ == "__main__":
    main()
