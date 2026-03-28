"""
TransUNet-RS — Training Loop
==============================
Full training pipeline with:
  - Mixed-precision training (AMP)
  - Gradient clipping
  - MixUp augmentation
  - TensorBoard / WandB logging
  - Checkpointing (best + periodic)
  - Validation with mIoU tracking

Usage::

    python -m src.training.train --config configs/training_config.yaml
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import yaml

# Local imports
from src.models.transunet_rs import TransUNetRS
from src.training.loss import CombinedLoss, mixup_criterion
from src.training.optimizer import build_optimizer, build_scheduler
from src.dataset.data_loader import create_dataloaders
from src.dataset.preprocessing import get_train_transforms, get_val_transforms, mixup_batch
from src.evaluation.metrics import SegmentationMetrics


# ===================================================================== #
#  Reproducibility
# ===================================================================== #
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================================================================== #
#  Training Engine
# ===================================================================== #
class Trainer:
    """End-to-end training manager for TransUNet-RS.

    Parameters
    ----------
    config : dict
        Full training configuration (parsed from YAML).
    model_config_path : str
        Path to model_config.yaml (for model construction).
    """

    def __init__(
        self,
        config: dict,
        model_config_path: str = "configs/model_config.yaml",
    ) -> None:
        self.cfg = config
        self.train_cfg = config["training"]
        self.device = torch.device(
            self.train_cfg.get("device", "cuda")
            if torch.cuda.is_available()
            else "cpu"
        )

        # Seed
        set_seed(self.train_cfg.get("seed", 42))

        # ── Model ────────────────────────────────────────────────────
        if os.path.exists(model_config_path):
            self.model = TransUNetRS.from_config(model_config_path).to(self.device)
        else:
            self.model = TransUNetRS(
                num_classes=config.get("data", {}).get("num_classes", 10),
                img_size=config.get("data", {}).get("image_size", 256),
            ).to(self.device)

        # ── Loss ─────────────────────────────────────────────────────
        loss_cfg = config.get("loss", {})
        self.criterion = CombinedLoss(
            num_classes=config.get("data", {}).get("num_classes", 10),
            ce_weight=loss_cfg.get("ce_weight", 0.5),
            dice_weight=loss_cfg.get("dice_weight", 0.5),
            label_smoothing=loss_cfg.get("label_smoothing", 0.1),
            ignore_index=loss_cfg.get("ignore_index", -1),
        ).to(self.device)

        # ── Optimizer & Scheduler ────────────────────────────────────
        opt_cfg = config.get("optimizer", {})
        self.optimizer = build_optimizer(
            self.model,
            lr=opt_cfg.get("lr", 1e-4),
            weight_decay=opt_cfg.get("weight_decay", 1e-4),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        sched_cfg = config.get("scheduler", {})
        self.scheduler = build_scheduler(
            self.optimizer,
            T_max=sched_cfg.get("T_max", self.train_cfg["epochs"]),
            eta_min=sched_cfg.get("eta_min", 1e-6),
            warmup_epochs=sched_cfg.get("warmup_epochs", 5),
            warmup_lr=sched_cfg.get("warmup_lr", 1e-6),
        )

        # ── Mixed Precision ──────────────────────────────────────────
        self.use_amp = self.train_cfg.get("mixed_precision", True)
        self.scaler = GradScaler(enabled=self.use_amp)

        # ── MixUp ────────────────────────────────────────────────────
        aug_cfg = config.get("augmentation", {})
        self.use_mixup = aug_cfg.get("mixup", False)
        self.mixup_alpha = aug_cfg.get("mixup_alpha", 0.2)

        # ── Data Loaders ─────────────────────────────────────────────
        data_cfg = config.get("data", {})
        image_size = data_cfg.get("image_size", 256)
        train_tf = get_train_transforms(image_size=image_size)
        val_tf = get_val_transforms(image_size=image_size)

        self.loaders = create_dataloaders(
            data_dir=data_cfg.get("data_dir", "data/processed"),
            batch_size=self.train_cfg.get("batch_size", 16),
            val_batch_size=self.train_cfg.get("val_batch_size", 32),
            num_workers=self.train_cfg.get("num_workers", 4),
            target_size=(image_size, image_size),
            train_split=data_cfg.get("train_split", 0.7),
            val_split=data_cfg.get("val_split", 0.15),
            test_split=data_cfg.get("test_split", 0.15),
            train_transform=train_tf,
            val_transform=val_tf,
        )

        # ── Metrics ──────────────────────────────────────────────────
        num_classes = data_cfg.get("num_classes", 10)
        self.metrics = SegmentationMetrics(num_classes=num_classes)

        # ── Checkpointing ────────────────────────────────────────────
        ckpt_cfg = config.get("checkpoint", {})
        self.save_dir = Path(ckpt_cfg.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = ckpt_cfg.get("save_best", True)
        self.save_every = ckpt_cfg.get("save_every", 10)
        self.best_metric = 0.0
        self.monitored_metric = ckpt_cfg.get("metric", "miou")

        # ── Logging ──────────────────────────────────────────────────
        log_cfg = config.get("logging", {})
        self.print_freq = log_cfg.get("print_freq", 10)
        self.writer = None
        if log_cfg.get("use_tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=log_cfg.get("log_dir", "runs"))
            except ImportError:
                pass

    # ------------------------------------------------------------------ #
    #  Training epoch
    # ------------------------------------------------------------------ #
    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch.

        Returns
        -------
        float — average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        loader = self.loaders["train"]
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            # Optional MixUp
            if self.use_mixup and random.random() < 0.5:
                images, masks_a, masks_b, lam = mixup_batch(
                    images, masks, alpha=self.mixup_alpha
                )
                use_mixup_loss = True
            else:
                use_mixup_loss = False

            # Forward + loss (mixed precision)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                if use_mixup_loss:
                    loss = mixup_criterion(
                        self.criterion, logits, masks_a, masks_b, lam
                    )
                else:
                    loss = self.criterion(logits, masks)

            # Backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

            if batch_idx % self.print_freq == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    # ------------------------------------------------------------------ #
    #  Validation epoch
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Run validation and compute metrics.

        Returns
        -------
        dict — metrics including loss, oa, miou, f1, kappa.
        """
        self.model.eval()
        self.metrics.reset()
        total_loss = 0.0
        num_batches = 0

        loader = self.loaders["val"]
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val]  ", leave=False)

        for batch in pbar:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with autocast(enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            preds = logits.argmax(dim=1)
            self.metrics.update(preds, masks)
            total_loss += loss.item()
            num_batches += 1

        results = self.metrics.compute()
        results["loss"] = total_loss / max(num_batches, 1)
        return results

    # ------------------------------------------------------------------ #
    #  Checkpointing
    # ------------------------------------------------------------------ #
    def save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "best_metric": self.best_metric,
        }

        # Periodic save
        if (epoch + 1) % self.save_every == 0:
            path = self.save_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save(state, path)

        # Best model
        if is_best:
            path = self.save_dir / "best_model.pth"
            torch.save(state, path)

        # Always save latest
        torch.save(state, self.save_dir / "latest.pth")

    # ------------------------------------------------------------------ #
    #  Full training loop
    # ------------------------------------------------------------------ #
    def train(self) -> None:
        """Run the full training loop."""
        epochs = self.train_cfg["epochs"]
        print(f"\n{'='*60}")
        print(f"  TransUNet-RS Training")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {epochs}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            t0 = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            # Step scheduler
            self.scheduler.step()

            # Check best
            current_metric = val_metrics.get(self.monitored_metric, 0.0)
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric

            # Save
            self.save_checkpoint(epoch, val_metrics, is_best)

            # Log
            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"mIoU: {val_metrics.get('miou', 0):.4f} | "
                f"OA: {val_metrics.get('oa', 0):.4f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.1f}s"
                + (" ★" if is_best else "")
            )

            # TensorBoard
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
                self.writer.add_scalar("Metrics/mIoU", val_metrics.get("miou", 0), epoch)
                self.writer.add_scalar("Metrics/OA", val_metrics.get("oa", 0), epoch)
                self.writer.add_scalar("LR", lr, epoch)

        print(f"\nTraining complete. Best {self.monitored_metric}: {self.best_metric:.4f}")
        if self.writer:
            self.writer.close()


# ===================================================================== #
#  CLI Entry Point
# ===================================================================== #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train TransUNet-RS")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model config YAML",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    trainer = Trainer(config, model_config_path=args.model_config)
    trainer.train()


if __name__ == "__main__":
    main()
