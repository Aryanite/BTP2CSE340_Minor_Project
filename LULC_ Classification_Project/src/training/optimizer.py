"""
TransUNet-RS — Optimizer & LR Scheduler
========================================
Factory functions for the AdamW optimizer and Cosine Annealing scheduler
with optional linear warmup.

Usage::

    from src.training.optimizer import build_optimizer, build_scheduler

    optimizer = build_optimizer(model, lr=1e-4)
    scheduler = build_scheduler(optimizer, T_max=100, warmup_epochs=5)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    SequentialLR,
)


# ===================================================================== #
#  Optimizer Builder
# ===================================================================== #
def build_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    no_decay_keywords: Optional[list] = None,
) -> AdamW:
    """Build an AdamW optimizer with optional per-parameter weight-decay
    exclusion (for biases, layer-norms, embeddings).

    Parameters
    ----------
    model : nn.Module
        The model to optimize.
    lr : float
        Base learning rate.
    weight_decay : float
        L2 regularization for non-excluded parameters.
    betas : tuple
        Adam beta coefficients.
    eps : float
        Adam epsilon for numerical stability.
    no_decay_keywords : list[str], optional
        Parameter name substrings that should not have weight decay
        (default: ["bias", "norm", "embed"]).

    Returns
    -------
    AdamW
    """
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "norm", "embed"]

    # Separate parameters into decay / no-decay groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name.lower() for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    return optimizer


# ===================================================================== #
#  Scheduler Builder
# ===================================================================== #
def build_scheduler(
    optimizer: AdamW,
    T_max: int = 100,
    eta_min: float = 1e-6,
    warmup_epochs: int = 5,
    warmup_lr: float = 1e-6,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build a Cosine Annealing LR scheduler with optional linear warmup.

    Parameters
    ----------
    optimizer : AdamW
        The optimizer.
    T_max : int
        Total number of epochs for cosine decay.
    eta_min : float
        Minimum learning rate at the end of cosine decay.
    warmup_epochs : int
        Number of warmup epochs (0 to disable).
    warmup_lr : float
        Starting LR for warmup (linearly ramps up to the base LR).

    Returns
    -------
    LR scheduler (SequentialLR with warmup + cosine, or plain cosine).
    """
    if warmup_epochs > 0:
        # Linear warmup
        base_lr = optimizer.defaults["lr"]

        def warmup_lambda(epoch: int) -> float:
            if epoch >= warmup_epochs:
                return 1.0
            return warmup_lr / base_lr + (1.0 - warmup_lr / base_lr) * (
                epoch / warmup_epochs
            )

        warmup_sched = LambdaLR(optimizer, lr_lambda=warmup_lambda)

        # Cosine decay after warmup
        cosine_sched = CosineAnnealingLR(
            optimizer,
            T_max=T_max - warmup_epochs,
            eta_min=eta_min,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )

    return scheduler


# ===================================================================== #
#  Quick test
# ===================================================================== #
if __name__ == "__main__":
    # Dummy model
    model = nn.Linear(10, 10)
    opt = build_optimizer(model, lr=1e-3, weight_decay=1e-4)
    sched = build_scheduler(opt, T_max=50, warmup_epochs=5)

    lrs = []
    for epoch in range(50):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()

    print("LR schedule (first 10):", [f"{lr:.6f}" for lr in lrs[:10]])
    print("LR schedule (last  5):", [f"{lr:.6f}" for lr in lrs[-5:]])
