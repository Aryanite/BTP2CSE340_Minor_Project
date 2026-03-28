"""
TransUNet-RS — CNN Encoder (ResNet-50 Backbone)
================================================
Extracts multi-scale feature maps from input images using a pretrained
ResNet-50.  Returns intermediate features for skip connections and the
final bottleneck feature map for the Transformer stage.

Feature map progression (input 3×256×256):
  stage1  →  64 × 128 × 128  (after conv1 + bn + relu)
  stage2  → 256 ×  64 ×  64  (after layer1)
  stage3  → 512 ×  32 ×  32  (after layer2)
  stage4  → 1024 ×  16 ×  16 (after layer3)  ← sent to transformer
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50Encoder(nn.Module):
    """ResNet-50 encoder that returns multi-scale skip-connection features.

    Parameters
    ----------
    pretrained : bool
        If ``True``, load ImageNet-pretrained weights.
    freeze_bn : bool
        If ``True``, freeze all BatchNorm layers (useful for fine-tuning with
        small batches).
    """

    # Channel dims produced at each extraction point
    STAGE_CHANNELS: List[int] = [64, 256, 512, 1024]

    def __init__(
        self,
        pretrained: bool = True,
        freeze_bn: bool = False,
    ) -> None:
        super().__init__()

        # Load base ResNet-50
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        # ── Stage 0: stem (conv1 → bn1 → relu → maxpool) ────────────
        self.stem = nn.Sequential(
            resnet.conv1,   # 3  → 64,  /2
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,  # 64 → 64,  /2  ⇒ 64×64×64
        )

        # ── Stages 1-3: residual blocks ──────────────────────────────
        self.stage1 = resnet.layer1  # 64  → 256,  64×64
        self.stage2 = resnet.layer2  # 256 → 512,  32×32
        self.stage3 = resnet.layer3  # 512 → 1024, 16×16
        # Note: we intentionally OMIT layer4 (2048ch) — the Transformer
        # bottleneck replaces it.

        if freeze_bn:
            self._freeze_batchnorm()

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run the encoder and collect skip features.

        Parameters
        ----------
        x : Tensor  [B, 3, H, W]

        Returns
        -------
        bottleneck : Tensor  [B, 1024, H/16, W/16]
            Feature map to feed into the Transformer.
        skips : dict[str, Tensor]
            ``"skip1"`` — [B,  64, H/4,  W/4]
            ``"skip2"`` — [B, 256, H/4,  W/4]
            ``"skip3"`` — [B, 512, H/8,  W/8]
        """
        # Stem: 3×256×256 → 64×64×64
        x0 = self.stem(x)

        # Stage 1: 64×64×64 → 256×64×64
        x1 = self.stage1(x0)

        # Stage 2: 256×64×64 → 512×32×32
        x2 = self.stage2(x1)

        # Stage 3: 512×32×32 → 1024×16×16
        x3 = self.stage3(x2)

        skips = {
            "skip1": x0,   # 64  ch, 1/4 resolution
            "skip2": x1,   # 256 ch, 1/4 resolution
            "skip3": x2,   # 512 ch, 1/8 resolution
        }

        return x3, skips

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def _freeze_batchnorm(self) -> None:
        """Freeze all BatchNorm parameters and set them to eval mode."""
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

    def get_output_channels(self) -> int:
        """Return the number of channels in the bottleneck output."""
        return self.STAGE_CHANNELS[-1]  # 1024


# ===================================================================== #
#  Quick smoke test
# ===================================================================== #
if __name__ == "__main__":
    model = ResNet50Encoder(pretrained=False)
    dummy = torch.randn(2, 3, 256, 256)
    bottleneck, skips = model(dummy)

    print("Bottleneck :", bottleneck.shape)   # [2, 1024, 16, 16]
    for k, v in skips.items():
        print(f"{k:>8s}    : {v.shape}")
