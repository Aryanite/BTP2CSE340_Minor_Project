"""
TransUNet-RS — Full Architecture Assembly
==========================================
Composes the ResNet-50 CNN encoder, Vision Transformer bottleneck, and
hybrid cross-attention decoder into a single end-to-end segmentation model.

    Input  →  CNN Encoder  →  ViT Bottleneck  →  Hybrid Decoder  →  Logits
              (skip features)                     (skip fusion)

Usage::

    model = TransUNetRS(num_classes=10, img_size=256)
    logits = model(images)  # [B, num_classes, 256, 256]
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import yaml

from .cnn_encoder import ResNet50Encoder
from .transformer import TransformerBottleneck
from .decoder import HybridDecoder


class TransUNetRS(nn.Module):
    """Hybrid CNN-Transformer architecture for LULC segmentation.

    Parameters
    ----------
    num_classes : int
        Number of land-cover classes.
    img_size : int
        Input image spatial dimension (assumed square).
    in_channels : int
        Number of input channels (3 for RGB).
    encoder_pretrained : bool
        Use ImageNet-pretrained ResNet-50 weights.
    freeze_bn : bool
        Freeze BatchNorm in the encoder.
    embed_dim : int
        Transformer embedding dimension.
    num_heads : int
        Number of attention heads in the Transformer.
    num_layers : int
        Number of Transformer encoder layers.
    mlp_ratio : float
        MLP expansion ratio inside each Transformer block.
    transformer_dropout : float
        Dropout for Transformer MLP and projection.
    attention_dropout : float
        Dropout for attention weights.
    decoder_channels : list[int] | None
        Channel progression in the decoder.
    skip_channels : list[int] | None
        Skip-connection channel sizes for each decoder stage.
    use_cross_attention : bool
        Fuse skips with cross-attention (True) or concat (False).
    cross_attention_heads : int
        Heads per cross-attention block in the decoder.
    decoder_dropout : float
        Dropout in decoder blocks.
    """

    def __init__(
        self,
        num_classes: int = 10,
        img_size: int = 256,
        in_channels: int = 3,
        encoder_pretrained: bool = True,
        freeze_bn: bool = False,
        # Transformer params
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        transformer_dropout: float = 0.1,
        attention_dropout: float = 0.0,
        # Decoder params
        decoder_channels: Optional[list] = None,
        skip_channels: Optional[list] = None,
        use_cross_attention: bool = True,
        cross_attention_heads: int = 8,
        decoder_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.img_size = img_size
        self.num_classes = num_classes

        # ── CNN Encoder ──────────────────────────────────────────────
        self.encoder = ResNet50Encoder(
            pretrained=encoder_pretrained,
            freeze_bn=freeze_bn,
        )
        encoder_out_ch = self.encoder.get_output_channels()  # 1024
        feature_size = img_size // 16  # 16 for 256×256 input

        # ── Transformer Bottleneck ───────────────────────────────────
        self.transformer = TransformerBottleneck(
            in_channels=encoder_out_ch,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=transformer_dropout,
            attention_dropout=attention_dropout,
            feature_size=feature_size,
        )

        # ── Hybrid Decoder ───────────────────────────────────────────
        self.decoder = HybridDecoder(
            encoder_channels=encoder_out_ch,
            decoder_channels=decoder_channels,
            skip_channels=skip_channels,
            num_classes=num_classes,
            use_cross_attention=use_cross_attention,
            cross_attention_heads=cross_attention_heads,
            dropout=decoder_dropout,
        )

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, C, H, W]  — input satellite image batch.

        Returns
        -------
        Tensor  [B, num_classes, H, W]  — per-pixel class logits.
        """
        # 1. Extract multi-scale features
        bottleneck, skips = self.encoder(x)

        # 2. Enrich bottleneck with global context
        bottleneck = self.transformer(bottleneck)

        # 3. Decode with skip fusions → pixel-wise logits
        logits = self.decoder(bottleneck, skips)

        return logits

    # ------------------------------------------------------------------ #
    #  Factory: build from YAML config
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config_path: str) -> "TransUNetRS":
        """Instantiate the model from a YAML configuration file.

        Parameters
        ----------
        config_path : str
            Path to ``model_config.yaml``.

        Returns
        -------
        TransUNetRS
            Fully constructed model.
        """
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        model_cfg = cfg.get("model", {})
        enc_cfg = cfg.get("encoder", {})
        trans_cfg = cfg.get("transformer", {})
        dec_cfg = cfg.get("decoder", {})

        return cls(
            num_classes=model_cfg.get("num_classes", 10),
            img_size=model_cfg.get("input_size", 256),
            in_channels=model_cfg.get("in_channels", 3),
            encoder_pretrained=enc_cfg.get("pretrained", True),
            freeze_bn=enc_cfg.get("freeze_bn", False),
            embed_dim=trans_cfg.get("embed_dim", 768),
            num_heads=trans_cfg.get("num_heads", 12),
            num_layers=trans_cfg.get("num_layers", 12),
            mlp_ratio=trans_cfg.get("mlp_ratio", 4.0),
            transformer_dropout=trans_cfg.get("dropout", 0.1),
            attention_dropout=trans_cfg.get("attention_dropout", 0.0),
            decoder_channels=dec_cfg.get("channels"),
            skip_channels=dec_cfg.get("skip_channels"),
            use_cross_attention=dec_cfg.get("use_cross_attention", True),
            cross_attention_heads=dec_cfg.get("cross_attention_heads", 8),
            decoder_dropout=dec_cfg.get("dropout", 0.1),
        )

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #
    def count_parameters(self) -> dict:
        """Return parameter counts per sub-module."""
        def _count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        return {
            "encoder": _count(self.encoder),
            "transformer": _count(self.transformer),
            "decoder": _count(self.decoder),
            "total": _count(self),
        }


# ===================================================================== #
#  Quick smoke test
# ===================================================================== #
if __name__ == "__main__":
    model = TransUNetRS(num_classes=10, img_size=256, encoder_pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    logits = model(x)
    print("Output shape:", logits.shape)  # [2, 10, 256, 256]

    counts = model.count_parameters()
    for k, v in counts.items():
        print(f"  {k}: {v:>12,}")
