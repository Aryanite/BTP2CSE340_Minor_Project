"""
TransUNet-RS — Hybrid Decoder with Cross-Attention Fusion
==========================================================
Progressive upsampling decoder that fuses Transformer-enriched bottleneck
features with multi-scale skip connections from the CNN encoder via
cross-attention blocks.

Decoder path (256×256 output, 10 classes):
  bottleneck (1024×16×16)
    ↑ upsample  →  fuse skip3 (512×32×32)  →  512×32×32
    ↑ upsample  →  fuse skip2 (256×64×64)  →  256×64×64
    ↑ upsample  →  fuse skip1  (64×64×64)  →   64×64×64
    ↑ upsample  →  16×256×256
    → 1×1 conv  →  num_classes × 256 × 256
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================== #
#  Cross-Attention Fusion Block
# ===================================================================== #
class CrossAttentionFusion(nn.Module):
    """Cross-attention between decoder features (query) and encoder skip
    features (key/value).  Allows the decoder to selectively attend to
    fine-grained spatial details from the encoder.

    Parameters
    ----------
    decoder_channels : int
        Channel depth of the decoder feature map (query source).
    skip_channels : int
        Channel depth of the encoder skip connection (key/value source).
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout on attention weights and projections.
    """

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        # Project both inputs to a common dimension
        self.d_model = decoder_channels
        self.head_dim = self.d_model // num_heads
        assert self.d_model % num_heads == 0

        self.q_proj = nn.Linear(decoder_channels, self.d_model)
        self.k_proj = nn.Linear(skip_channels, self.d_model)
        self.v_proj = nn.Linear(skip_channels, self.d_model)
        self.out_proj = nn.Linear(self.d_model, decoder_channels)

        self.norm_q = nn.LayerNorm(decoder_channels)
        self.norm_kv = nn.LayerNorm(skip_channels)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.scale = self.head_dim ** -0.5

    def forward(
        self, decoder_feat: torch.Tensor, skip_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        decoder_feat : Tensor  [B, C_dec, H, W]
        skip_feat    : Tensor  [B, C_skip, H, W]  (same spatial size)

        Returns
        -------
        Tensor  [B, C_dec, H, W]  — decoder features enriched with skip info
        """
        B, C_d, H, W = decoder_feat.shape
        N = H * W

        # Flatten to sequences
        q = decoder_feat.flatten(2).transpose(1, 2)  # [B, N, C_d]
        kv = skip_feat.flatten(2).transpose(1, 2)    # [B, N, C_s]

        # Norm + project
        q = self.q_proj(self.norm_q(q))    # [B, N, d_model]
        k = self.k_proj(self.norm_kv(kv))  # [B, N, d_model]
        v = self.v_proj(self.norm_kv(kv))  # [B, N, d_model]

        # Reshape for multi-head attention
        q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, self.d_model)
        out = self.proj_drop(self.out_proj(out))

        # Residual + reshape back to 2-D
        out = out.transpose(1, 2).reshape(B, C_d, H, W) + decoder_feat
        return out


# ===================================================================== #
#  Decoder Upsample Block
# ===================================================================== #
class DecoderBlock(nn.Module):
    """Single decoder stage: upsample → concat/fuse skip → conv → conv.

    Parameters
    ----------
    in_channels : int
        Channels from the previous decoder stage.
    skip_channels : int
        Channels from the corresponding encoder skip connection (0 if no skip).
    out_channels : int
        Output channels of this block.
    use_cross_attention : bool
        If ``True``, fuse skip via cross-attention; otherwise simple concat.
    cross_attention_heads : int
        Number of heads for cross-attention (ignored if not used).
    upsample_mode : str
        ``"bilinear"`` or ``"nearest"``.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_cross_attention: bool = True,
        cross_attention_heads: int = 8,
        upsample_mode: str = "bilinear",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        self.has_skip = skip_channels > 0

        # Optional cross-attention or simple concatenation
        if self.has_skip and use_cross_attention:
            self.cross_attn = CrossAttentionFusion(
                decoder_channels=in_channels,
                skip_channels=skip_channels,
                num_heads=cross_attention_heads,
                dropout=dropout,
            )
            conv_in = in_channels  # cross-attn replaces concat
        elif self.has_skip:
            self.cross_attn = None
            conv_in = in_channels + skip_channels  # concat
        else:
            self.cross_attn = None
            conv_in = in_channels

        # Two 3×3 conv blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(conv_in, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout2d(dropout)

    def forward(
        self,
        x: torch.Tensor,
        skip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : Tensor  [B, C_in, H, W]
        skip : Tensor  [B, C_skip, H', W']  or ``None``

        Returns
        -------
        Tensor  [B, C_out, 2H, 2W]
        """
        # Upsample ×2
        x = F.interpolate(
            x, scale_factor=2, mode=self.upsample_mode, align_corners=False
        )

        # Fuse with skip connection
        if skip is not None and self.has_skip:
            # Ensure spatial dims match (handle rounding)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            if self.cross_attn is not None:
                x = self.cross_attn(x, skip)
            else:
                x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x


# ===================================================================== #
#  Full Hybrid Decoder
# ===================================================================== #
class HybridDecoder(nn.Module):
    """Multi-stage hybrid decoder with progressive upsampling and
    cross-attention skip fusions.

    Default configuration for 256×256 input:
        Stage 0: 1024×16×16  → fuse skip3 (512×32×32)  → 512×32×32
        Stage 1:  512×32×32  → fuse skip2 (256×64×64)  → 256×64×64
        Stage 2:  256×64×64  → fuse skip1  (64×64×64)  →  64×128×128
        Stage 3:   64×128×128 → (no skip)              →  16×256×256

    Parameters
    ----------
    encoder_channels : int
        Bottleneck channel count from the encoder (default 1024).
    decoder_channels : list[int]
        Output channels for each decoder stage.
    skip_channels : list[int]
        Channel count for each skip connection (in order from deepest to
        shallowest).  Use ``0`` for stages with no skip.
    num_classes : int
        Number of output segmentation classes.
    use_cross_attention : bool
        Use cross-attention fusion (True) or simple concat (False).
    cross_attention_heads : int
        Number of heads per cross-attention block.
    upsample_mode : str
        Interpolation mode for upsampling.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        encoder_channels: int = 1024,
        decoder_channels: Optional[List[int]] = None,
        skip_channels: Optional[List[int]] = None,
        num_classes: int = 10,
        use_cross_attention: bool = True,
        cross_attention_heads: int = 8,
        upsample_mode: str = "bilinear",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if decoder_channels is None:
            decoder_channels = [512, 256, 64, 16]
        if skip_channels is None:
            skip_channels = [512, 256, 64, 0]  # last stage has no skip

        assert len(decoder_channels) == len(skip_channels)

        blocks: List[DecoderBlock] = []
        in_ch = encoder_channels
        for out_ch, s_ch in zip(decoder_channels, skip_channels):
            blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=s_ch,
                    out_channels=out_ch,
                    use_cross_attention=use_cross_attention,
                    cross_attention_heads=cross_attention_heads,
                    upsample_mode=upsample_mode,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.blocks = nn.ModuleList(blocks)

        # Final 1×1 classification head
        self.seg_head = nn.Conv2d(decoder_channels[-1], num_classes, 1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        bottleneck : Tensor  [B, C_enc, H, W]
        skips : dict
            ``"skip3"`` → [B, 512, 32, 32]
            ``"skip2"`` → [B, 256, 64, 64]
            ``"skip1"`` → [B,  64, 64, 64]

        Returns
        -------
        Tensor  [B, num_classes, H_out, W_out]
        """
        # Map skips to an ordered list (deepest → shallowest, then None)
        skip_list = [
            skips.get("skip3"),
            skips.get("skip2"),
            skips.get("skip1"),
            None,  # last stage has no skip
        ]

        x = bottleneck
        for block, skip in zip(self.blocks, skip_list):
            x = block(x, skip)

        logits = self.seg_head(x)
        return logits


# ===================================================================== #
#  Quick smoke test
# ===================================================================== #
if __name__ == "__main__":
    decoder = HybridDecoder(encoder_channels=1024, num_classes=10)

    bottleneck = torch.randn(2, 1024, 16, 16)
    skips = {
        "skip1": torch.randn(2, 64, 64, 64),
        "skip2": torch.randn(2, 256, 64, 64),
        "skip3": torch.randn(2, 512, 32, 32),
    }
    out = decoder(bottleneck, skips)
    print("Decoder output:", out.shape)  # [2, 10, 256, 256]
