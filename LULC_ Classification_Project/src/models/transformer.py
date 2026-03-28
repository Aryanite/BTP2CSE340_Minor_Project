"""
TransUNet-RS — Vision Transformer Bottleneck
=============================================
Takes the 1024-channel, 16×16 feature map from the CNN encoder, projects it
into a sequence of patch tokens, adds learnable positional embeddings, and
processes them through ``num_layers`` standard Transformer encoder blocks
(multi-head self-attention + FFN).

The output is reshaped back to a 2-D feature map for the decoder.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================================================================== #
#  Multi-Head Self-Attention
# ===================================================================== #
class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with optional dropout."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, N, D]   (N = number of tokens, D = embed_dim)

        Returns
        -------
        Tensor  [B, N, D]
        """
        B, N, D = x.shape

        # Compute Q, K, V  →  [B, num_heads, N, head_dim]
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


# ===================================================================== #
#  Transformer MLP (Feed-Forward Network)
# ===================================================================== #
class TransformerMLP(nn.Module):
    """Two-layer MLP with GELU activation used inside each Transformer block."""

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


# ===================================================================== #
#  Transformer Encoder Block
# ===================================================================== #
class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block: LN → MHSA → LN → MLP."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim, num_heads, attention_dropout, dropout
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = TransformerMLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ===================================================================== #
#  Patch Embedding (linear projection of flattened CNN features)
# ===================================================================== #
class PatchEmbedding(nn.Module):
    """Project each spatial location (1×1 patch) of the CNN feature map into
    the Transformer embedding dimension.

    Parameters
    ----------
    in_channels : int
        Channel depth of the incoming CNN feature map (e.g. 1024).
    embed_dim : int
        Transformer embedding dimension (e.g. 768).
    feature_size : int
        Spatial size of the feature map (e.g. 16 for 16×16).
    use_learnable_pos : bool
        If ``True``, use a learnable positional embedding; otherwise use
        fixed sinusoidal encoding.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        embed_dim: int = 768,
        feature_size: int = 16,
        use_learnable_pos: bool = True,
    ) -> None:
        super().__init__()
        self.num_tokens = feature_size * feature_size  # 256
        self.proj = nn.Linear(in_channels, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        if use_learnable_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_tokens, embed_dim)
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_buffer(
                "pos_embed",
                self._sinusoidal_embedding(self.num_tokens, embed_dim),
            )

    # ── sinusoidal fallback ──────────────────────────────────────────
    @staticmethod
    def _sinusoidal_embedding(n_tokens: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(1, n_tokens, dim)
        position = torch.arange(0, n_tokens).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, C, H, W]  (e.g. B×1024×16×16)

        Returns
        -------
        Tensor  [B, N, D]  where N = H*W, D = embed_dim
        """
        B, C, H, W = x.shape
        # Flatten spatial dims and project
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.proj(x)                  # [B, N, D]
        x = self.norm(x)
        x = x + self.pos_embed            # add positional encoding
        return x


# ===================================================================== #
#  Full Transformer Bottleneck
# ===================================================================== #
class TransformerBottleneck(nn.Module):
    """Vision Transformer bottleneck for TransUNet-RS.

    Takes a CNN feature map, embeds it into tokens, runs ``num_layers``
    Transformer encoder blocks, and reshapes the output back to a 2-D
    feature map.

    Parameters
    ----------
    in_channels : int
        Channels of the input CNN feature map (default 1024).
    embed_dim : int
        Transformer hidden dimension (default 768).
    num_heads : int
        Number of attention heads (default 12).
    num_layers : int
        Number of Transformer encoder layers (default 12).
    mlp_ratio : float
        Expansion ratio for the MLP inside each block (default 4.0).
    dropout : float
        Dropout rate for MLP and projection layers.
    attention_dropout : float
        Dropout rate inside the attention matrix.
    feature_size : int
        Spatial size of the input feature map (default 16).
    """

    def __init__(
        self,
        in_channels: int = 1024,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        feature_size: int = 16,
        use_learnable_pos: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.feature_size = feature_size

        # Patch embedding (1×1 patches on 16×16 feature map)
        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            feature_size=feature_size,
            use_learnable_pos=use_learnable_pos,
        )

        # Stack of Transformer encoder blocks
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final layer-norm
        self.norm = nn.LayerNorm(embed_dim)

        # Project back to CNN channel space for the decoder
        self.proj_out = nn.Linear(embed_dim, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, C, H, W]  (e.g. B×1024×16×16)

        Returns
        -------
        Tensor  [B, C, H, W]  — same spatial size, enriched with global context
        """
        B, C, H, W = x.shape

        # Embed → Transformer → Norm
        tokens = self.patch_embed(x)   # [B, N, D]
        tokens = self.blocks(tokens)   # [B, N, D]
        tokens = self.norm(tokens)     # [B, N, D]

        # Project back to original channel dim and reshape to 2-D
        tokens = self.proj_out(tokens)  # [B, N, C]
        out = tokens.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]

        return out


# ===================================================================== #
#  Quick smoke test
# ===================================================================== #
if __name__ == "__main__":
    vit = TransformerBottleneck(
        in_channels=1024,
        embed_dim=768,
        num_heads=12,
        num_layers=12,
        feature_size=16,
    )
    dummy = torch.randn(2, 1024, 16, 16)
    out = vit(dummy)
    print("Transformer output:", out.shape)  # [2, 1024, 16, 16]
    print(f"Parameters: {sum(p.numel() for p in vit.parameters()):,}")
