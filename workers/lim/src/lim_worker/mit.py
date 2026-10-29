"""MiT-Huge encoder for LIM, ported from the colleague's MixMIM checkout.

Source: ``rs_segmentation/upstream/selfsup/models/backbones/mixmim_mit_backbone.py``
and ``models_mit.py::MitVisionTransformerForMixMIM`` (encoder half only). The
port keeps the parameter names of ``mit_huge_600.pth`` (``patch_embed1.proj.*``,
``block3.7.attn.kv.*``, ``absolute_pos_embed``) so the archive loads without a
rename table, and drops the masked-pretraining path (``attn_mask``, ``mask_token``,
``decoder_*``) -- inference and fine-tuning never use it.

Two things that matter for 1024-px iSAID tiles:

* ``absolute_pos_embed`` is a single 56x56 (224/4) grid; a 1024 tile produces
  256x256 tokens, so it is bicubically interpolated at forward time;
* stage 1 uses stride 4, which is the reason LIM keeps 3-7 feature cells on a
  20-px car while the /16 backbones keep 1-2 (design 2026-10-29 §2.4).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def sincos_2d(embed_dim: int, grid_size: int) -> np.ndarray:
    """2-D sin-cos position grid, as used to initialise ``absolute_pos_embed``.

    Only matters for random-init runs; the pretrained checkpoint replaces it.
    Follows the MAE helper the MixMIM codebase uses.
    """
    if embed_dim % 2:
        raise ValueError(f"embed_dim {embed_dim} must be even")
    grid_w, grid_h = np.meshgrid(
        np.arange(grid_size, dtype=np.float32), np.arange(grid_size, dtype=np.float32)
    )
    half = embed_dim // 2
    omega = 1.0 / 10000 ** (np.arange(half // 2, dtype=np.float64) / (half / 2.0))

    def one_axis(positions: np.ndarray) -> np.ndarray:
        angles = np.einsum("m,d->md", positions.reshape(-1), omega)
        return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)

    return np.concatenate([one_axis(grid_h), one_axis(grid_w)], axis=1)


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep)
    return x.div(keep) * mask


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class DWConv(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, _, channels = x.shape
        x = x.transpose(1, 2).contiguous().view(batch, channels, height, width)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(batch, tokens, self.num_heads, channels // self.num_heads)
        q = q.permute(0, 2, 1, 3).contiguous()
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).contiguous().reshape(batch, channels, height, width)
            x_ = self.sr(x_).reshape(batch, channels, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_)
        else:
            kv = self.kv(x)
        kv = kv.reshape(batch, -1, 2, self.num_heads, channels // self.num_heads)
        kv = kv.permute(2, 0, 3, 1, 4).contiguous()
        key, value = kv[0], kv[1]
        attn = (q @ key.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ value).transpose(1, 2).contiguous().reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio: int = 1,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=1e-6)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        return x + self.drop_path(self.mlp(self.norm2(x), height, width))


class OverlapPatchEmbed(nn.Module):
    """Image -> overlapping patch tokens (SegFormer style)."""

    def __init__(
        self, patch_size: int = 7, stride: int = 4, in_chans: int = 3, embed_dim: int = 768
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        x = self.proj(x)
        _, _, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2).contiguous()
        return self.norm(tokens), height, width


class MixMIMEncoder(nn.Module):
    """MiT-Huge: 4 stages at strides 4/8/16/32, one shared absolute pos embed."""

    def __init__(
        self,
        in_chans: int = 3,
        embed_dims: tuple[int, ...] = (256, 512, 1024, 2048),
        num_heads: tuple[int, ...] = (4, 8, 16, 32),
        depths: tuple[int, ...] = (3, 4, 18, 3),
        sr_ratios: tuple[int, ...] = (8, 4, 2, 1),
        mlp_ratios: tuple[float, ...] = (4, 4, 4, 4),
        drop_path_rate: float = 0.0,
        qkv_bias: bool = True,
        pos_embed_size: int = 56,
        pos_embed_dim: int = 256,
    ):
        super().__init__()
        self.embed_dims = tuple(embed_dims)
        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_chans, embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])

        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cursor = 0
        stages = []
        for index in range(4):
            stages.append(
                nn.ModuleList(
                    Block(
                        dim=embed_dims[index],
                        num_heads=num_heads[index],
                        mlp_ratio=mlp_ratios[index],
                        qkv_bias=qkv_bias,
                        drop_path_rate=rates[cursor + offset],
                        sr_ratio=sr_ratios[index],
                    )
                    for offset in range(depths[index])
                )
            )
            cursor += depths[index]
        self.block1, self.block2, self.block3, self.block4 = stages
        self.norm1 = nn.LayerNorm(embed_dims[0], eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dims[1], eps=1e-6)
        self.norm3 = nn.LayerNorm(embed_dims[2], eps=1e-6)
        self.norm4 = nn.LayerNorm(embed_dims[3], eps=1e-6)
        # MixMIM's sinusoidal grid; loaded from the checkpoint, interpolated per
        # input size (224 -> 1024 is a 56 -> 256 grid).
        self.absolute_pos_embed = nn.Parameter(
            torch.zeros(1, pos_embed_size * pos_embed_size, pos_embed_dim),
            requires_grad=False,
        )
        # Activation checkpointing costs ~30% speed but is the only way a
        # 1024-px full fine-tune fits: stage 1 materialises a 65k-token
        # attention map. Off by default, enabled per run.
        self.use_checkpoint = False
        self.absolute_pos_embed.data.copy_(
            torch.from_numpy(sincos_2d(pos_embed_dim, pos_embed_size)).float().unsqueeze(0)
        )

    def _pos_embed(self, height: int, width: int) -> torch.Tensor:
        base = self.absolute_pos_embed
        side = int(math.isqrt(base.shape[1]))
        grid = base.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        if (side, side) != (height, width):
            grid = F.interpolate(grid, size=(height, width), mode="bicubic", align_corners=False)
        return grid.flatten(2).transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        batch = x.shape[0]
        use_checkpoint = self.use_checkpoint and self.training
        outputs: list[torch.Tensor] = []
        for index, (embed, blocks, norm) in enumerate(
            (
                (self.patch_embed1, self.block1, self.norm1),
                (self.patch_embed2, self.block2, self.norm2),
                (self.patch_embed3, self.block3, self.norm3),
                (self.patch_embed4, self.block4, self.norm4),
            ),
            start=1,
        ):
            tokens, height, width = embed(x)
            if index == 1:
                tokens = tokens + self._pos_embed(height, width)
            for block in blocks:
                if use_checkpoint:
                    tokens = torch.utils.checkpoint.checkpoint(
                        block, tokens, height, width, use_reentrant=False
                    )
                else:
                    tokens = block(tokens, height, width)
            tokens = norm(tokens)
            x = tokens.reshape(batch, height, width, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)
        return outputs


def mit_huge(in_chans: int = 3, drop_path_rate: float = 0.0) -> MixMIMEncoder:
    return MixMIMEncoder(in_chans=in_chans, drop_path_rate=drop_path_rate)


def load_encoder(path: Path, encoder: MixMIMEncoder) -> dict[str, int]:
    """Load the encoder half of a MixMIM checkpoint; return key counts.

    ``mit_huge_600.pth`` stores ``{'model': state_dict, ...}`` where the state
    dict also holds the reconstruction decoder (27.9 M parameters) and the mask
    token. Only the encoder keys are kept, and every one of them must match --
    a partially loaded 524 M encoder would silently score worse.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    state = (
        state
        or (checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None)
        or checkpoint
    )
    encoder_state = {
        key: value
        for key, value in state.items()
        if not key.startswith("decoder") and key != "mask_token"
    }
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    dropped = sorted(key for key in state if key not in encoder_state)
    return {
        "loaded": len(encoder_state) - len(unexpected),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "dropped": len(dropped),
    }
