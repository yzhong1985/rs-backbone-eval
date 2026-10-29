"""Shared lightweight task heads (torch), used by every model worker.

These live in the root package so all backbones are decoded by exactly the
same modules and parameter counts. Only worker environments import this
module; the control plane never loads torch.

- ``LiteSpatialHead``: unified head for single-date spatial tasks (T1/T2/T4).
- ``LinearProbe``: 1x1 convolution, the linear-probe baseline.
- ``UPerNetHead``: shared UPerNet decoder (PPM + FPN + classifier) for the
  iSAID track, GroupNorm by default -- terratorch's ``ConvModule`` hard-codes
  ``nn.BatchNorm2d`` and only exists in the prithvi worker, so the head is
  re-implemented here with torch only (design 2026-10-29, §2.1).
- ``TemporalSegmentationHead`` / ``TemporalClassificationHead`` /
  ``TemporalRegressionHead``: unified lightweight temporal head for
  multi-date tasks (T3/T5/T6/T7), following the evaluation design 6.3:
  a shared 1x1 projection to 256 dims per date, real date encodings plus a
  temporal mask, then a 2-layer / 4-head temporal transformer running along
  the time axis at every spatial position.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn


class LiteSpatialHead(nn.Module):
    """[B,C,h,w] -> [B,K,h,w]: 1x1 proj, two residual 3x3 blocks, pixel logits."""

    def __init__(self, in_channels: int, num_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GroupNorm(min(32, hidden), hidden),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, hidden), hidden),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, hidden), hidden),
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(hidden, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.project(features)
        x = x + self.block2(self.block1(x))
        x = self.activation(x)
        x = self.dropout(x)
        return self.classifier(x)


class LinearProbe(nn.Module):
    """[B,C,h,w] -> [B,K,h,w]: a single 1x1 convolution."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def _group_count(channels: int, requested: int = 32) -> int:
    """Largest divisor of ``channels`` up to ``requested`` (GroupNorm needs it)."""
    for groups in range(min(requested, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def default_norm(channels: int) -> nn.Module:
    """GroupNorm, the iSAID track's declared normalisation (design §2.2)."""
    return nn.GroupNorm(_group_count(channels), channels)


class PyramidPoolingModule(nn.Module):
    """UPerNet PPM: pool at several scales, project, and concatenate."""

    def __init__(
        self,
        in_channels: int,
        pool_bins: Sequence[int] = (1, 2, 3, 6),
        norm_layer: Callable[[int], nn.Module] = default_norm,
    ):
        super().__init__()
        self.bins = tuple(pool_bins)
        pool_channels = max(in_channels // len(self.bins), 1)
        self.branches = nn.ModuleList(
            nn.Sequential(nn.AdaptiveAvgPool2d(bin_size), nn.Conv2d(in_channels, pool_channels, 1))
            for bin_size in self.bins
        )
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + pool_channels * len(self.bins), in_channels, 3, padding=1),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        size = features.shape[-2:]
        pooled = [
            F.interpolate(branch(features), size=size, mode="bilinear", align_corners=False)
            for branch in self.branches
        ]
        return self.project(torch.cat([features, *pooled], dim=1))


class UPerNetHead(nn.Module):
    """Multi-level ``[B,Ci,Hi,Wi]`` -> ``[B,K,Hi,Wi]`` logits (PPM + FPN).

    ``features`` is a pyramid ordered finest-first (levels are resampled to the
    first level's resolution, so any ``N >= 1`` works; a single stride-16 map
    still decodes, it is just coarser). ``norm_layer`` defaults to GroupNorm --
    terratorch's ``ConvModule`` hard-codes BatchNorm and is only installed in
    the prithvi worker, so this head is torch-only by construction.
    """

    def __init__(
        self,
        in_channels: int | Sequence[int],
        num_classes: int,
        fpn_dim: int = 256,
        pool_bins: Sequence[int] = (1, 2, 3, 6),
        norm_layer: Callable[[int], nn.Module] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        channels = [in_channels] if isinstance(in_channels, int) else list(in_channels)
        if not channels:
            raise ValueError("UPerNetHead needs at least one input level")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.in_channels = channels
        norm = norm_layer or default_norm
        self.lateral = nn.ModuleList(nn.Conv2d(width, fpn_dim, 1) for width in channels)
        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1),
                norm(fpn_dim),
                nn.ReLU(inplace=True),
            )
            for _ in channels
        )
        self.ppm = PyramidPoolingModule(fpn_dim, pool_bins, norm)
        self.fuse = nn.Sequential(
            nn.Conv2d(fpn_dim * len(channels), fpn_dim, 3, padding=1),
            norm(fpn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(fpn_dim, num_classes, 1)

    def forward(
        self,
        features: Sequence[torch.Tensor],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if len(features) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} levels, got {len(features)}")
        laterals = [conv(level) for conv, level in zip(self.lateral, features, strict=True)]
        # top-down path: coarse context is added into the finer levels
        for index in range(len(laterals) - 2, -1, -1):
            laterals[index] = laterals[index] + F.interpolate(
                laterals[index + 1],
                size=laterals[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        laterals[-1] = self.ppm(laterals[-1])
        smoothed = [block(level) for block, level in zip(self.smooth, laterals, strict=True)]
        size = smoothed[0].shape[-2:]
        merged = [
            level
            if level.shape[-2:] == size
            else F.interpolate(level, size=size, mode="bilinear", align_corners=False)
            for level in smoothed
        ]
        logits = self.classifier(self.fuse(torch.cat(merged, dim=1)))
        if output_size is not None and tuple(logits.shape[-2:]) != tuple(output_size):
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits


def date_encoding(dates: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal encoding of day-of-year dates: [B,T] -> [B,T,dim]."""
    positions = dates.float().unsqueeze(-1)  # [B,T,1]
    half = dim // 2
    frequencies = torch.exp(
        torch.arange(half, device=dates.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = positions * frequencies  # [B,T,half]
    encoding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if encoding.shape[-1] < dim:  # odd dim: pad one zero column
        encoding = torch.nn.functional.pad(encoding, (0, dim - encoding.shape[-1]))
    return encoding


class TemporalSegmentationHead(nn.Module):
    """[B,T,C,h,w] -> [B,K,H,W]: shared per-date projection, temporal transformer,
    then a small convolutional decoder."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.project = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.date_dim = 16
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden + self.date_dim,
            nhead=heads,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        # project back to the hidden width after the date encoding is consumed
        self.post = nn.Conv2d(hidden + self.date_dim, hidden, kernel_size=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, hidden), hidden),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, hidden), hidden),
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(hidden, num_classes, kernel_size=1)

    def _fuse(
        self, features: torch.Tensor, dates: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """[B,T,C,h,w] -> [B,hidden,h,w] with masked temporal attention."""
        batch, time, _, height, width = features.shape
        x = self.project(features.reshape(batch * time, -1, height, width))
        x = x.reshape(batch, time, -1, height, width).permute(0, 3, 4, 1, 2)  # [B,h,w,T,C]
        x = x.reshape(batch * height * width, time, -1)
        encoding = date_encoding(dates, self.date_dim).unsqueeze(1)  # [B,1,T,d]
        encoding = encoding.expand(batch, height * width, time, self.date_dim)
        encoding = encoding.reshape(batch * height * width, time, self.date_dim)
        x = torch.cat([x, encoding], dim=-1)
        padding_mask = ~mask.bool().unsqueeze(1).expand(batch, height * width, time)
        padding_mask = padding_mask.reshape(batch * height * width, time)
        # A fully masked row would make attention softmax over an empty set
        # (NaN); let position 0 attend and rely on the mask multiply below.
        fully_masked = padding_mask.all(dim=-1, keepdim=True)
        first = torch.zeros_like(padding_mask)
        first[:, 0] = True
        padding_mask = padding_mask & ~(fully_masked & first)
        attended = self.temporal(x, src_key_padding_mask=padding_mask)
        attended = attended.reshape(batch, height, width, time, -1).permute(0, 3, 4, 1, 2)
        attended = attended * mask.view(batch, time, 1, 1, 1)
        counts = mask.sum(dim=1).clamp(min=1).view(batch, 1, 1, 1)
        fused = attended.sum(dim=1) / counts
        return self.post(fused)

    def forward(
        self,
        features: torch.Tensor,
        dates: torch.Tensor,
        mask: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        x = self._fuse(features, dates, mask)
        x = x + self.block2(self.block1(x))
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        if output_size is not None:
            logits = torch.nn.functional.interpolate(
                logits, size=output_size, mode="bilinear", align_corners=False
            )
        return logits


class TemporalClassificationHead(nn.Module):
    """[B,T,C,h,w] -> [B,K]: temporal fusion, spatial pooling, linear classifier."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fuse = TemporalSegmentationHead(
            in_channels, num_classes, hidden, layers, heads, dropout
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(hidden, num_classes)

    def forward(
        self, features: torch.Tensor, dates: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        fused = self.fuse._fuse(features, dates, mask)
        return self.classifier(self.pool(fused).flatten(1))


class TemporalRegressionHead(nn.Module):
    """[B,T,C,h,w] -> [B,1,H,W]: temporal fusion then a residual decoder."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 256,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fuse = TemporalSegmentationHead(in_channels, 1, hidden, layers, heads, dropout)

    def forward(
        self,
        features: torch.Tensor,
        dates: torch.Tensor,
        mask: torch.Tensor,
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        return self.fuse(features, dates, mask, output_size)


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
