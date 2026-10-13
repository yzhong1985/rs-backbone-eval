"""Shared lightweight task heads (torch), used by every model worker.

These live in the root package so all backbones are decoded by exactly the
same modules and parameter counts. Only worker environments import this
module; the control plane never loads torch.

- ``LiteSpatialHead``: unified head for single-date spatial tasks (T1/T2/T4).
- ``LinearProbe``: 1x1 convolution, the linear-probe baseline.
- ``TemporalSegmentationHead`` / ``TemporalClassificationHead`` /
  ``TemporalRegressionHead``: unified lightweight temporal head for
  multi-date tasks (T3/T5/T6/T7), following the evaluation design 6.3:
  a shared 1x1 projection to 256 dims per date, real date encodings plus a
  temporal mask, then a 2-layer / 4-head temporal transformer running along
  the time axis at every spatial position.
"""

from __future__ import annotations

import math

import torch
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
