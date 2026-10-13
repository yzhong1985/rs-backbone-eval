"""Shared lightweight task heads (torch), used by every model worker.

These live in the root package so all three backbones are decoded by exactly
the same modules and parameter counts. Only worker environments import this
module; the control plane never loads torch.

``LiteSpatialHead`` follows the evaluation design's unified lightweight
spatial head for single-date spatial tasks (T1/T2/T4).
"""

from __future__ import annotations

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


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    params = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
