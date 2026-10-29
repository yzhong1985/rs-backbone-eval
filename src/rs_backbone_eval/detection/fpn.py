"""Feature pyramid for the detection head.

The segmentation track uses ``heads.UPerNetHead``, which bundles PPM + FPN; the
detector needs only the top-down pyramid, at a fixed 256 channels, on top of
whatever multi-level features a backbone exposes. Keeping it separate means a
backbone that has native /4../32 levels (LIM, MaRS) and one that has to derive
them (Clay, Galileo) feed the *same* pyramid, so the detector stays the fixed
variable (FAIR1M plan §6.1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ..heads import default_norm


class FPN(nn.Module):
    """Top-down FPN with 1x1 laterals, 3x3 smoothing and 2x nearest upsampling."""

    def __init__(self, in_channels: list[int], out_channels: int = 256) -> None:
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("FPN needs at least two input levels")
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.lateral = nn.ModuleList(
            [nn.Conv2d(channels, out_channels, kernel_size=1) for channels in in_channels]
        )
        self.smooth = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    default_norm(out_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in in_channels
            ]
        )

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(features) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} levels, got {len(features)}")
        laterals = [conv(feature) for conv, feature in zip(self.lateral, features, strict=True)]
        for index in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(
                laterals[index], size=laterals[index - 1].shape[-2:], mode="nearest"
            )
            laterals[index - 1] = laterals[index - 1] + upsampled
        return [smooth(values) for smooth, values in zip(self.smooth, laterals, strict=True)]
