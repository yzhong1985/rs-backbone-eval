"""Head architectures for the frozen-AEF T11 track (shared with the probe CLI).

The linear/mlp/conv/upernet architectures are the same four declared in the
probe stage (``scripts/t11_aef_probe.py``); the registry tracks register mlp
and upernet as the formal heads and the worker keeps all four for parity.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CLASSES = 20
IGNORE = 19
FEATURES_AEF = 64


class LinearHead(nn.Module):
    def __init__(self, in_ch: int = FEATURES_AEF, classes: int = CLASSES) -> None:
        super().__init__()
        self.net = nn.Conv2d(in_ch, classes, 1)

    def forward(self, x):
        return self.net(x)


class MLPHead(nn.Module):
    def __init__(
        self, in_ch: int = FEATURES_AEF, hidden: int = 128, classes: int = CLASSES
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, classes, 1),
        )

    def forward(self, x):
        return self.net(x)


class ContextHead(nn.Module):
    def __init__(self, in_ch: int = FEATURES_AEF, width: int = 64, classes: int = CLASSES) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_ch
        for dilation in (1, 2, 4):
            layers.extend(
                (
                    nn.Conv2d(channels, width, 3, padding=dilation, dilation=dilation, bias=False),
                    nn.GroupNorm(8, width),
                    nn.GELU(),
                )
            )
            channels = width
        self.net = nn.Sequential(*layers, nn.Conv2d(width, classes, 1))

    def forward(self, x):
        return self.net(x)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class UPerNetHead(nn.Module):
    """Small UPerNet over a synthetic 1/2/4/8 pyramid from the AEF grid.

    AEF delivers a single 10 m map, so the pyramid is constructed; capacity
    matches the probe stage (PPM + two-level FPN over width 128).
    """

    def __init__(self, in_ch: int = FEATURES_AEF, width: int = 128, classes: int = CLASSES) -> None:
        super().__init__()
        self.stem = _ConvBlock(in_ch, width)
        self.down2 = _ConvBlock(width, width * 2, 2)
        self.down3 = _ConvBlock(width * 2, width * 2, 2)
        self.down4 = _ConvBlock(width * 2, width * 2, 2)
        self.ppm = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(width * 2, width, 1, bias=False),
                    nn.GroupNorm(8, width),
                    nn.GELU(),
                )
                for size in (1, 2, 3, 6)
            ]
        )
        self.ppm_fuse = nn.Sequential(
            nn.Conv2d(width * 6, width * 2, 1, bias=False),
            nn.GroupNorm(8, width * 2),
            nn.GELU(),
        )
        self.lateral = nn.ModuleList(
            [nn.Conv2d(ch, width, 1) for ch in (width, width * 2, width * 2, width * 2)]
        )
        self.smooth = nn.ModuleList([_ConvBlock(width, width) for _ in range(4)])
        self.classifier = nn.Sequential(
            _ConvBlock(width * 4, width * 2), nn.Conv2d(width * 2, classes, 1)
        )

    def forward(self, x):
        size = x.shape[-2:]
        levels = [self.stem(x)]
        for module in (self.down2, self.down3, self.down4):
            levels.append(module(levels[-1]))
        last = levels[-1]
        pooled = [module(last) for module in self.ppm]
        pooled = [
            F.interpolate(item, size=last.shape[-2:], mode="bilinear", align_corners=False)
            for item in pooled
        ]
        fused = self.ppm_fuse(torch.cat([last, *pooled], dim=1))
        levels[-1] = fused
        laterals = [module(level) for module, level in zip(self.lateral, levels, strict=True)]
        for index in range(len(laterals) - 2, -1, -1):
            laterals[index] = laterals[index] + F.interpolate(
                laterals[index + 1],
                size=laterals[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        merged = [module(item) for module, item in zip(self.smooth, laterals, strict=True)]
        merged = [
            F.interpolate(item, size=size, mode="bilinear", align_corners=False) for item in merged
        ]
        return self.classifier(torch.cat(merged, dim=1))


HEADS = {"linear": LinearHead, "mlp": MLPHead, "conv": ContextHead, "upernet": UPerNetHead}


class IoUEvaluator:
    """Online confusion over CLASSES with the declared ignore index."""

    def __init__(self) -> None:
        self.confusion = np.zeros((CLASSES, CLASSES), dtype=np.int64)

    def add(self, targets, predicted) -> None:
        mask = targets != IGNORE
        flat = (targets[mask] * CLASSES + predicted[mask]).numpy()
        self.confusion += np.bincount(flat, minlength=CLASSES * CLASSES).reshape(CLASSES, CLASSES)

    def value(self) -> dict[str, float]:
        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp
        present = self.confusion.sum(axis=1) > 0
        iou = np.where(present, tp / np.maximum(tp + fp + fn, 1), np.nan)
        foreground = [float(iou[i]) for i in range(1, IGNORE) if present[i]]
        return {
            "miou": float(np.nanmean(iou)),
            "foreground": float(np.mean(foreground)) if foreground else float("nan"),
        }
