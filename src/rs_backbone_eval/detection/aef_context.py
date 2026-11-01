"""AEF context injection for the detection head (diagnostic, not a protocol).

AlphaEarth Foundations provides a precomputed 64-band embedding field on a 10 m
UTM grid. On a 0.8 m FAIR1M tile it is a coarse "where am I / what is around"
map: useful as *context*, far too coarse to detect the objects themselves.

`AefContext` projects that map to the detector's FPN width and adds it to one
pyramid level. `scripts/aef_fusion_probe.py` uses it three ways:

- ``none``: LIM only (baseline);
- ``real``: the tile's actual AEF window;
- ``shuffled``: the same AEF values, randomly permuted within the tile. Same
  statistics and parameters, no spatial correspondence — the capacity control.
  A gain that survives only in ``real`` is attributable to the context, not to
  the extra 1x1 convolution.

The production detector is untouched: callers pass the already-modified feature
list to ``RotatedFasterRCNN.losses`` / ``.detections``.
"""

from __future__ import annotations

import torch
from torch import nn

from ..heads import default_norm

AEF_CHANNELS = 64
# The /8 FPN level balances the two failure modes: /4 is dominated by hairline
# detail a 10 m field cannot provide, /16 is already coarser than the AEF cell.
DEFAULT_LEVEL = 1


class AefContext(nn.Module):
    """``[B,64,h,w]`` context -> add to one pyramid level of the same size."""

    def __init__(self, channels: int = 256, level: int = DEFAULT_LEVEL):
        super().__init__()
        self.level = level
        self.project = nn.Sequential(
            nn.Conv2d(AEF_CHANNELS, channels, kernel_size=1),
            default_norm(channels),
            nn.GELU(),
        )

    def forward(self, features: list[torch.Tensor], context: torch.Tensor) -> list[torch.Tensor]:
        """Return a new feature list; the input list is not modified in place.

        The context is resampled to the level's grid, so a caller can hand in the
        /8-sized map (as the extractor produces) regardless of the target level.
        """
        if self.level >= len(features):
            raise ValueError(f"level {self.level} out of range for {len(features)} features")
        target = features[self.level]
        if target.shape[1] != self.project[0].out_channels:
            raise ValueError(
                f"level {self.level} has {target.shape[1]} channels but the context head "
                f"projects to {self.project[0].out_channels}"
            )
        if context.shape[-2:] != target.shape[-2:]:
            context = nn.functional.interpolate(
                context, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        injected = target + self.project(context).to(target.dtype)
        return [
            injected if index == self.level else feature for index, feature in enumerate(features)
        ]


def shuffle_within_tile(
    context: torch.Tensor, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Permute each sample's spatial positions; keeps values, destroys layout."""
    batch, channels, height, width = context.shape
    flat = context.reshape(batch, channels, height * width)
    index = torch.randperm(height * width, generator=generator, device=context.device)
    return flat[:, :, index].reshape(batch, channels, height, width)
