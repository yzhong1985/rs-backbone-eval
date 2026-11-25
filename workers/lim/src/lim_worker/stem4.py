"""Four-band first layer for the frozen FT-B0 LIM (T13 plan §4, frozen version).

The RGB convolution of ``patch_embed1`` and its token LayerNorm stay frozen. A
parallel single-band convolution reads the NIR band with the same kernel, stride
and padding. Its output is added to the RGB convolution before the LayerNorm. The
NIR weights start at zero, so the stem reproduces the RGB model at step 0, and
gradients reach only the NIR convolution.
"""

from __future__ import annotations

import torch
from torch import nn

from .mit import OverlapPatchEmbed


class FourBandPatchEmbed(nn.Module):
    """``patch_embed1`` that accepts ``[B, 4, H, W]`` (R, G, B, NIR)."""

    def __init__(self, base: OverlapPatchEmbed):
        super().__init__()
        self.base = base
        proj = base.proj
        self.nir = nn.Conv2d(
            1,
            proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
            bias=False,
        )
        nn.init.zeros_(self.nir.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        y = self.base.proj(x[:, :3]) + self.nir(x[:, 3:4])
        _, _, height, width = y.shape
        tokens = y.flatten(2).transpose(1, 2).contiguous()
        return self.base.norm(tokens), height, width


def add_nir_stem(encoder: nn.Module) -> FourBandPatchEmbed:
    """Replace ``encoder.patch_embed1`` in place and return the new module."""
    stem = FourBandPatchEmbed(encoder.patch_embed1)
    encoder.patch_embed1 = stem
    return stem
