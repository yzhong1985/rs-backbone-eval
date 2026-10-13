"""Explicit six-band spectral adapter for AnySat's fixed ten-band S2 entry.

AnySat's public ``s2`` entrance always consumes ten bands
(B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12). The T1 official protocol supplies six
bands (BLUE,GREEN,RED,NIR_NARROW,SWIR_1,SWIR_2). Padding the missing red-edge
and broad-NIR bands with zeros is forbidden by the evaluation design, so this
module learns the four missing bands explicitly. Its parameter count and
training state are reported separately from the task head.
"""

from __future__ import annotations

import torch
from torch import nn

# AnySat's fixed S2 band order.
ANYSAT_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Task band name -> index into the 13-band Sen1Floods11 L1C chip.
CHIP_BAND_INDICES = {
    "COASTAL_AEROSOL": 0,
    "BLUE": 1,
    "GREEN": 2,
    "RED": 3,
    "RED_EDGE_1": 4,
    "RED_EDGE_2": 5,
    "RED_EDGE_3": 6,
    "NIR_BROAD": 7,
    "NIR_NARROW": 8,
    "WATER_VAPOR": 9,
    "CIRRUS": 10,
    "SWIR_1": 11,
    "SWIR_2": 12,
}

# Which AnySat input band each task band directly provides.
TASK_BAND_TO_ANYSAT = {
    "BLUE": "B2",
    "GREEN": "B3",
    "RED": "B4",
    "NIR_NARROW": "B8A",
    "SWIR_1": "B11",
    "SWIR_2": "B12",
}


def chip_indices_for_bands(bands: list[str]) -> list[int]:
    return [CHIP_BAND_INDICES[band] for band in bands]


def present_anysat_positions(bands: list[str]) -> list[int]:
    """Positions in AnySat's ten-band tensor that the task bands provide."""
    names = [TASK_BAND_TO_ANYSAT[band] for band in bands]
    return [ANYSAT_S2_BANDS.index(name) for name in names]


class SpectralAdapter(nn.Module):
    """Per-pixel 1x1-conv MLP mapping six normalized bands to AnySat's tens.

    The output lives in the same normalized reflectance space as AnySat's S2
    entrance, so the frozen (or fine-tuned) backbone can consume it directly.
    Present bands are passed through a residual skip to preserve their values;
    the MLP only has to synthesize the four missing red-edge/NIR bands.
    """

    def __init__(self, task_bands: list[str], hidden: int = 32):
        super().__init__()
        self.task_bands = list(task_bands)
        self.present_positions = present_anysat_positions(task_bands)
        self.n_inputs = len(task_bands)
        self.n_outputs = len(ANYSAT_S2_BANDS)
        self.mlp = nn.Sequential(
            nn.Conv2d(self.n_inputs, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, self.n_outputs, kernel_size=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """[B,T,C6,H,W] -> [B,T,10,H,W]."""
        batch, time, channels, height, width = images.shape
        if channels != self.n_inputs:
            raise ValueError(f"expected {self.n_inputs} input bands, got {channels}")
        flat = images.reshape(batch * time, channels, height, width)
        output = self.mlp(flat)
        # Residual copy of the present bands so they are not distorted.
        for source, target in enumerate(self.present_positions):
            output[:, target : target + 1] = flat[:, source : source + 1]
        return output.reshape(batch, time, self.n_outputs, height, width)
