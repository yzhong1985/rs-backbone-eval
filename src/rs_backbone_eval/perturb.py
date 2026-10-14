"""Input perturbations for measuring backbone flexibility (dimension D2).

One training run is evaluated several times with different input perturbations.
This measures how well a frozen backbone tolerates changed inputs at inference
time; it is deliberately *not* re-adaptation, and every run records that in its
tags.

Dropped bands are replaced with the train-split mean, never zero padding (the
design forbids zero padding, and a mean value keeps the input distribution
unbiased). Resolution is emulated by downsampling and resizing back, which is
what a coarser GSD looks like to a patch-based encoder.
"""

from __future__ import annotations

import os

import torch
from torch import nn

NONE = "none"
BAND_DROP = "band_drop"
RESOLUTION = "resolution"


# The standard D2 set (docs/骨干能力评分卡与扰动协议.md): keep only visible
# light, drop SWIR, and emulate coarser ground sampling distances.
DEFAULT_PERTURBATIONS = [
    "band_drop:NIR_NARROW,SWIR_1,SWIR_2",
    "band_drop:SWIR_1,SWIR_2",
    "resolution:2",
    "resolution:4",
]


def perturbations_from_env(default: str = NONE) -> list[str]:
    """Parse ``RS_EVAL_PERTURB`` as a comma-separated list.

    ``RS_EVAL_PERTURB=default`` selects :data:`DEFAULT_PERTURBATIONS`; the
    default when unset is no perturbation at all.
    """
    raw = os.environ.get("RS_EVAL_PERTURB", default)
    if raw.strip().lower() == "default":
        return list(DEFAULT_PERTURBATIONS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve(name: str) -> dict[str, object]:
    """Expand ``band_drop:NIR_NARROW,SWIR_1`` / ``resolution:4`` into parameters."""
    head, _, tail = name.partition(":")
    head = head.strip().lower()
    if head == BAND_DROP:
        return {
            "kind": BAND_DROP,
            "bands": [item.strip() for item in tail.split(",") if item.strip()],
        }
    if head == RESOLUTION:
        return {"kind": RESOLUTION, "factor": max(2, int(tail or 2))}
    return {"kind": NONE}


def band_indices_to_drop(all_bands: list[str], bands_to_drop: list[str]) -> list[int]:
    """Positions of the dropped bands inside the model's band order (upper-cased compare)."""
    wanted = {band.upper() for band in bands_to_drop}
    return [index for index, band in enumerate(all_bands) if band.upper() in wanted]


def apply_band_drop(
    images: torch.Tensor, drop_indices: list[int], fill: float = 0.0
) -> torch.Tensor:
    """Replace the selected channels with the train-split mean, on a copy.

    ``images`` is already normalized, so every channel's train mean is ``0.0``
    (the default). Pass another value only if the caller normalizes
    differently; the fill is recorded in the run tags either way.
    """
    if not drop_indices:
        return images
    result = images.clone()
    for index in drop_indices:
        if index < result.shape[1]:
            result[:, index] = fill
    return result


def apply_resolution(images: torch.Tensor, factor: int, mode: str = "area") -> torch.Tensor:
    """Downsample by ``factor`` and resize back, keeping the original shape."""
    if factor <= 1:
        return images
    squeeze_time = images.dim() == 5
    source = images[:, 0] if squeeze_time else images
    height, width = source.shape[-2], source.shape[-1]
    small = nn.functional.interpolate(
        source, size=(height // factor, width // factor), mode=mode, antialias=False
    )
    restored = nn.functional.interpolate(
        small, size=(height, width), mode="bilinear", align_corners=False
    )
    return restored[:, None] if squeeze_time else restored


def describe(name: str) -> str:
    """Short, human-readable label recorded in the run's metric split."""
    spec = resolve(name)
    kind = spec["kind"]
    if kind == BAND_DROP:
        return f"test_{BAND_DROP}-{'+'.join(spec['bands'])}"
    if kind == RESOLUTION:
        return f"test_{RESOLUTION}x{spec['factor']}"
    return "test"
