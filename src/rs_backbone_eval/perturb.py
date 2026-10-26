"""Input perturbations for measuring backbone flexibility (dimension D2).

One training run is evaluated several times with different input perturbations.
This measures how well a frozen backbone tolerates changed inputs at inference
time; it is deliberately *not* re-adaptation, and every run records that in its
tags.

Dropped bands are replaced with the train-split mean, never zero padding (the
design forbids zero padding, and a mean value keeps the input distribution
unbiased). Resolution is emulated by downsampling and resizing back, which is
what a coarser GSD looks like to a patch-based encoder.

Metadata perturbations leave the pixels alone and corrupt what the model is
*told* about them, because customer data often lacks or misstates metadata:
``meta_date:const`` replaces every acquisition date with one placeholder
(an unknown date), ``meta_gsd:<factor>`` declares a GSD ``factor`` times the
true one. They only apply to a model that consumes that metadata (Clay reads
GSD and wavelengths; AnySat dates; OlmoEarth dates and GSD; Prithvi on these
paths reads none); a worker skips the ones that do not apply and records its
metadata inputs in the run tags instead.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NONE = "none"
BAND_DROP = "band_drop"
RESOLUTION = "resolution"
META_DATE = "meta_date"
META_GSD = "meta_gsd"
METADATA_KINDS = (META_DATE, META_GSD)
# Date given to every sample under ``meta_date:const`` (mid-year, so the
# placeholder is not systematically a winter or summer date).
PLACEHOLDER_DATE = "2020-07-01"


# The standard D2 set (docs/evaluation/骨干能力评分卡与扰动协议.md): keep only visible
# light, drop SWIR, and emulate coarser ground sampling distances.
DEFAULT_PERTURBATIONS = [
    "band_drop:NIR_NARROW,SWIR_1,SWIR_2",
    "band_drop:SWIR_1,SWIR_2",
    "resolution:2",
    "resolution:4",
]

# Metadata robustness set: unknown date, GSD declared 2x too coarse / too fine.
METADATA_PERTURBATIONS = [
    "meta_date:const",
    "meta_gsd:2",
    "meta_gsd:0.5",
]


def perturbations_from_env(default: str = NONE) -> list[str]:
    """Parse ``RS_EVAL_PERTURB`` as a semicolon-separated list of experiments.

    The tokens ``default`` and ``metadata`` expand to
    :data:`DEFAULT_PERTURBATIONS` and :data:`METADATA_PERTURBATIONS` and can be
    combined (``default;metadata``); an empty value means no perturbation.
    Experiments are separated by ``;`` so a band list such as
    ``band_drop:NIR_NARROW,SWIR_1`` stays a single experiment (commas separate
    the bands inside one experiment).
    """
    raw = os.environ.get("RS_EVAL_PERTURB", default)
    expanded: list[str] = []
    for item in (part.strip() for part in raw.split(";")):
        if not item:
            continue
        if item.lower() == "default":
            expanded.extend(DEFAULT_PERTURBATIONS)
        elif item.lower() == "metadata":
            expanded.extend(METADATA_PERTURBATIONS)
        else:
            expanded.append(item)
    return list(dict.fromkeys(expanded))


def resolve(name: str) -> dict[str, object]:
    """Expand ``band_drop:NIR_NARROW,SWIR_1`` / ``resolution:4`` into parameters."""
    head, _, tail = name.partition(":")
    head = head.strip().lower()
    if head == BAND_DROP:
        bands = [item.strip() for item in tail.split(",") if item.strip()]
        if not bands:
            raise ValueError(f"{BAND_DROP} needs at least one band, got {name!r}")
        return {"kind": BAND_DROP, "bands": bands}
    if head == RESOLUTION:
        return {"kind": RESOLUTION, "factor": max(2, int(tail or 2))}
    if head == META_DATE:
        if tail.strip().lower() not in {"", "const"}:
            raise ValueError(f"{META_DATE} supports only 'const', got {name!r}")
        return {"kind": META_DATE, "date": PLACEHOLDER_DATE}
    if head == META_GSD:
        factor = float(tail or 2)
        if factor <= 0 or factor == 1:
            raise ValueError(f"{META_GSD} needs a positive factor other than 1, got {name!r}")
        return {"kind": META_GSD, "factor": factor}
    if head in {NONE, ""}:
        return {"kind": NONE}
    raise ValueError(
        f"unknown perturbation {name!r}; expected {NONE}, {BAND_DROP}, {RESOLUTION}, "
        f"{META_DATE} or {META_GSD}"
    )


def applies(spec: dict[str, object], metadata_inputs: set[str]) -> bool:
    """Whether a resolved perturbation can act on a model with these metadata inputs.

    Pixel perturbations always apply. ``meta_date`` needs ``"date"`` and
    ``meta_gsd`` needs ``"gsd"`` among the metadata the model consumes.
    """
    if spec["kind"] == META_DATE:
        return "date" in metadata_inputs
    if spec["kind"] == META_GSD:
        return "gsd" in metadata_inputs
    return spec["kind"] != NONE


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
    from torch import nn

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
    if kind == META_DATE:
        return f"test_{META_DATE}-const"
    if kind == META_GSD:
        return f"test_{META_GSD}x{spec['factor']:g}"
    return "test"
