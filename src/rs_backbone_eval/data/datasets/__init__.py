"""Dataset-specific manifest builders for officially split downstream data."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path

from ...registry import TaskSpec
from .burn_intensity import build_burn_intensity_manifest
from .five_billion_pixels import build_five_billion_pixels_manifest
from .hls_burn_scars import build_hls_burn_scars_manifest
from .landslide4sense import build_landslide4sense_manifest
from .multicrop import build_multicrop_manifest
from .sen1floods11 import build_sen1floods11_manifest

ManifestBuilder = Callable[[Path, TaskSpec], "list"]  # list of GeoSequenceSample

BUILDERS: dict[str, ManifestBuilder] = {
    "Sen1Floods11": build_sen1floods11_manifest,
    "Sen1Floods11-S1": partial(build_sen1floods11_manifest, layer="S1Hand", date_field="s1_date"),
    "hls_burn_scars": build_hls_burn_scars_manifest,
    "burn_intensity": build_burn_intensity_manifest,
    "Landslide4Sense": build_landslide4sense_manifest,
    "multi-temporal-crop-classification": build_multicrop_manifest,
    "Five-Billion-Pixels": build_five_billion_pixels_manifest,
}


def builder_for(dataset_id: str) -> ManifestBuilder:
    """Return the manifest builder registered for one dataset."""
    try:
        return BUILDERS[dataset_id]
    except KeyError as exc:
        raise ValueError(
            f"no manifest builder for dataset {dataset_id}; known: {', '.join(sorted(BUILDERS))}"
        ) from exc
