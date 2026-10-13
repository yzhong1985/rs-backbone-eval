"""Readers for band-stacked multi-temporal rasters (worker-only imports).

Some datasets ship one raster whose bands are stacked as ``T x C`` instead of
one file per date (T5 does: 18 bands = 3 dates x 6 HLS bands). TerraTorch's
official loader confirms the layout is **date-major**
(``"(time channels) h w"``), i.e. bands ``[t*C : (t+1)*C]`` hold date ``t``.

Labels follow the official recipe: value ``0`` means "invalid" and is shifted
to the ignore index (``reduce_zero_label``), so classes 1..13 become 0..12.

This module imports rasterio, which only worker environments have; the control
plane never imports it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

# Official per-band statistics for the six HLS bands of T5 (raw reflectance
# scale, no /10000), taken from TerraTorch's MultiTemporalCropClassification
# data module so the normalization matches the reference configuration.
MULTICROP_MEANS = {
    "BLUE": 494.905781,
    "GREEN": 815.239594,
    "RED": 924.335066,
    "NIR_NARROW": 2968.881459,
    "SWIR_1": 2634.621962,
    "SWIR_2": 1739.579917,
}
MULTICROP_STDS = {
    "BLUE": 284.925432,
    "GREEN": 357.84876,
    "RED": 575.566823,
    "NIR_NARROW": 896.601013,
    "SWIR_1": 951.900334,
    "SWIR_2": 921.407808,
}
MULTICROP_IGNORE_INDEX = -1


def read_stacked_dates(path: Path, channels: int = 6) -> np.ndarray:
    """Read a date-major band-stacked raster as ``[T, C, H, W]`` float32."""
    with rasterio.open(path) as source:
        data = source.read().astype(np.float32)
    if data.shape[0] % channels != 0:
        raise ValueError(f"{path.name}: {data.shape[0]} bands are not a multiple of {channels}")
    time_steps = data.shape[0] // channels
    return data.reshape(time_steps, channels, data.shape[1], data.shape[2])


def read_multicrop_label(path: Path, *, reduce_zero_label: bool = True) -> np.ndarray:
    """Read the label and map the invalid class 0 to the ignore index."""
    with rasterio.open(path) as source:
        label = source.read(1).astype(np.int64)
    if reduce_zero_label:
        label = label - 1
    return label


def multicrop_statistics(bands: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean/std in the task's band order."""
    mean = np.asarray([MULTICROP_MEANS[band] for band in bands], dtype=np.float32)
    std = np.asarray([MULTICROP_STDS[band] for band in bands], dtype=np.float32)
    return mean, std
