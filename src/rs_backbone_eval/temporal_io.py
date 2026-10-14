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

# Official per-date statistics for T3 (burn intensity), taken from TerraTorch's
# BurnIntensityNonGeoDataModule. Each band carries three values, one per time
# step in the official order (pre, during, post). The six HLS bands are stored
# as int16 reflectance (raw scale, no /10000), matching these magnitudes.
BURN_INTENSITY_MEANS = {
    "BLUE": [331.6921, 896.8024, 348.8031],
    "GREEN": [555.1077, 1093.9736, 500.2181],
    "RED": [605.2513, 1142.7225, 597.9034],
    "NIR": [1761.3884, 1890.2156, 1552.0403],
    "SWIR_1": [1117.1825, 1408.0839, 1293.0919],
    "SWIR_2": [2168.0090, 2270.9753, 1362.1312],
}
BURN_INTENSITY_STDS = {
    "BLUE": [213.0656, 1620.4131, 314.7517],
    "GREEN": [273.0910, 1628.4181, 365.6746],
    "RED": [414.8322, 1600.7698, 424.8185],
    "NIR": [818.7486, 1236.8453, 804.9058],
    "SWIR_1": [677.2739, 1153.7432, 795.4156],
    "SWIR_2": [612.9131, 1495.8365, 661.6196],
}


def read_hls_dates(paths: list[Path], channels: int = 6) -> np.ndarray:
    """Read one six-band HLS raster per date as ``[T, C, H, W]`` float32."""
    frames = []
    for path in paths:
        with rasterio.open(path) as source:
            data = source.read().astype(np.float32)
        if data.shape[0] != channels:
            raise ValueError(f"{path.name}: {data.shape[0]} bands, expected {channels}")
        frames.append(data)
    return np.stack(frames, axis=0)


def read_burn_intensity_label(path: Path) -> np.ndarray:
    """Read the burn intensity label; classes 0..4 are all valid (no remap)."""
    with rasterio.open(path) as source:
        return source.read(1).astype(np.int64)


def burn_intensity_statistics(bands: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-date, per-band mean/std in the task's band order: shape ``[T, C]``."""
    mean = np.asarray([BURN_INTENSITY_MEANS[band] for band in bands], dtype=np.float32)
    std = np.asarray([BURN_INTENSITY_STDS[band] for band in bands], dtype=np.float32)
    return mean.T, std.T


# T4 (Landslide4Sense): 14-band HDF5 patches (128x128) whose bands follow the
# TerraTorch order below. The official config uses the six bands
# [BLUE, GREEN, RED, NIR_BROAD, SWIR_1, SWIR_2]; the HuggingFace copy's per-band
# scale does not match TerraTorch's published statistics, so the mean/std below
# are computed over the full official train split (3,799 patches) and applied
# identically to every backbone. Masks are binary (0 = non-landslide,
# 1 = landslide) with no ignore index.
LANDSLIDE_BAND_ORDER = (
    "COASTAL AEROSOL",
    "BLUE",
    "GREEN",
    "RED",
    "RED_EDGE_1",
    "RED_EDGE_2",
    "RED_EDGE_3",
    "NIR_BROAD",
    "WATER_VAPOR",
    "CIRRUS",
    "SWIR_1",
    "SWIR_2",
    "SLOPE",
    "DEM",
)
LANDSLIDE_MEANS = {
    "BLUE": 0.922701,
    "GREEN": 0.954109,
    "RED": 0.959639,
    "NIR_BROAD": 1.046756,
    "SWIR_1": 1.049497,
    "SWIR_2": 1.037032,
}
LANDSLIDE_STDS = {
    "BLUE": 0.220698,
    "GREEN": 0.318425,
    "RED": 0.572375,
    "NIR_BROAD": 0.494845,
    "SWIR_1": 0.532293,
    "SWIR_2": 0.662800,
}


def landslide_statistics(bands: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean/std in the task's band order (train-split statistics)."""
    mean = np.asarray([LANDSLIDE_MEANS[band] for band in bands], dtype=np.float32)
    std = np.asarray([LANDSLIDE_STDS[band] for band in bands], dtype=np.float32)
    return mean, std


def read_landslide_sample(
    image_path: Path, label_path: Path, bands: list[str], size: int = 224
) -> tuple[np.ndarray, np.ndarray]:
    """Read one Landslide4Sense patch as ``[C, size, size]`` and ``[size, size]``.

    The 14-band HDF5 image is subset to ``bands`` and both image and mask are
    resized to ``size`` (the official config resizes 128 -> 224); the mask uses
    nearest interpolation to stay binary.
    """
    import h5py
    import torch

    indices = [LANDSLIDE_BAND_ORDER.index(band) for band in bands]
    with h5py.File(image_path, "r") as handle:
        image = np.asarray(handle["img"])[..., indices]  # H, W, C
    with h5py.File(label_path, "r") as handle:
        label = np.asarray(handle["mask"])
    image_t = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    image_t = torch.nn.functional.interpolate(
        image_t, size=(size, size), mode="bilinear", align_corners=False
    )[0]
    label_t = torch.from_numpy(label).long().unsqueeze(0).unsqueeze(0).float()
    label_t = torch.nn.functional.interpolate(label_t, size=(size, size), mode="nearest")[
        0, 0
    ].long()
    return image_t.numpy(), label_t.numpy()


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
