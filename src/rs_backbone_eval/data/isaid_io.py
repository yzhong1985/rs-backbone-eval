"""Reading iSAID/DOTA imagery and masks for the segmentation workers.

Shared by the LIM and Clay workers (both evaluate the iSAID T9 cells), so the
palette and the band-count handling cannot drift apart -- a 4-band or 1-band
DOTA tile crashed two different call sites before this module existed
(review 2026-10-29).

Only workers import this: it pulls rasterio lazily so the control plane and its
tests (numpy only) stay dependency-free.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

# iSAID palette (P0-a report): colour -> class id. Ids follow iSAID_train.json's
# categories (1 storage_tank ... 15 Helicopter); black is background.
# Index = class id used by PALETTE and by the worker's per-class IoU metrics.
# Published iSAID mIoU is normally averaged over the 15 foreground classes;
# our headline metric includes background (id 0), so both numbers are printed.
ISAID_CLASS_NAMES = [
    "background",
    "storage_tank",
    "Large_Vehicle",
    "Small_Vehicle",
    "plane",
    "ship",
    "Swimming_pool",
    "Harbor",
    "tennis_court",
    "Ground_Track_Field",
    "Soccer_ball_field",
    "baseball_diamond",
    "Bridge",
    "basketball_court",
    "Roundabout",
    "Helicopter",
]

PALETTE = {
    (0x00, 0x00, 0x00): 0,
    (0x00, 0x3F, 0x3F): 1,  # storage_tank
    (0x00, 0x7F, 0x7F): 2,  # Large_Vehicle
    (0x00, 0x00, 0x7F): 3,  # Small_Vehicle
    (0x00, 0x7F, 0xFF): 4,  # plane
    (0x00, 0x00, 0x3F): 5,  # ship
    (0x00, 0x00, 0xFF): 6,  # Swimming_pool
    (0x00, 0x64, 0x9B): 7,  # Harbor
    (0x00, 0x3F, 0x7F): 8,  # tennis_court
    (0x00, 0x3F, 0xFF): 9,  # Ground_Track_Field
    (0x00, 0x7F, 0xBF): 10,  # Soccer_ball_field
    (0x00, 0x3F, 0x00): 11,  # baseball_diamond
    (0x00, 0x7F, 0x3F): 12,  # Bridge
    (0x00, 0x3F, 0xBF): 13,  # basketball_court
    (0x00, 0xBF, 0x7F): 14,  # Roundabout
    (0x00, 0x00, 0xBF): 15,  # Helicopter
}


def rgb_only(array: np.ndarray) -> np.ndarray:
    """Return ``[..., 3]``: drop alpha, replicate a single (or gray+alpha) band.

    Some DOTA scenes are RGBA and ~14% of iSAID images are single-band. They are
    kept -- dropping them would remove 14% of the benchmark -- and converted
    here so every consumer sees three channels.
    """
    bands = array.shape[-1]
    if bands == 3:
        return array
    if bands == 4:  # RGBA
        return array[..., :3]
    if bands == 1:  # grayscale
        return np.repeat(array, 3, axis=-1)
    if bands == 2:  # gray + alpha
        return np.repeat(array[..., :1], 3, axis=-1)
    raise ValueError(f"expected 1-4 bands, got {bands}")


def colour_lookup() -> np.ndarray:
    """``[256,256,256]`` RGB -> class id (unknown colours stay background)."""
    lut = np.zeros((256, 256, 256), dtype=np.uint8)
    for (red, green, blue), klass in PALETTE.items():
        lut[red, green, blue] = klass
    return lut


def read_rgb(path: Path, window: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Read a PNG (whole or ``[x, y, w, h]``) as ``[h, w, 3]`` uint8.

    Reads outside the raster are filled with zero, which is exactly the padding
    the tiling protocol expects at an edge tile.
    """
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning
    from rasterio.windows import Window

    with warnings.catch_warnings():
        # DOTA/iSAID PNGs carry no geotransform; the warning would fire per tile.
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(path) as source:
            if window is None:
                data = source.read()
            else:
                x, y, width, height = window
                data = source.read(window=Window(x, y, width, height), boundless=True, fill_value=0)
    return rgb_only(np.transpose(data, (1, 2, 0)))


def read_label(path: Path, window: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Read a semantic mask and map its palette colours to class ids."""
    mask = read_rgb(path, window)
    return colour_lookup()[mask[..., 0], mask[..., 1], mask[..., 2]]


def band_counts(rows: list[dict], data_root: Path) -> dict[int, int]:
    """How many source images carry 1/3/4 bands (header read only)."""
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning

    counts: dict[int, int] = {}
    seen: set[str] = set()
    for row in rows:
        relative = row["image_paths"][0]
        if relative in seen:
            continue
        seen.add(relative)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(data_root / relative) as source:
                counts[source.count] = counts.get(source.count, 0) + 1
    return counts
