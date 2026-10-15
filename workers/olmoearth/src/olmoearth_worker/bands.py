"""Map the project's six-band Sentinel-2 chips onto OlmoEarth's S2 L2A entry.

OlmoEarth's ``SENTINEL2_L2A`` modality expects **all 12 bands** and the encoder
selects fixed positions from the channel axis, so a six-band tensor raises
``IndexError``. This module builds the 12-band tensor by placing our bands at
their OlmoEarth positions and filling the rest, and reports which band *sets*
are fully present.

Masks in OlmoEarth are per band-set, not per band: band-set 1 holds both our
SWIR bands (B11/B12) and four bands we do not have (B05/B06/B07/B8A), so that
set must be sent with filled values rather than masked out. The caller records
this as an input adapter, not as a pure frozen probe.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# OlmoEarth SENTINEL2_L2A band order, grouped as OlmoEarth's three band-sets.
OLMOEARTH_S2L2A_BANDSETS: tuple[tuple[str, ...], ...] = (
    ("B02", "B03", "B04", "B08"),
    ("B05", "B06", "B07", "B8A", "B11", "B12"),
    ("B01", "B09"),
)
OLMOEARTH_S2L2A_BANDS: tuple[str, ...] = tuple(
    band for band_set in OLMOEARTH_S2L2A_BANDSETS for band in band_set
)
BANDSET_INDEX: dict[str, int] = {band: index for index, band in enumerate(OLMOEARTH_S2L2A_BANDS)}

# Project band name -> OlmoEarth band name.
PROJECT_TO_OLMOEARTH: dict[str, str] = {
    "BLUE": "B02",
    "GREEN": "B03",
    "RED": "B04",
    "NIR_NARROW": "B08",
    "SWIR_1": "B11",
    "SWIR_2": "B12",
}

FILL_VALUE = 0.0


def bandset_presence(bands: Sequence[str]) -> tuple[bool, bool, bool]:
    """Which OlmoEarth band-sets are fully covered by ``bands``."""
    mapped = {PROJECT_TO_OLMOEARTH.get(name, name) for name in bands}
    return tuple(set(band_set) <= mapped for band_set in OLMOEARTH_S2L2A_BANDSETS)  # type: ignore[return-value]


def to_olmoearth_s2l2a(
    pixels: np.ndarray, bands: Sequence[str], *, fill: float = FILL_VALUE
) -> np.ndarray:
    """Expand a ``[C, H, W]`` project chip to OlmoEarth's ``[12, H, W]`` layout.

    Bands the project does not have are filled with ``fill``; ``bandset_presence``
    tells the caller which band-sets are complete so it can set the masks.
    """
    array = np.asarray(pixels)
    if array.ndim != 3:
        raise ValueError(f"pixels must be [C, H, W], got {array.shape}")
    if len(bands) != array.shape[0]:
        raise ValueError(f"got {len(bands)} band names for {array.shape[0]} channels")
    unknown = [name for name in bands if name not in PROJECT_TO_OLMOEARTH]
    if unknown:
        raise ValueError(f"unmapped project bands: {unknown}")

    out = np.full((len(OLMOEARTH_S2L2A_BANDS), *array.shape[1:]), fill, dtype=array.dtype)
    for channel, name in enumerate(bands):
        out[BANDSET_INDEX[PROJECT_TO_OLMOEARTH[name]]] = array[channel]
    return out
