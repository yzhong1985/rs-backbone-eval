"""Map the project's six-band Sentinel-2 chips onto OlmoEarth's S2 L2A entry.

OlmoEarth's ``SENTINEL2_L2A`` modality expects **all 12 bands** and the encoder
selects fixed positions from the channel axis, so a six-band tensor raises
``IndexError``. This module builds the 12-band tensor by placing our bands at
their OlmoEarth positions and filling the rest.

Band names follow the project's own mapping: ``NIR_NARROW`` is Sentinel-2
**B8A** (865 nm narrow), not B08. With the project's six bands
(``BLUE/GREEN/RED/NIR_NARROW/SWIR_1/SWIR_2``) **no OlmoEarth band-set is fully
covered**:

- band-set 0 ``[B02,B03,B04,B08]``: missing B08
- band-set 1 ``[B05,B06,B07,B8A,B11,B12]``: missing B05/B06/B07
- band-set 2 ``[B01,B09]``: absent

Because OlmoEarth masks are per band-set (not per band), a partly-present set
cannot be masked band-by-band: it must be sent **online with filled values** for
the bands we lack, and only a set with *no* band present may be masked MISSING.
The caller records this as an input adapter, not as a pure frozen probe.
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

# Project band name -> OlmoEarth band name. NIR_NARROW is B8A (865 nm narrow).
PROJECT_TO_OLMOEARTH: dict[str, str] = {
    "BLUE": "B02",
    "GREEN": "B03",
    "RED": "B04",
    "NIR_NARROW": "B8A",
    "SWIR_1": "B11",
    "SWIR_2": "B12",
}

FILL_VALUE = 0.0


def _mapped(bands: Sequence[str]) -> set[str]:
    unknown = [name for name in bands if name not in PROJECT_TO_OLMOEARTH]
    if unknown:
        raise ValueError(f"unmapped project bands: {unknown}")
    return {PROJECT_TO_OLMOEARTH[name] for name in bands}


def bandset_coverage(bands: Sequence[str]) -> tuple[bool, ...]:
    """Which OlmoEarth band-sets are **fully** covered by ``bands``."""
    mapped = _mapped(bands)
    return tuple(set(band_set) <= mapped for band_set in OLMOEARTH_S2L2A_BANDSETS)


def bandset_usable(bands: Sequence[str]) -> tuple[bool, ...]:
    """Which band-sets have **at least one** band present (drives the mask).

    A partly-present band-set must be sent online with filled values; only a set
    with no band present is masked MISSING.
    """
    mapped = _mapped(bands)
    return tuple(bool(set(band_set) & mapped) for band_set in OLMOEARTH_S2L2A_BANDSETS)


def to_olmoearth_s2l2a(
    pixels: np.ndarray, bands: Sequence[str], *, fill: float = FILL_VALUE
) -> np.ndarray:
    """Expand a ``[C, H, W]`` project chip to OlmoEarth's ``[12, H, W]`` layout.

    Bands the project does not have are filled with ``fill``; ``bandset_usable``
    tells the caller which band-sets to send online so a partly-present set (for
    example the SWIR set) is not dropped.
    """
    array = np.asarray(pixels)
    if array.ndim != 3:
        raise ValueError(f"pixels must be [C, H, W], got {array.shape}")
    if len(bands) != array.shape[0]:
        raise ValueError(f"got {len(bands)} band names for {array.shape[0]} channels")
    _mapped(bands)  # reject unknown names before writing

    out = np.full((len(OLMOEARTH_S2L2A_BANDS), *array.shape[1:]), fill, dtype=array.dtype)
    for channel, name in enumerate(bands):
        out[BANDSET_INDEX[PROJECT_TO_OLMOEARTH[name]]] = array[channel]
    return out


def bandset_usable_after_drop(
    bands: Sequence[str], drop_indices: Sequence[int]
) -> tuple[bool, ...]:
    """``bandset_usable`` for the input after ``drop_indices`` channels are removed.

    A perturbation test (``band_drop:B11,B12``) treats the dropped bands as if
    they were never acquired: their values become the fill and a band-set that
    has none of its bands left real is masked MISSING, matching how the encoder
    was pretrained with partially present band-sets.
    """
    remaining = [band for index, band in enumerate(bands) if index not in set(drop_indices)]
    return bandset_usable(remaining)
