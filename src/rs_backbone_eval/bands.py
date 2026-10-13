"""Auditable band-name mappings shared by every model worker.

AnySat's S2 entrance is fixed to ten bands; Clay and Prithvi take the task's
band list directly. When a task's bands do not cover AnySat's ten, a spectral
adapter (explicitly trained and reported) synthesizes the missing ones -- the
design forbids zero padding.

Keeping the mapping here (torch-free) lets the control plane and tests audit
it without importing a model framework.
"""

from __future__ import annotations

# AnySat's fixed Sentinel-2 band order.
ANYSAT_S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Task band name -> AnySat S2 band name it can supply directly.
TASK_BAND_TO_ANYSAT = {
    # Sentinel-2 L2A / L1C naming
    "BLUE": "B2",
    "GREEN": "B3",
    "RED": "B4",
    "NIR_NARROW": "B8A",
    "NIR_BROAD": "B8",
    "SWIR_1": "B11",
    "SWIR_2": "B12",
    "RED_EDGE_1": "B5",
    "RED_EDGE_2": "B6",
    "RED_EDGE_3": "B7",
}

# Band-name synonyms across datasets (HLS, Landsat, S2) mapped to task names.
BAND_ALIASES = {
    "NIR": "NIR_BROAD",  # HLS / Landsat naming for the broad NIR band
    "NIR1": "NIR_BROAD",
    "SWIR1": "SWIR_1",
    "SWIR2": "SWIR_2",
    "COASTAL_AEROSOL": "COASTAL_AEROSOL",
    "REDEDGE1": "RED_EDGE_1",
    "REDEDGE2": "RED_EDGE_2",
    "REDEDGE3": "RED_EDGE_3",
}

# Approximate central wavelengths in micrometres for HLS bands, used to build
# auditable derived metadata for sensor-agnostic models (Clay).
HLS_WAVELENGTHS_UM = {
    "BLUE": 0.48,
    "GREEN": 0.56,
    "RED": 0.655,
    "NIR_BROAD": 0.865,
    "NIR_NARROW": 0.865,
    "SWIR_1": 1.61,
    "SWIR_2": 2.20,
}


def canonical_band(name: str) -> str:
    """Normalize a dataset-specific band name to the task vocabulary."""
    upper = name.upper()
    return BAND_ALIASES.get(upper, upper)


def present_anysat_positions(bands: list[str]) -> list[int]:
    """Positions in AnySat's ten-band tensor that the task bands provide."""
    positions: list[int] = []
    for band in bands:
        name = TASK_BAND_TO_ANYSAT.get(canonical_band(band))
        if name is None:
            raise ValueError(f"task band {band} has no AnySat S2 counterpart")
        positions.append(ANYSAT_S2_BANDS.index(name))
    return positions


def missing_anysat_bands(bands: list[str]) -> list[str]:
    """AnySat S2 bands the task bands do not provide (adapter must synthesize)."""
    provided = set(present_anysat_positions(bands))
    return [name for index, name in enumerate(ANYSAT_S2_BANDS) if index not in provided]
