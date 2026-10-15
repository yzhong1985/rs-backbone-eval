"""Tests for the OlmoEarth S2 L2A band adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers" / "olmoearth" / "src"))

from olmoearth_worker.bands import (  # noqa: E402
    BANDSET_INDEX,
    OLMOEARTH_S2L2A_BANDS,
    bandset_presence,
    to_olmoearth_s2l2a,
)

PROJECT_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]


def test_olmoearth_band_order_matches_the_three_bandsets() -> None:
    assert len(OLMOEARTH_S2L2A_BANDS) == 12
    assert OLMOEARTH_S2L2A_BANDS[:4] == ("B02", "B03", "B04", "B08")
    # our six bands land at 0,1,2,3 (band-set 0) and 8,9 (SWIR in band-set 1)
    assert [BANDSET_INDEX[name] for name in ("B02", "B03", "B04", "B08", "B11", "B12")] == [
        0,
        1,
        2,
        3,
        8,
        9,
    ]


def test_to_olmoearth_places_bands_and_fills_the_rest() -> None:
    chip = np.arange(6 * 2 * 3, dtype=np.float32).reshape(6, 2, 3)
    out = to_olmoearth_s2l2a(chip, PROJECT_BANDS)
    assert out.shape == (12, 2, 3)
    for channel, position in enumerate([0, 1, 2, 3, 8, 9]):
        np.testing.assert_array_equal(out[position], chip[channel])
    # bands we do not have are filled
    for position in [4, 5, 6, 7, 10, 11]:
        assert (out[position] == 0.0).all()


def test_bandset_presence_flags_partial_swir_set() -> None:
    # band-set 0 is complete; band-set 1 is partial (B11/B12 only); band-set 2 absent
    assert bandset_presence(PROJECT_BANDS) == (True, False, False)


def test_unmapped_band_is_rejected() -> None:
    with pytest.raises(ValueError, match="unmapped"):
        to_olmoearth_s2l2a(np.zeros((2, 2, 2), dtype=np.float32), ["BLUE", "BOGUS"])


def test_channel_count_must_match_names() -> None:
    with pytest.raises(ValueError, match="band names"):
        to_olmoearth_s2l2a(np.zeros((6, 2, 2), dtype=np.float32), PROJECT_BANDS[:5])
