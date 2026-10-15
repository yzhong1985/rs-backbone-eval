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
    bandset_coverage,
    bandset_usable,
    bandset_usable_after_drop,
    to_olmoearth_s2l2a,
)

PROJECT_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]


def test_olmoearth_band_order_matches_the_three_bandsets() -> None:
    assert len(OLMOEARTH_S2L2A_BANDS) == 12
    assert OLMOEARTH_S2L2A_BANDS[:4] == ("B02", "B03", "B04", "B08")
    assert OLMOEARTH_S2L2A_BANDS[4:10] == ("B05", "B06", "B07", "B8A", "B11", "B12")


def test_nir_narrow_maps_to_b8a_not_b08() -> None:
    # The project's NIR_NARROW is Sentinel-2 B8A (865 nm narrow), not B08.
    assert BANDSET_INDEX["B8A"] == 7
    assert BANDSET_INDEX["B08"] == 3


def test_project_bands_cover_no_bandset_fully_but_use_two() -> None:
    # B08 (set 0) and B05/B06/B07 (set 1) are missing, so nothing is complete...
    assert bandset_coverage(PROJECT_BANDS) == (False, False, False)
    # ...but set 0 and set 1 each have a present band, so they must be sent
    # online (with filled values) rather than masked away.
    assert bandset_usable(PROJECT_BANDS) == (True, True, False)


def test_to_olmoearth_places_bands_and_fills_the_rest() -> None:
    chip = np.arange(6 * 2 * 3, dtype=np.float32).reshape(6, 2, 3)
    out = to_olmoearth_s2l2a(chip, PROJECT_BANDS)
    assert out.shape == (12, 2, 3)
    for channel, position in enumerate([0, 1, 2, 7, 8, 9]):
        np.testing.assert_array_equal(out[position], chip[channel])
    # B08 and B05/B06/B07 (set 1) and B01/B09 are filled
    for position in [3, 4, 5, 6, 10, 11]:
        assert (out[position] == 0.0).all()


def test_unmapped_band_is_rejected() -> None:
    with pytest.raises(ValueError, match="unmapped"):
        to_olmoearth_s2l2a(np.zeros((2, 2, 2), dtype=np.float32), ["BLUE", "BOGUS"])


def test_channel_count_must_match_names() -> None:
    with pytest.raises(ValueError, match="band names"):
        to_olmoearth_s2l2a(np.zeros((6, 2, 2), dtype=np.float32), PROJECT_BANDS[:5])


def test_bandset_usable_after_drop_masks_sets_without_a_real_band() -> None:
    """A dropped band counts as never acquired, so its set may go MISSING."""
    # dropping the two SWIRs keeps set 2 alive through NIR_NARROW (B8A)
    assert bandset_usable_after_drop(PROJECT_BANDS, [4, 5]) == (True, True, False)
    # dropping everything set 2 really contains (B8A, B11, B12) masks the set
    assert bandset_usable_after_drop(PROJECT_BANDS, [3, 4, 5]) == (True, False, False)
    # B08 was never real; without the visible bands set 1 has no real member
    assert bandset_usable_after_drop(PROJECT_BANDS, [0, 1, 2]) == (False, True, False)
