import pytest

from rs_backbone_eval.bands import (
    ANYSAT_S2_BANDS,
    HLS_WAVELENGTHS_UM,
    canonical_band,
    missing_anysat_bands,
    present_anysat_positions,
)

T1_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
T2_BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]


def test_t1_bands_map_onto_anysat_s2() -> None:
    positions = present_anysat_positions(T1_BANDS)

    assert positions == [
        ANYSAT_S2_BANDS.index(name) for name in ["B2", "B3", "B4", "B8A", "B11", "B12"]
    ]
    assert missing_anysat_bands(T1_BANDS) == ["B5", "B6", "B7", "B8"]


def test_hls_bands_map_onto_anysat_s2() -> None:
    positions = present_anysat_positions(T2_BANDS)

    assert positions == [
        ANYSAT_S2_BANDS.index(name) for name in ["B2", "B3", "B4", "B8", "B11", "B12"]
    ]
    # The adapter must synthesize the three red edges plus the narrow NIR.
    assert missing_anysat_bands(T2_BANDS) == ["B5", "B6", "B7", "B8A"]


def test_band_aliases_normalize() -> None:
    assert canonical_band("nir") == "NIR_BROAD"
    assert canonical_band("SWIR1") == "SWIR_1"
    assert canonical_band("blue") == "BLUE"


def test_unknown_band_is_rejected() -> None:
    with pytest.raises(ValueError, match="no AnySat S2 counterpart"):
        present_anysat_positions(["COASTAL_AEROSOL"])


def test_hls_wavelengths_cover_every_task_band_name() -> None:
    # T3 (burn intensity) names the HLS narrow NIR "NIR"; the Clay T3 encoder
    # looks the name up directly, so the table must carry the alias.
    t3_bands = ["BLUE", "GREEN", "RED", "NIR", "SWIR_1", "SWIR_2"]
    assert all(band in HLS_WAVELENGTHS_UM for band in t3_bands)
    assert HLS_WAVELENGTHS_UM["NIR"] == HLS_WAVELENGTHS_UM["NIR_NARROW"]
