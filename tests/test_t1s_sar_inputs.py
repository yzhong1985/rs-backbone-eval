"""T1S (Sentinel-1) input handling in the Clay and AnySat workers.

The T1S paths had never completed a real run: Clay looked up its radar
statistics under the wrong key and used an invented wavelength instead of its
own ``sentinel-1-rtc`` entry, both workers let NaN backscatter into the
backbone, and AnySat passed the year as its day-of-year date. These tests pin
the corrected input contract on synthetic arrays (CPU only, no data).
Run in a torch environment (e.g. workers/prithvi/.venv/bin/python -m pytest).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rasterio")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers" / "anysat" / "src"))
sys.path.insert(0, str(ROOT / "workers" / "clay" / "src"))

from anysat_worker.t1_sar import IGNORE_INDEX, fill_invalid, sar_channels  # noqa: E402
from clay_worker.t1 import derive_metadata  # noqa: E402

CLAY_METADATA = """
sentinel-2-l2a:
  band_order: [blue, green, red, nir, swir16, swir22]
  gsd: 10
  bands:
    mean: {blue: 1.0, green: 2.0, red: 3.0, nir: 4.0, swir16: 5.0, swir22: 6.0}
    std: {blue: 1.0, green: 1.0, red: 1.0, nir: 1.0, swir16: 1.0, swir22: 1.0}
    wavelength: {blue: 0.493, green: 0.56, red: 0.665, nir: 0.842, swir16: 1.61, swir22: 2.19}
sentinel-1-rtc:
  band_order: [vv, vh]
  gsd: 10
  bands:
    mean: {vv: -12.113, vh: -18.673}
    std: {vv: 8.314, vh: 8.017}
    wavelength: {vv: 3.5, vh: 4.0}
"""


def test_clay_sar_uses_its_own_sentinel1_metadata(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "metadata.yaml").write_text(CLAY_METADATA, encoding="utf-8")
    meta = derive_metadata(tmp_path, ["VV", "VH"], "sar")
    assert meta["sensor"] == "sentinel-1-rtc"
    assert meta["bands"] == ["vv", "vh"]
    # Keys match the band order the worker indexes with (the old KeyError).
    for field in ("mean", "std", "wavelength"):
        assert list(meta["bands_meta"][field]) == ["vv", "vh"]
    # The pretraining wavelengths, not the physical C-band ~5.5e4 um.
    assert meta["bands_meta"]["wavelength"] == {"vv": 3.5, "vh": 4.0}


def test_clay_optical_metadata_is_unchanged(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "metadata.yaml").write_text(CLAY_METADATA, encoding="utf-8")
    meta = derive_metadata(tmp_path, ["BLUE", "SWIR_2"])
    assert meta["sensor"] == "sentinel-2-l2a"
    assert meta["bands_meta"]["wavelength"] == {"blue": 0.493, "swir22": 2.19}
    assert meta["wavelength_override"] is None


def test_clay_wavelength_override_drives_the_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T1S wavelength ablation: the override must reach the dynamic embedding."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "metadata.yaml").write_text(CLAY_METADATA, encoding="utf-8")

    monkeypatch.setenv("CLAY_WORKER_WAVELENGTHS_UM", "55000,55000")
    meta = derive_metadata(tmp_path, ["VV", "VH"], "sar")
    assert meta["bands_meta"]["wavelength"] == {"vv": 55000.0, "vh": 55000.0}
    assert meta["wavelength_override"] == [55000.0, 55000.0]

    monkeypatch.setenv("CLAY_WORKER_WAVELENGTHS_UM", "4.0,3.5")
    meta = derive_metadata(tmp_path, ["VV", "VH"], "sar")
    assert meta["bands_meta"]["wavelength"] == {"vv": 4.0, "vh": 3.5}


def test_clay_wavelength_override_rejects_a_wrong_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "metadata.yaml").write_text(CLAY_METADATA, encoding="utf-8")

    monkeypatch.setenv("CLAY_WORKER_WAVELENGTHS_UM", "3.5")
    with pytest.raises(ValueError, match="needs 2 values"):
        derive_metadata(tmp_path, ["VV", "VH"], "sar")


def test_anysat_fill_invalid_masks_unobserved_pixels() -> None:
    raw = np.full((2, 4, 4), -10.0, dtype=np.float32)
    raw[:, 0, 0] = np.nan
    raw[1, 3, 3] = np.nan  # one polarisation missing is still unobserved
    chip = sar_channels(raw)
    label = np.ones((4, 4), dtype=np.int64)
    fill = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    filled, masked = fill_invalid(chip, label, fill)

    assert np.isfinite(filled).all()
    assert filled[:, 0, 0].tolist() == [1.0, 2.0, 3.0]
    assert masked[0, 0] == IGNORE_INDEX and masked[3, 3] == IGNORE_INDEX
    assert (masked == IGNORE_INDEX).sum() == 2
    # Observed pixels are untouched.
    assert filled[0, 1, 1] == -10.0 and masked[1, 1] == 1


def test_anysat_fill_invalid_is_a_no_op_on_clean_chips() -> None:
    chip = sar_channels(np.full((2, 3, 3), -8.0, dtype=np.float32))
    label = np.zeros((3, 3), dtype=np.int64)
    filled, masked = fill_invalid(chip, label, None)
    assert filled is chip and masked is label
