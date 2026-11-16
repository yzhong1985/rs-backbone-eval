"""CPU-only tests for the WHU-OPT-SAR sheet-name georeferencing helper."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    spec = importlib.util.spec_from_file_location(
        "whu_tile_bbox", ROOT / "scripts" / "whu_tile_bbox.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses resolves the defining module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sheet_bounds_match_the_gb_t_13989_grid() -> None:
    module = load_module()
    # H row = 28..32 N; column 49 = 108..114 E; row 004 / column 021 inside it.
    sheet = module.decode("NH49E004021.tif")
    assert sheet.north == pytest.approx(31.5)
    assert sheet.south == pytest.approx(31.5 - 1 / 6)
    assert sheet.west == pytest.approx(113.0)
    assert sheet.east == pytest.approx(113.25)


def test_sheet_index_001_is_the_north_west_corner() -> None:
    module = load_module()
    sheet = module.decode("NH49E001001")
    assert sheet.north == pytest.approx(32.0)
    assert sheet.west == pytest.approx(108.0)


def test_pixel_grid_is_the_same_in_both_axes() -> None:
    module = load_module()
    # This equality is what shows the raster fills the sheet bounds exactly.
    assert pytest.approx(module.DEG_PER_PX_Y, rel=1e-12) == module.DEG_PER_PX_X
    assert pytest.approx(0.25) == module.TILE_WIDTH_PX * module.DEG_PER_PX_X
    assert pytest.approx(1 / 6) == module.TILE_HEIGHT_PX * module.DEG_PER_PX_Y


def test_pixel_centre_and_affine_round_trip() -> None:
    module = load_module()
    sheet = module.decode("NH49E004021")
    lon, lat = sheet.pixel_centre(0, 0)
    assert lon == pytest.approx(sheet.west + module.DEG_PER_PX_X / 2)
    assert lat == pytest.approx(sheet.north - module.DEG_PER_PX_Y / 2)
    a, b, c, d, e, f = sheet.transform()
    assert (b, d) == (0.0, 0.0)
    assert (c, f) == (sheet.west, sheet.north)
    assert a == pytest.approx(module.DEG_PER_PX_X)
    assert e == pytest.approx(-module.DEG_PER_PX_Y)
    with pytest.raises(ValueError):
        sheet.pixel_centre(module.TILE_WIDTH_PX, 0)


def test_duplicates_and_bad_names() -> None:
    module = load_module()
    assert module.decode("optical/NH49E004021(1).tif").name == "NH49E004021"
    with pytest.raises(ValueError):
        module.decode("NH49E025001")  # row index above 24
    with pytest.raises(ValueError):
        module.decode("NH49X004021")  # scale letter must be E
    with pytest.raises(ValueError):
        module.decode("NH49E004021.tiff")
