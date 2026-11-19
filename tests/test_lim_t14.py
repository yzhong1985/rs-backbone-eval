"""CPU tests for the T14 change-mask writer (LIM worker env, rasterio only).

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_t14.py -q
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")
from lim_worker import t14_labels  # noqa: E402

AOI = "L15-TEST_0_0_13"
SIZE = 16


def _geojson(path: Path, boxes: list[tuple[int, int, int, int]]) -> None:
    features = []
    for x0, y0, x1, y1 in boxes:
        ring = [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]
        features.append(
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8"
    )


def _row(stamp1: str, stamp2: str) -> dict:
    stem = "global_monthly_{}_mosaic_" + AOI
    t1 = f"raw/spacenet7/train/{AOI}/images/{stem.format(stamp1)}.tif"
    t2 = f"raw/spacenet7/train/{AOI}/images/{stem.format(stamp2)}.tif"
    return {
        "task_id": "T14",
        "sample_id": f"{AOI}_{stamp1}_{stamp2}",
        "group_id": AOI,
        "split": "train",
        "image_paths": [t1, t2],
        "label_path": f"derived/spacenet7-change/{AOI}/{stamp1}_{stamp2}.tif",
        "window": [0, 0, SIZE, SIZE],
        "transform": [4.78, 0.0, 0.0, 0.0, -4.78, 0.0],
        "crs": "EPSG:3857",
    }


def test_change_is_new_building_pixels_only(tmp_path: Path) -> None:
    labels = tmp_path / "raw/spacenet7/train" / AOI / "labels_match_pix"
    base = f"global_monthly_{{}}_mosaic_{AOI}_Buildings.geojson"
    _geojson(labels / base.format("2018_01"), [(2, 2, 5, 5)])
    _geojson(labels / base.format("2018_02"), [(2, 2, 5, 5), (10, 10, 12, 12)])
    row = _row("2018_01", "2018_02")
    mask, missing, outside = t14_labels.change_mask(tmp_path, row)
    assert mask.shape == (SIZE, SIZE) and mask.dtype == np.uint8
    assert mask.sum() == 4
    assert mask[10:12, 10:12].all()
    assert mask[2:5, 2:5].sum() == 0
    assert missing == 2  # no UDM files in this fixture
    assert outside == 0


R = 6378137.0
# Row transform of L15-0358E-1220N_1433_3310_13 (2018-12 image, EPSG:3857, 1024 px window).
AOI_TRANSFORM = [4.77731426716, 0.0, -13027315.604699157, 0.0, -4.77731426716, 3845088.270857503]


def _pixel_to_lonlat(col: float, row: float, transform: list[float]) -> tuple[float, float]:
    """Inverse of the spherical Web Mercator used by the module: pixel corner to lon/lat."""
    a, _, c, _, e, f = transform
    x, y = a * col + c, e * row + f
    lon = math.degrees(x / R)
    lat = math.degrees(2 * math.atan(math.exp(y / R)) - math.pi / 2)
    return lon, lat


def _udm_lonlat(path: Path, boxes: list[tuple[int, int, int, int]], transform: list[float]) -> None:
    """Write a CRS84 UDM geojson whose rectangles cover the given pixel boxes."""
    features = []
    for x0, y0, x1, y1 in boxes:
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        ring = [list(_pixel_to_lonlat(c, r, transform)) for c, r in corners]
        features.append(
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    crs = {"type": "name", "properties": {"name": t14_labels.UDM_CRS}}
    doc = {"type": "FeatureCollection", "crs": crs, "features": features}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _udm(root: Path, stamp: str, boxes: list[tuple[int, int, int, int]]) -> None:
    """UDM for one month, with pixel boxes of the synthetic 16 px row written as lon/lat."""
    stem = f"global_monthly_{stamp}_mosaic_{AOI}"
    _udm_lonlat(
        root / "raw/spacenet7/train" / AOI / "labels" / f"{stem}_UDM.geojson",
        boxes,
        [4.78, 0.0, 0.0, 0.0, -4.78, 0.0],
    )


def test_cloud_from_either_date_is_255_even_over_a_building(tmp_path: Path) -> None:
    labels = tmp_path / "raw/spacenet7/train" / AOI / "labels_match_pix"
    base = f"global_monthly_{{}}_mosaic_{AOI}_Buildings.geojson"
    _geojson(labels / base.format("2018_01"), [])
    _geojson(labels / base.format("2018_02"), [(2, 2, 5, 5), (10, 10, 12, 12)])
    _udm(tmp_path, "2018_01", [(2, 2, 5, 5)])  # cloud at t1 over the t2 building
    _udm(tmp_path, "2018_02", [(10, 10, 11, 12)])  # cloud at t2 over half of the other one
    mask, missing, outside = t14_labels.change_mask(tmp_path, _row("2018_01", "2018_02"))
    assert missing == 0 and outside == 0
    assert (mask[2:5, 2:5] == 255).all()  # 9 px, all cloud
    assert (mask[10:12, 10] == 255).all()  # cloud wins over the building
    assert (mask[10:12, 11] == 1).all()  # building outside the cloud stays change
    assert int((mask == 255).sum()) == 11
    assert int((mask == 1).sum()) == 2


def test_missing_udm_files_are_no_cloud_and_are_counted(tmp_path: Path) -> None:
    labels = tmp_path / "raw/spacenet7/train" / AOI / "labels_match_pix"
    base = f"global_monthly_{{}}_mosaic_{AOI}_Buildings.geojson"
    _geojson(labels / base.format("2018_01"), [])
    _geojson(labels / base.format("2018_02"), [(10, 10, 12, 12)])
    _udm(tmp_path, "2018_02", [])  # t2 UDM exists but is empty; t1 UDM is absent
    row = _row("2018_01", "2018_02")
    mask, missing, outside = t14_labels.change_mask(tmp_path, row)
    assert missing == 1 and outside == 0
    assert int((mask == 255).sum()) == 0 and mask.sum() == 4
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    counts = t14_labels.build_masks(tmp_path, manifest)
    assert counts == {
        "written": 1,
        "skipped": 0,
        "udm_missing": 1,
        "udm_outside": 0,
        "cloud_px": 0,
    }


def test_writes_geotiff_on_the_image_grid_and_skips_existing(tmp_path: Path) -> None:
    labels = tmp_path / "raw/spacenet7/train" / AOI / "labels_match_pix"
    base = f"global_monthly_{{}}_mosaic_{AOI}_Buildings.geojson"
    _geojson(labels / base.format("2018_01"), [])
    _geojson(labels / base.format("2018_02"), [(10, 10, 12, 12)])
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(r)
            for r in [_row("2018_01", "2018_02"), {**_row("2018_01", "2018_02"), "task_id": "T13"}]
        )
        + "\n",
        encoding="utf-8",
    )
    assert t14_labels.build_masks(tmp_path, manifest) == {
        "written": 1,
        "skipped": 0,
        "udm_missing": 2,
        "udm_outside": 0,
        "cloud_px": 0,
    }
    assert t14_labels.build_masks(tmp_path, manifest) == {
        "written": 0,
        "skipped": 1,
        "udm_missing": 0,
        "udm_outside": 0,
        "cloud_px": 0,
    }
    import rasterio

    with rasterio.open(tmp_path / _row("2018_01", "2018_02")["label_path"]) as src:
        assert src.crs.to_string() == "EPSG:3857"
        assert src.transform.a == 4.78 and src.transform.e == -4.78
        assert int(src.read(1).sum()) == 4


def test_missing_footprint_file_is_an_error(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        t14_labels.change_mask(tmp_path, _row("2018_01", "2018_02"))


def _udm_path(root: Path, stamp: str) -> Path:
    return (
        root
        / "raw/spacenet7/train"
        / AOI
        / "labels"
        / f"global_monthly_{stamp}_mosaic_{AOI}_UDM.geojson"
    )


def test_udm_lonlat_rectangle_burns_exactly_its_pixels(tmp_path: Path) -> None:
    labels = tmp_path / "raw/spacenet7/train" / AOI / "labels_match_pix"
    base = f"global_monthly_{{}}_mosaic_{AOI}_Buildings.geojson"
    _geojson(labels / base.format("2018_01"), [])
    _geojson(labels / base.format("2018_02"), [])
    _udm(tmp_path, "2018_01", [(3, 4, 7, 9)])  # cols 3-6, rows 4-8
    mask, missing, outside = t14_labels.change_mask(tmp_path, _row("2018_01", "2018_02"))
    expected = np.zeros((SIZE, SIZE), dtype=bool)
    expected[4:9, 3:7] = True
    assert np.array_equal(mask == 255, expected)
    assert missing == 1 and outside == 0


def test_aoi_udm_in_lonlat_lands_inside_the_1024_window(tmp_path: Path) -> None:
    # Pixel rectangle cols 777-1027, rows 44-263 of the L15-0358E-1220N_1433_3310_13 grid.
    # Columns 1024-1027 lie past the 1024 px window, so only cols 777-1023 burn.
    row = {
        **_row("2018_04", "2018_05"),
        "transform": AOI_TRANSFORM,
        "window": [0, 0, 1024, 1024],
    }
    path = t14_labels.udm_path(tmp_path, row, row["image_paths"][1])
    _udm_lonlat(path, [(777, 44, 1027, 263)], AOI_TRANSFORM)
    cloud, missing, outside = t14_labels.cloud_mask(tmp_path, row, (1024, 1024))
    assert missing == 1 and outside == 0  # bbox crosses the window edge, so not outside
    assert int(cloud.sum()) == 247 * 219
    assert cloud[44:263, 777:1024].all()
    assert int(cloud[:44].sum() + cloud[263:].sum()) == 0


def test_lonlat_projection_matches_epsg3857_reference() -> None:
    # Reference values from pyproj (OGC:CRS84 -> EPSG:3857) for the AOI sample point.
    x0, y0 = t14_labels.lonlat_to_web_mercator(0.0, 0.0)
    assert x0 == pytest.approx(0.0, abs=1e-6) and y0 == pytest.approx(0.0, abs=1e-6)
    lon, lat = -116.98227926688941, 32.618195912781005
    x, y = t14_labels.lonlat_to_web_mercator(lon, lat)
    assert x == pytest.approx(-13022407.759826653, abs=1e-3)
    assert y == pytest.approx(3844734.8225707985, abs=1e-3)
    col, row = t14_labels.lonlat_to_pixel(lon, lat, AOI_TRANSFORM)
    assert col == pytest.approx(1027.3230099685425, abs=1e-3)
    assert row == pytest.approx(73.98472592317768, abs=1e-3)


def test_pixel_corner_round_trips_through_lonlat() -> None:
    lon, lat = _pixel_to_lonlat(777.0, 44.0, AOI_TRANSFORM)
    col, row = t14_labels.lonlat_to_pixel(lon, lat, AOI_TRANSFORM)
    assert col == pytest.approx(777.0, abs=1e-6)
    assert row == pytest.approx(44.0, abs=1e-6)


def test_rotated_transform_is_rejected() -> None:
    with pytest.raises(ValueError, match="rotated"):
        t14_labels.lonlat_to_pixel(0.0, 0.0, [4.78, 0.1, 0.0, 0.0, -4.78, 0.0])


@pytest.mark.parametrize(
    "crs",
    [{"type": "name", "properties": {"name": "EPSG:4326"}}, None],
    ids=["epsg4326", "no-crs"],
)
def test_udm_with_features_must_declare_crs84(tmp_path: Path, crs: dict | None) -> None:
    ring = [[-116.98, 32.61], [-116.97, 32.61], [-116.97, 32.62], [-116.98, 32.61]]
    feature = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [ring]},
    }
    doc: dict = {"type": "FeatureCollection", "features": [feature]}
    if crs is not None:
        doc["crs"] = crs
    path = _udm_path(tmp_path, "2018_01")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="crs"):
        t14_labels.cloud_mask(tmp_path, _row("2018_01", "2018_02"), (SIZE, SIZE))


def test_empty_udm_needs_no_crs(tmp_path: Path) -> None:
    path = _udm_path(tmp_path, "2018_01")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    row = _row("2018_01", "2018_02")
    cloud, missing, outside = t14_labels.cloud_mask(tmp_path, row, (SIZE, SIZE))
    assert not cloud.any() and missing == 1 and outside == 0


def test_udm_entirely_outside_the_window_is_counted_and_warned(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _udm(tmp_path, "2018_01", [(100, 100, 105, 105), (-20, 2, -10, 5)])
    with caplog.at_level(logging.WARNING, logger="lim_worker.t14_labels"):
        cloud, missing, outside = t14_labels.cloud_mask(
            tmp_path, _row("2018_01", "2018_02"), (SIZE, SIZE)
        )
    assert not cloud.any() and missing == 1 and outside == 2
    assert any("outside the image window" in r.getMessage() for r in caplog.records)


def test_partly_outside_udm_is_counted_without_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _udm(tmp_path, "2018_01", [(2, 2, 5, 5), (100, 100, 105, 105)])
    with caplog.at_level(logging.WARNING, logger="lim_worker.t14_labels"):
        cloud, missing, outside = t14_labels.cloud_mask(
            tmp_path, _row("2018_01", "2018_02"), (SIZE, SIZE)
        )
    assert int(cloud.sum()) == 9 and missing == 1 and outside == 1
    assert not [r for r in caplog.records if "outside the image window" in r.getMessage()]
