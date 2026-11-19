"""CPU tests for the T14 change-mask writer (LIM worker env, rasterio only).

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_t14.py -q
"""

from __future__ import annotations

import json
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
    mask, missing = t14_labels.change_mask(tmp_path, row)
    assert mask.shape == (SIZE, SIZE) and mask.dtype == np.uint8
    assert mask.sum() == 4
    assert mask[10:12, 10:12].all()
    assert mask[2:5, 2:5].sum() == 0
    assert missing == 2  # no UDM files in this fixture


def _udm(root: Path, stamp: str, boxes: list[tuple[int, int, int, int]]) -> None:
    stem = f"global_monthly_{stamp}_mosaic_{AOI}"
    _geojson(root / "raw/spacenet7/train" / AOI / "labels" / f"{stem}_UDM.geojson", boxes)


def test_cloud_from_either_date_is_255_even_over_a_building(tmp_path: Path) -> None:
    labels = tmp_path / "raw/spacenet7/train" / AOI / "labels_match_pix"
    base = f"global_monthly_{{}}_mosaic_{AOI}_Buildings.geojson"
    _geojson(labels / base.format("2018_01"), [])
    _geojson(labels / base.format("2018_02"), [(2, 2, 5, 5), (10, 10, 12, 12)])
    _udm(tmp_path, "2018_01", [(2, 2, 5, 5)])  # cloud at t1 over the t2 building
    _udm(tmp_path, "2018_02", [(10, 10, 11, 12)])  # cloud at t2 over half of the other one
    mask, missing = t14_labels.change_mask(tmp_path, _row("2018_01", "2018_02"))
    assert missing == 0
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
    mask, missing = t14_labels.change_mask(tmp_path, row)
    assert missing == 1
    assert int((mask == 255).sum()) == 0 and mask.sum() == 4
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    counts = t14_labels.build_masks(tmp_path, manifest)
    assert counts == {"written": 1, "skipped": 0, "udm_missing": 1, "cloud_px": 0}


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
        "cloud_px": 0,
    }
    assert t14_labels.build_masks(tmp_path, manifest) == {
        "written": 0,
        "skipped": 1,
        "udm_missing": 0,
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
