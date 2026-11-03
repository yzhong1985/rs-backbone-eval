"""Synthetic end-to-end checks for the AEF extraction core (round 5 review).

Skipped where rasterio is absent (root environment): raster raster IO lives in
the worker environments, consistent with test_isaid_tiles.py. CPU only, no
external data - every input is written into ``tmp_path``.

Covers (review 2026-11-03, rounds 4-5):

* a fresh extract() must work end to end (round 5 P1: ``np.savez`` appended
  ``.npz`` to the tmp name and the atomic rename crashed);
* candidates must be gated by ``utm_zone`` even when numeric UTM bboxes of
  different zones overlap (earlier round P1, in-place verified);
* a patch straddling two COGs is mosaicked, not overwritten;
* ``_cache_ok`` rejects an unusable npz w/ sidecar pair... well: a wrong
  shape/dtype npz plus a non-JSON sidecar must be rejected and recomputed;
* outputs carry the provenance fields used later by the manifest audit.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rasterio")
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "aef_extract_pastis.py"
PATCHES = ("1", "2")
SIZE = 141  # pixels at 10 m -> 1410 m, fully covers a 1 m SPOT patch grid


@pytest.fixture()
def out_dir(tmp_path):
    out = tmp_path / "features" / "2019"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_module(tmp_path: Path):
    spec = importlib.util.spec_from_file_location("aef_extract_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PASTIS = tmp_path / "PASTIS-HD"
    module.COGS = tmp_path / "aef-cogs"
    return module


def write_raster(path: Path, *, crs: bool, count: int, value: int, west, north, pixel: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((count, SIZE, SIZE), value, dtype=np.int8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=SIZE,
        height=SIZE,
        count=count,
        dtype="int8",
        nodata=-128,
        transform=from_origin(west, north, pixel, pixel),
        crs=crs,
    ) as dataset:
        dataset.write(data)


def write_spot(path: Path, west: float, north: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((3, 1280, 1280), 7, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1280,
        height=1280,
        count=3,
        dtype="uint8",
        transform=from_origin(west, north, 1.0, 1.0),
        crs="EPSG:32630",  # the value the CRS assert compares against is
        # driven by centre_utm, so this is a placeholder
    ) as dataset:
        dataset.write(data)


def make_meta(module, tmp_path: Path) -> list[dict]:
    """Two patches: one clean 30N, one straddling two 31N COGs."""
    root = module.COGS / "2019"
    # Zone 30N COG holding value 42 over its own extent.
    write_raster(root / "cog30.tiff", crs=rasterio.crs.CRS.from_epsg(32630), count=64, value=42,
                 west=100400, north=4400995, pixel=10)
    # Zone 31N decoy whose *numeric* bbox covers the 30N area: the zone gate
    # must reject it (the value 80 would overwrite the correct features).
    write_raster(root / "decoy31.tiff", crs=rasterio.crs.CRS.from_epsg(32631), count=64, value=80,
                 west=100400, north=4400995, pixel=10)
    # 31N mosaic pair with a 600 m overlap; values 43 and 44.
    write_raster(root / "c31a.tiff", crs=rasterio.crs.CRS.from_epsg(32631), count=64, value=43,
                 west=102000, north=4400995, pixel=10)
    write_raster(root / "c31b.tiff", crs=rasterio.crs.CRS.from_epsg(32631), count=64, value=44,
                 west=102850, north=4400995, pixel=10)
    # Numeric-overlap decoy in 30N for patch 2: must also be rejected.
    write_raster(root / "decoy30.tiff", crs=rasterio.crs.CRS.from_epsg(32630), count=64, value=80,
                 west=102000, north=4400995, pixel=10)

    spot_dir = module.PASTIS / "DATA_SPOT/PASTIS_SPOT6_RVB_1M00_2019"
    write_spot(spot_dir / "SPOT6_RVB_1M00_2019_1.tif", west=100500, north=4400995)
    write_spot(spot_dir / "SPOT6_RVB_1M00_2019_2.tif", west=102000, north=4400995)

    rows = [
        {"year": 2019, "utm_zone": "30N", "path": "gs://x/cog30.tiff", "utm_west": 100400,
         "utm_south": 4399585, "utm_east": 101810, "utm_north": 4400995},
        {"year": 2019, "utm_zone": "31N", "path": "gs://x/decoy31.tiff", "utm_west": 100400,
         "utm_south": 4399585, "utm_east": 101810, "utm_north": 4400995},
        {"year": 2019, "utm_zone": "31N", "path": "gs://x/c31a.tiff", "utm_west": 102000,
         "utm_south": 4399585, "utm_east": 103410, "utm_north": 4400995},
        {"year": 2019, "utm_zone": "31N", "path": "gs://x/c31b.tiff", "utm_west": 102850,
         "utm_south": 4399585, "utm_east": 104260, "utm_north": 4400995},
        {"year": 2019, "utm_zone": "30N", "path": "gs://x/decoy30.tiff", "utm_west": 102000,
         "utm_south": 4399585, "utm_east": 103410, "utm_north": 4400995},
    ]

    def geometry(_self, patch):
        return [{"ID_PATCH": 1}, {"ID_PATCH": 2}]  # replaced by centre_utm stub

    zones = {1: (100500.0, 4399720.0, "30N"), 2: (102005.0, 4399720.0, "31N")}
    module.centre_utm = lambda patch: zones[int(patch["properties"]["ID_PATCH"])]  # type: ignore[assignment]
    return rows


def patch_dict(pid: int) -> dict:
    return {
        "properties": {
            "ID_PATCH": pid,
            "TILE": f"t{pid:05d}",
            "Fold": 1,
            "dates-S2": {"0": 20190715},
        },
        "geometry": {"coordinates": [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]},
    }


def read_features(path: Path):
    with np.load(path) as archive:
        return np.asarray(archive["features"]), np.asarray(archive["valid"])


def test_fresh_extract_zone_gate_and_mosaic(tmp_path, out_dir):
    module = load_module(tmp_path)
    rows = make_meta(module, tmp_path)

    result = module.extract(patch_dict(1), 2019, rows, out_dir)
    assert result.get("error") is None, result
    features, valid = read_features(out_dir / "1.npz")
    assert features.shape == (64, 128, 128)
    assert valid.all()
    # The 31N decoy would have written 80s onto the grid.
    assert features.min() == 42.0 and features.max() == 42.0

    result = module.extract(patch_dict(2), 2019, rows, out_dir)
    assert result.get("error") is None, result
    features, valid = read_features(out_dir / "2.npz")
    assert valid.all()
    # Left half lives in c31a (43), the overlap/right region in c31b (44).
    # x = 102005 + 10k, so the c31b-only region starts at k=85 (x >= 102850).
    left = features[:, :, :85]
    right = features[:, :, 85:]
    assert np.all(left == 43.0)
    assert np.all(right == 44.0)

    meta = json.loads((out_dir / "2.json").read_text(encoding="utf-8"))
    assert meta["quantization"] == "raw_aef_int8_as_float16"
    assert meta["script_commit"] and meta["index_sha256"]
    assert set(meta["cogs"]) == {"c31a.tiff", "c31b.tiff"}
    assert sorted(meta["cog_sha256"]) == ["c31a.tiff", "c31b.tiff"]
    assert meta["valid_share"] == 1.0
    # No leftover temp artefacts from the atomic writes.
    assert not list(out_dir.glob("*.tmp"))


def test_idempotent_skip_and_corrupt_cache_recompute(tmp_path, out_dir):
    module = load_module(tmp_path)
    rows = make_meta(module, tmp_path)
    module.extract(patch_dict(1), 2019, rows, out_dir)

    assert module.extract(patch_dict(1), 2019, rows, out_dir) == {"patch": 1, "skipped": True}

    # Wrong-shaped features with a non-JSON sidecar used to pass validation
    # (review 2026-11-03, round 5, P2); now they must be re-extracted.
    with open(out_dir / "1.npz", "wb") as handle:
        np.savez(
            handle,
            features=np.zeros((1,), dtype=np.float16),
            valid=np.zeros((1,), np.float32),
        )
    (out_dir / "1.json").write_text("not json", encoding="utf-8")

    assert module.extract(patch_dict(1), 2019, rows, out_dir) != {"patch": 1, "skipped": True}
    features, valid = read_features(out_dir / "1.npz")
    assert features.shape == (64, 128, 128) and features.max() == 42.0
    json.loads((out_dir / "1.json").read_text(encoding="utf-8"))


def test_missing_zone_candidate_fails_closed(tmp_path, out_dir):
    module = load_module(tmp_path)
    rows = [row for row in make_meta(module, tmp_path) if row["path"] != "gs://x/cog30.tiff"]
    result = module.extract(patch_dict(1), 2019, rows, out_dir)
    assert "error" in result  # numeric overlap alone must NOT satisfy the patch
