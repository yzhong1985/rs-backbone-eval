"""CPU-only tests for the T14 (SpaceNet 7 change) manifest builder.

The rasters are synthetic TIFF headers with only the tags the builder reads;
no pixel data is needed and no rasterio is imported.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from rs_backbone_eval.data.datasets import builder_for
from rs_backbone_eval.data.datasets.sen1floods11 import ManifestBuildError
from rs_backbone_eval.data.datasets.spacenet7 import (
    TEST_AOIS,
    VALIDATION_AOIS,
    aoi_split,
    read_grid,
)
from rs_backbone_eval.data.sequence import validate_sample
from rs_backbone_eval.registry import Registry

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = (-8_900_000.0, 1_500_000.0)


def _task():
    return next(t for t in Registry(ROOT / "configs").tasks() if t.task_id == "T14")


def write_tif(path: Path, width: int, height: int, epsg: int = 3857) -> None:
    """Little-endian classic TIFF with the tags the builder reads (no pixel data)."""
    entries = [
        (256, 4, 1, struct.pack("<I", width)),
        (257, 4, 1, struct.pack("<I", height)),
        (33550, 12, 3, struct.pack("<3d", 4.78, 4.78, 0.0)),
        (33922, 12, 6, struct.pack("<6d", 0.0, 0.0, 0.0, *ORIGIN, 0.0)),
        (34735, 3, 8, struct.pack("<8H", 1, 1, 0, 1, 3072, 0, 1, epsg)),
    ]
    count_bytes = 2 + 12 * len(entries) + 4
    data_at = 8 + count_bytes
    ifd = struct.pack("<H", len(entries))
    blob = b""
    for tag, kind, count, value in entries:
        if len(value) <= 4:
            field = value.ljust(4, b"\0")
        else:
            field = struct.pack("<I", data_at + len(blob))
            blob += value
        ifd += struct.pack("<HHI", tag, kind, count) + field
    ifd += struct.pack("<I", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"II*\0" + struct.pack("<I", 8) + ifd + blob)


def make_aoi(root: Path, aoi: str, months: list[tuple[int, int]], **shape) -> None:
    images = root / "raw" / "spacenet7" / "train" / aoi / "images"
    for year, month in months:
        name = f"global_monthly_{year:04d}_{month:02d}_mosaic_{aoi}.tif"
        write_tif(
            images / name, shape.get("width", 8), shape.get("height", 6), shape.get("epsg", 3857)
        )


def _pad(root: Path) -> None:
    """Twelve valid AOIs so the split (6 validation + 6 test) can be drawn."""
    for aoi in _aois(12):
        make_aoi(root, "pad-" + aoi, [(2018, 1), (2018, 2)])


def _aois(count: int) -> list[str]:
    return [f"L15-{i:04d}E-0000N_0_0_13" for i in range(count)]


def test_reads_grid_from_tags(tmp_path: Path) -> None:
    path = tmp_path / "a.tif"
    write_tif(path, 1024, 1023)
    (width, height), crs, affine = read_grid(path)
    assert (width, height) == (1024, 1023) and crs == "EPSG:3857"
    assert affine == pytest.approx([4.78, 0.0, ORIGIN[0], 0.0, -4.78, ORIGIN[1]])


def test_builds_consecutive_pairs_with_aoi_split(tmp_path: Path) -> None:
    task = _task()
    aois = _aois(14)
    for aoi in aois:
        make_aoi(tmp_path, aoi, [(2018, 1), (2018, 2), (2018, 3)])
    samples = builder_for("SpaceNet-7")(tmp_path, task)
    assert len(samples) == 14 * 2
    by_split = {
        s: {x.group_id for x in samples if x.split == s} for s in ("train", "validation", "test")
    }
    assert len(by_split["validation"]) == VALIDATION_AOIS
    assert len(by_split["test"]) == TEST_AOIS
    assert by_split["train"] == set(aois) - by_split["validation"] - by_split["test"]
    sample = next(s for s in samples if s.sample_id.endswith("2018-02_2018-03"))
    assert sample.image_paths == [
        "raw/spacenet7/train/"
        + sample.group_id
        + "/images/global_monthly_2018_02_mosaic_"
        + sample.group_id
        + ".tif",
        "raw/spacenet7/train/"
        + sample.group_id
        + "/images/global_monthly_2018_03_mosaic_"
        + sample.group_id
        + ".tif",
    ]
    assert sample.timestamps == ["2018-02", "2018-03"]
    assert sample.label_path == f"derived/spacenet7-change-v2/{sample.group_id}/2018-02_2018-03.tif"
    assert sample.crs == "EPSG:3857" and sample.window == [0, 0, 8, 6]
    assert sample.gsd_m == pytest.approx(4.78) and sample.gsd_measured_m is not None
    assert validate_sample(sample, task) == []


def test_pairs_across_a_month_gap_are_skipped(tmp_path: Path) -> None:
    _pad(tmp_path)
    make_aoi(tmp_path, _aois(1)[0], [(2018, 1), (2018, 2), (2018, 4), (2018, 5)])
    samples = builder_for("SpaceNet-7")(tmp_path, _task())
    own = [s.timestamps for s in samples if s.group_id == _aois(1)[0]]
    assert own == [["2018-01", "2018-02"], ["2018-04", "2018-05"]]


def test_wrong_crs_is_rejected(tmp_path: Path) -> None:
    _pad(tmp_path)
    make_aoi(tmp_path, _aois(1)[0], [(2018, 1), (2018, 2)], epsg=4326)
    with pytest.raises(ManifestBuildError, match="EPSG:3857"):
        builder_for("SpaceNet-7")(tmp_path, _task())


def test_mismatched_grid_within_an_aoi_is_rejected(tmp_path: Path) -> None:
    _pad(tmp_path)
    aoi = _aois(1)[0]
    make_aoi(tmp_path, aoi, [(2018, 1)])
    make_aoi(tmp_path, aoi, [(2018, 2)], width=9)
    with pytest.raises(ManifestBuildError, match="share one grid"):
        builder_for("SpaceNet-7")(tmp_path, _task())


def test_aoi_split_is_fixed_and_needs_enough_aois() -> None:
    aois = _aois(20)
    first = aoi_split(aois)
    assert first == aoi_split(list(reversed(aois)))
    assert sum(v == "validation" for v in first.values()) == VALIDATION_AOIS
    assert sum(v == "test" for v in first.values()) == TEST_AOIS
    with pytest.raises(ManifestBuildError):
        aoi_split(_aois(VALIDATION_AOIS + TEST_AOIS))
