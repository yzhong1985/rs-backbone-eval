"""iSAID tiling + manifest tests (P2, review 2026-10-29).

The bug these pin: an earlier tiling formula dropped 418/1411 train images and
covered only ~80% of the pixels, which would have silently truncated training
and broken whole-image evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rs_backbone_eval.data.datasets import builder_for
from rs_backbone_eval.data.tiling import (
    tile_origins,
    tiles_covering,
    tiles_padded,
    valid_fraction,
)
from rs_backbone_eval.registry import TaskSpec


def test_tiles_cover_every_pixel() -> None:
    """Every pixel belongs to at least one tile, including the last edge band."""
    width, height, crop, stride = 2500, 1300, 1024, 512
    covered = [[False] * width for _ in range(height)]
    for x, y in tiles_covering(width, height, crop, stride):
        for row in range(y, min(y + crop, height)):
            for col in range(x, min(x + crop, width)):
                covered[row][col] = True
    assert all(all(row) for row in covered)


def test_small_images_get_one_padded_tile() -> None:
    """An image below the crop is padded whole, never dropped."""
    assert tile_origins(387, 1024, 512) == [0]
    assert tiles_padded(387, 278, 1024, 512) == 1
    share = valid_fraction(387, 278, 1024, 512, (0, 0))
    assert 0.09 < share < 0.11


def test_padded_tile_count_matches_the_audit() -> None:
    """The formula the P0-a audit uses (ceil, padded) is the builder's formula."""
    # ceil((3875-1024)/512)+1 = 7 columns, ceil((5502-1024)/512)+1 = 10 rows
    assert tiles_padded(3875, 5502, 1024, 512) == 7 * 10
    assert tiles_padded(1, 1, 1024, 512) == 1


def _task() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "task_id": "T9",
            "application": "Aerial",
            "dataset_id": "iSAID",
            "family": "segmentation",
            "sensor": "aerial-rgb",
            "time_steps": 1,
            "bands": ["RED", "GREEN", "BLUE"],
            "num_outputs": 16,
            "ignore_index": 255,
            "primary_metric": {"name": "mean_iou", "direction": "maximize"},
            "official_config_url": "https://example.invalid/isaid",
        }
    )


def write_png(path: Path, width: int, height: int) -> None:
    """Minimal valid PNG header; the builder only reads the IHDR size."""
    import struct
    import zlib

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = b"IHDR" + ihdr
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(ihdr))
        + chunk
        + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    )


@pytest.fixture()
def isaid_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    raw = root / "raw" / "isaid"
    masks = root / "derived" / "isaid" / "masks"
    for split in ("train", "val"):
        (raw / split).mkdir(parents=True)
        (raw / "dota" / "images" / split).mkdir(parents=True)
        (masks / split / "images").mkdir(parents=True)
    # one odd-sized image (below the crop) and one that needs an edge tile
    for split, name, size in (
        ("train", "P0000.png", (300, 200)),
        ("train", "P0001.png", (1600, 1100)),
    ):
        write_png(raw / "dota" / "images" / split / name, *size)
        write_png(masks / split / "images" / f"{Path(name).stem}_instance_color_RGB.png", *size)
    payload = {
        "images": [
            {"id": 0, "file_name": "P0000.png", "seg_file_name": "P0000_instance_color_RGB.png"},
            {"id": 1, "file_name": "P0001.png", "seg_file_name": "P0001_instance_color_RGB.png"},
        ],
        "categories": [],
        "annotations": [],
    }
    (raw / "train" / "iSAID_train.json").write_text(json.dumps(payload), encoding="utf-8")
    (raw / "val" / "iSAID_val.json").write_text(
        json.dumps({"images": [], "categories": [], "annotations": []}), encoding="utf-8"
    )
    return root


def test_builder_emits_tiles_with_group_and_valid_fraction(isaid_root: Path) -> None:
    samples = builder_for("iSAID")(isaid_root, _task())

    # 300x200 -> one padded tile; 1600x1100 -> 3 columns x 2 rows with padding
    assert len(samples) == 1 + 6
    small = [s for s in samples if s.group_id == "P0000"]
    assert len(small) == 1
    assert small[0].window == [0, 0, 1024, 1024]
    assert 0.05 < small[0].valid_fraction < 0.07
    assert small[0].split == "train"
    # tiles of one source image share a group, so holdout-manifest cannot leak
    assert {s.group_id for s in samples} == {"P0000", "P0001"}
    assert all(s.label_path.startswith("derived/isaid/masks/") for s in samples)
    assert all(len(s.checksum) == 64 for s in samples)
