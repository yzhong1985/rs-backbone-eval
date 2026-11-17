"""CPU-only tests for the T13 (WHU-OPT-SAR optical) task and manifest builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from rs_backbone_eval.data.datasets import builder_for
from rs_backbone_eval.data.datasets.sen1floods11 import ManifestBuildError
from rs_backbone_eval.data.datasets.whu_opt_sar import (
    CROP,
    DIVISION_FILE,
    VALIDATION_SHEETS,
    read_division,
    validation_sheets,
)
from rs_backbone_eval.data.whu_sheets import CRS, DEG_PER_PX_X, decode
from rs_backbone_eval.registry import Registry

ROOT = Path(__file__).resolve().parents[1]

# 12 real sheet names: 10 train + 2 test is enough to hold out 8 validation sheets
TRAIN = [
    "NH49E004021",
    "NH49E005014",
    "NH49E005023",
    "NH49E005024",
    "NH49E006011",
    "NH49E006022",
    "NH49E006024",
    "NH49E007013",
    "NH49E007023",
    "NH49E007024",
]
TEST = ["NH49E001013", "NH49E001014"]


def _task():
    return next(t for t in Registry(ROOT / "configs").tasks() if t.task_id == "T13")


def _fake_dataset(root: Path, train=TRAIN, test=TEST) -> Path:
    raw = root / "raw" / "whu-opt-sar"
    for sub in ("optical", "lbl"):
        (raw / sub).mkdir(parents=True)
    for name in [*train, *test]:
        (raw / "optical" / f"{name}.tif").write_bytes(b"img")
        (raw / "lbl" / f"{name}.tif").write_bytes(name.encode())
    text = "train:\n" + "".join(f"{n}.tif\n" for n in train)
    text += "\ntest:\n" + "".join(f"{n}.tif\n" for n in test)
    (raw / DIVISION_FILE).write_text(text, encoding="utf-8")
    return root


def test_task_registered_with_eight_classes_and_background_ignored() -> None:
    task = _task()
    assert task.dataset_id == "WHU-OPT-SAR"
    assert task.num_outputs == 8
    assert task.ignore_index == 0
    assert task.bands == ["RED", "GREEN", "BLUE", "NIR_BROAD"]


def test_division_parser_rejects_duplicates_and_orphans(tmp_path: Path) -> None:
    good = tmp_path / "ok.txt"
    good.write_text("train:\nNH49E004021.tif\ntest:\nNH49E001013.tif\n", encoding="utf-8")
    assert read_division(good) == {"NH49E004021": "train", "NH49E001013": "test"}

    dup = tmp_path / "dup.txt"
    dup.write_text("train:\nNH49E004021.tif\ntest:\nNH49E004021.tif\n", encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="twice"):
        read_division(dup)

    orphan = tmp_path / "orphan.txt"
    orphan.write_text("NH49E004021.tif\n", encoding="utf-8")
    with pytest.raises(ManifestBuildError, match="before any section"):
        read_division(orphan)


def test_validation_sheets_are_fixed_and_drawn_from_train() -> None:
    first = validation_sheets(TRAIN)
    assert first == validation_sheets(list(reversed(TRAIN)))
    assert len(first) == VALIDATION_SHEETS and first <= set(TRAIN)


def test_manifest_keeps_sheets_whole_and_georeferences_tiles(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path)
    samples = builder_for("WHU-OPT-SAR")(root, _task())

    split_of: dict[str, set[str]] = {}
    for sample in samples:
        split_of.setdefault(sample.group_id, set()).add(sample.split)
    # one split per sheet: no tile of a sheet leaks into another split
    assert all(len(splits) == 1 for splits in split_of.values())
    assert {g for g, s in split_of.items() if s == {"test"}} == set(TEST)
    assert sum(s == {"validation"} for s in split_of.values()) == VALIDATION_SHEETS

    per_sheet = [s for s in samples if s.group_id == "NH49E004021"]
    assert len(per_sheet) == 15 * 10  # 5556 x 3704 at 512 / 384, last tile clamped
    windows = {tuple(s.window) for s in per_sheet}
    assert (0, 0, CROP, CROP) in windows and (5556 - CROP, 3704 - CROP, CROP, CROP) in windows

    sheet = decode("NH49E004021")
    tile = next(s for s in per_sheet if s.window[:2] == [384, 768])
    assert tile.crs == CRS
    a, _, c, _, e, f = tile.transform
    assert a == pytest.approx(DEG_PER_PX_X)
    assert c == pytest.approx(sheet.west + 384 * DEG_PER_PX_X)
    assert f == pytest.approx(sheet.north + 768 * e)


def test_missing_label_is_an_error(tmp_path: Path) -> None:
    root = _fake_dataset(tmp_path)
    (root / "raw" / "whu-opt-sar" / "lbl" / f"{TEST[0]}.tif").unlink()
    with pytest.raises(ManifestBuildError, match="missing"):
        builder_for("WHU-OPT-SAR")(root, _task())
