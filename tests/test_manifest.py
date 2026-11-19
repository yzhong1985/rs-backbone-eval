import json
from pathlib import Path

import pytest

from rs_backbone_eval.data.manifest import (
    GroupLeakageError,
    ManifestRowError,
    audit_manifest,
    require_no_group_leakage,
)


def sample(sample_id: str, group_id: str, split: str) -> dict:
    return {
        "task_id": "T3",
        "sample_id": sample_id,
        "group_id": group_id,
        "split": split,
        "image_paths": ["pre.tif", "during.tif", "post.tif"],
        "label_path": "label.tif",
        "sensor": "hls",
        "bands": ["BLUE", "GREEN", "RED", "NIR", "SWIR_1", "SWIR_2"],
        "timestamps": ["pre", "during", "post"],
        "temporal_mask": [True, True, True],
        "crs": "EPSG:32610",
        "transform": [30, 0, 0, 0, -30, 0],
        "gsd_m": 30,
        "checksum": "a" * 64,
    }


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_manifest_audit_accepts_grouped_split(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [sample("one", "fire-a", "train"), sample("two", "fire-b", "test")])

    audit = audit_manifest(path)

    assert audit.valid
    assert audit.sample_count == 2
    assert audit.split_counts == {"train": 1, "test": 1}


def test_manifest_audit_detects_group_leakage(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [sample("one", "fire-a", "train"), sample("two", "fire-a", "test")])

    audit = audit_manifest(path)

    assert not audit.valid
    assert audit.leakage == {"T3/fire-a": ["test", "train"]}


def test_manifest_audit_detects_temporal_shape_error(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    row = sample("one", "fire-a", "train")
    row["timestamps"] = ["pre"]
    write_manifest(path, [row])

    audit = audit_manifest(path)

    assert not audit.valid
    assert "equal lengths" in audit.errors[0]


def _t13(sample_id: str, group_id: str, split: str) -> dict:
    return {**sample(sample_id, group_id, split), "task_id": "T13"}


def test_require_no_group_leakage_refuses_a_group_in_two_splits(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [_t13("one", "sheet-1", "train"), _t13("two", "sheet-1", "validation")])

    with pytest.raises(GroupLeakageError, match="sheet-1 in train, validation"):
        require_no_group_leakage(path, "T13")


def test_require_no_group_leakage_passes_when_groups_are_disjoint(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path,
        [
            _t13("one", "sheet-1", "train"),
            _t13("two", "sheet-2", "validation"),
            _t13("three", "sheet-3", "test"),
        ],
    )

    require_no_group_leakage(path, "T13")


def test_require_no_group_leakage_ignores_other_tasks_in_a_shared_manifest(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path,
        [
            _t13("one", "sheet-1", "train"),
            sample("two", "fire-a", "train"),
            sample("three", "fire-a", "test"),
        ],
    )

    require_no_group_leakage(path, "T13")
    with pytest.raises(GroupLeakageError):
        require_no_group_leakage(path, "T3")


def test_require_no_group_leakage_fails_closed_on_a_malformed_row_of_the_task(
    tmp_path: Path,
) -> None:
    # The malformed train row shares sheet-1 with a test row. audit_manifest would skip
    # it and report no leak; the guard must refuse instead.
    path = tmp_path / "manifest.jsonl"
    broken = _t13("one", "sheet-1", "train")
    del broken["checksum"]
    write_manifest(path, [broken, _t13("two", "sheet-1", "test")])

    with pytest.raises(ManifestRowError, match="line 1"):
        require_no_group_leakage(path, "T13")


def test_require_no_group_leakage_ignores_a_malformed_row_of_another_task(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.jsonl"
    broken = sample("two", "fire-a", "train")
    del broken["checksum"]
    write_manifest(
        path,
        [_t13("one", "sheet-1", "train"), _t13("three", "sheet-2", "test"), broken],
    )

    require_no_group_leakage(path, "T13")


def test_require_no_group_leakage_fails_closed_on_an_unreadable_line(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(_t13("one", "sheet-1", "train")) + "\n{not json\n", encoding="utf-8")

    with pytest.raises(ManifestRowError, match="line 2"):
        require_no_group_leakage(path, "T13")
