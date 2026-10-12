import json
from pathlib import Path

from rs_backbone_eval.manifest import audit_manifest


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
