from pathlib import Path

from rs_backbone_eval.contracts import ManifestSample
from rs_backbone_eval.data.manifest import audit_task_manifest, manifest_sha256, write_manifest


def test_write_manifest_is_deterministic(tmp_path: Path, make_task, make_row) -> None:
    rows: list[ManifestSample] = [
        make_row("two", "fire-b", "test"),
        make_row("one", "fire-a", "train"),
    ]
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    digest_one = write_manifest(first, rows)
    digest_two = write_manifest(second, list(reversed(rows)))

    assert digest_one == digest_two
    assert digest_one == manifest_sha256(first)
    assert first.read_text() == second.read_text()
    assert '"sample_id":"one"' in first.read_text().splitlines()[0]


def test_task_audit_accepts_valid_manifest(tmp_path: Path, make_task, make_row) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path,
        [
            make_row("one", "fire-a", "train"),
            make_row("two", "fire-b", "validation"),
            make_row("three", "fire-c", "test"),
        ],
    )

    audit = audit_task_manifest(path, make_task())

    assert audit.valid
    assert audit.sample_count == 3
    assert audit.split_counts == {"train": 1, "validation": 1, "test": 1}


def test_task_audit_rejects_contract_violations(tmp_path: Path, make_task, make_row) -> None:
    path = tmp_path / "manifest.jsonl"
    rows: list[ManifestSample] = [
        make_row("one", "fire-a", "train"),
        make_row("two", "fire-b", "validation", bands=["BLUE", "GREEN", "RED"]),
        make_row("three", "fire-c", "test", sensor="sentinel-2"),
        make_row(
            "four",
            "fire-d",
            "test",
            image_paths=["only.tif"],
            timestamps=["pre"],
            temporal_mask=[True],
        ),
        make_row("five", "fire-e", "val"),
        make_row("one", "fire-a", "train"),
    ]
    path.write_text("\n".join(row.model_dump_json() for row in rows) + "\n", encoding="utf-8")

    audit = audit_task_manifest(path, make_task())

    assert not audit.valid
    assert audit.duplicate_samples == ["one"]
    assert len(audit.band_mismatches) == 1
    assert len(audit.sensor_mismatches) == 1
    assert len(audit.time_step_mismatches) == 1
    assert len(audit.unknown_splits) == 1
    assert audit.unknown_splits[0].endswith("five: val")
    assert audit.error_count > 0


def test_task_audit_requires_train_split(tmp_path: Path, make_task, make_row) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [make_row("one", "fire-a", "test")])

    audit = audit_task_manifest(path, make_task())

    assert not audit.valid
    assert audit.missing_splits == ["train"]


def test_task_audit_rejects_foreign_task_rows(tmp_path: Path, make_task, make_row) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [make_row("one", "fire-a", "train", task_id="T1")])

    audit = audit_task_manifest(path, make_task())

    assert not audit.valid
    assert audit.sample_count == 0
    assert "does not match" in audit.schema_errors[0]


def test_task_audit_detects_group_leakage(tmp_path: Path, make_task, make_row) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(
        path,
        [
            make_row("one", "fire-a", "train"),
            make_row("two", "fire-a", "test"),
        ],
    )

    audit = audit_task_manifest(path, make_task())

    assert not audit.valid
    assert audit.leakage == {"fire-a": ["test", "train"]}
