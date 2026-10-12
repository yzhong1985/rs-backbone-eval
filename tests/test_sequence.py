from pathlib import Path

import pytest

from rs_backbone_eval.contracts import ManifestSample
from rs_backbone_eval.data.sequence import SampleValidationError, SequenceDataset, validate_sample


def write_manifest(path: Path, rows: list[ManifestSample]) -> None:
    path.write_text("\n".join(row.model_dump_json() for row in rows) + "\n", encoding="utf-8")


def test_sequence_dataset_streams_validated_samples(tmp_path: Path, make_task, make_row) -> None:
    path = tmp_path / "manifest.jsonl"
    rows = [
        make_row("one", "fire-a", "train"),
        make_row("two", "fire-b", "validation"),
        make_row("three", "fire-c", "test"),
    ]
    write_manifest(path, rows)

    dataset = SequenceDataset(path, make_task())

    assert [sample.sample_id for sample in dataset] == ["one", "two", "three"]
    assert dataset.sample_count() == 3
    assert dataset.split_counts() == {"train": 1, "validation": 1, "test": 1}


def test_sequence_dataset_filters_by_split(tmp_path: Path, make_task, make_row) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [make_row("one", "fire-a", "train"), make_row("two", "fire-b", "test")])

    dataset = SequenceDataset(path, make_task())

    assert [sample.sample_id for sample in dataset.by_split("test")] == ["two"]


def test_sequence_dataset_rejects_contract_violation(tmp_path: Path, make_task, make_row) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, [make_row("one", "fire-a", "train", bands=["BLUE", "GREEN", "RED"])])

    dataset = SequenceDataset(path, make_task())

    with pytest.raises(SampleValidationError, match="bands"):
        list(dataset)


def test_validate_sample_reports_every_violation(make_task, make_row) -> None:
    sample = make_row(
        "one",
        "fire-a",
        "val",
        sensor="sentinel-2",
        image_paths=["only.tif"],
        timestamps=["pre"],
        temporal_mask=[True],
    )

    errors = validate_sample(sample, make_task())

    assert len(errors) == 3
    assert any("sensor" in error for error in errors)
    assert any("frames" in error for error in errors)
    assert any("split" in error for error in errors)


def test_valid_timestamps_skip_missing_dates(make_row) -> None:
    sample = make_row(
        "one",
        "fire-a",
        "train",
        temporal_mask=[True, False, True],
        timestamps=["pre", "during", "post"],
    )

    assert sample.time_count == 3
    assert sample.valid_timestamps == ["pre", "post"]
