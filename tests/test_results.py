import json
from pathlib import Path

import pytest

from rs_backbone_eval.contracts import ArtifactRecord, RunRecord
from rs_backbone_eval.results import RunNotFoundError, RunRepository


def run_payload(run_id: str = "T1-anysat-seed42") -> dict:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": "2026-10-12T08:00:00Z",
        "updated_at": "2026-10-12T09:00:00Z",
        "status": "completed",
        "task_id": "T1",
        "model_id": "anysat_base",
        "protocol_id": "frozen_probe",
        "seed": 42,
        "git_commit": "abc1234",
        "model_checkpoint_sha256": "a" * 64,
        "dataset": {
            "dataset_id": "Sen1Floods11",
            "version": "official",
            "manifest_sha256": "b" * 64,
            "split": "test",
        },
        "input": {
            "sensor": "sentinel-2",
            "bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            "time_steps": 1,
            "gsd_m": 10,
            "preprocessing_id": "s2-six-band-v1",
        },
        "training": {
            "regime": "frozen_probe",
            "head_id": "linear",
            "trainable_parameters": 1024,
            "total_parameters": 100001024,
            "epochs_completed": 50,
            "optimizer_steps": 320,
            "wall_time_seconds": 1800,
            "peak_gpu_memory_mb": 4096,
        },
        "metrics": [
            {
                "name": "water_iou",
                "value": 0.81,
                "split": "test",
                "direction": "maximize",
                "is_primary": True,
            }
        ],
        "curves": [
            {"name": "loss", "split": "train", "step": 1, "value": 0.9},
            {"name": "loss", "split": "train", "step": 2, "value": 0.6},
        ],
        "artifacts": [],
        "tags": {"track": "smoke"},
    }


def write_run(root: Path, payload: dict) -> Path:
    directory = root / payload["run_id"]
    directory.mkdir(parents=True)
    path = directory / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_run_contract_exposes_primary_metric() -> None:
    record = RunRecord.model_validate(run_payload())
    assert record.primary_metric() is not None
    assert record.primary_metric().value == 0.81


def test_artifact_contract_rejects_path_escape() -> None:
    with pytest.raises(ValueError):
        ArtifactRecord(kind="preview", path="../secret.png")


def test_repository_lists_valid_runs_and_reports_invalid(tmp_path: Path) -> None:
    write_run(tmp_path, run_payload())
    invalid = tmp_path / "broken"
    invalid.mkdir()
    (invalid / "run.json").write_text("{}", encoding="utf-8")

    repository = RunRepository(tmp_path)
    records, errors = repository.list_records()

    assert [record.run_id for record in records] == ["T1-anysat-seed42"]
    assert len(errors) == 1
    assert repository.summary()["invalid_run_count"] == 1


def test_repository_blocks_artifact_escape(tmp_path: Path) -> None:
    write_run(tmp_path, run_payload())
    with pytest.raises(RunNotFoundError):
        RunRepository(tmp_path).artifact_path("T1-anysat-seed42", "../../outside")
