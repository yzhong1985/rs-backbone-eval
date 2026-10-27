"""Tests for scripts/rederive_metrics.py.

Review 2026-10-27: a multi-class run whose confusion matrix has a class absent
from both ground truth and prediction stored ``null`` metrics, and the whole
``run.json`` became unreadable. The counts are still in the artifact, so the
record must be recoverable without retraining.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from rs_backbone_eval.results import RunRepository

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rederive_metrics", ROOT / "scripts" / "rederive_metrics.py"
)
assert SPEC and SPEC.loader
rederive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rederive)

# Class 0 is the ignored class: it never appears in the ground truth or the
# prediction, so its IoU is 0/0 -- exactly the T8 situation.
COUNTS = [
    [0, 0, 0, 0],
    [0, 6, 2, 0],
    [0, 1, 4, 1],
    [0, 0, 1, 5],
]


def _payload(run_id: str = "T8-clay-seed17", counts: list[list[int]] | None = None) -> dict:
    computed = rederive.recompute(np.asarray(counts or COUNTS, dtype=np.int64))
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": "2026-10-26T17:24:04Z",
        "updated_at": "2026-10-26T18:16:12Z",
        "status": "completed",
        "task_id": "T8",
        "model_id": "clay_v1_5",
        "protocol_id": "frozen_probe",
        "seed": 17,
        "git_commit": "abc1234",
        "model_checkpoint_sha256": "a" * 64,
        "dataset": {
            "dataset_id": "Five-Billion-Pixels",
            "manifest_sha256": "b" * 64,
            "split": "all",
        },
        "input": {
            "sensor": "gf-2",
            "bands": ["BLUE", "GREEN", "RED", "NIR_BROAD"],
            "time_steps": 1,
            "preprocessing_id": "gf2_train_stats_four_band",
        },
        "training": {
            "regime": "frozen_probe",
            "head_id": "lite_spatial_head",
            "trainable_parameters": 1450521,
            "total_parameters": 312893465,
            "epochs_completed": 50,
            "optimizer_steps": 0,
            "wall_time_seconds": 2661.9,
            "peak_gpu_memory_mb": 1291.5,
        },
        "metrics": [
            {
                "name": "mean_iou",
                "value": computed["mean_iou"],
                "split": "test",
                "direction": "maximize",
                "is_primary": True,
            },
            {
                "name": "macro_f1",
                "value": computed["macro_f1"],
                "split": "test",
                "direction": "maximize",
                "is_primary": False,
            },
            {
                "name": "frequency_weighted_iou",
                "value": None,
                "split": "test",
                "direction": "maximize",
                "is_primary": False,
            },
            {
                "name": "iou",
                "value": None,
                "class_name": "class_0",
                "split": "test",
                "direction": "maximize",
                "is_primary": False,
            },
        ],
        "artifacts": [{"kind": "confusion_matrix", "path": "confusion_per_scene.json"}],
        "tags": {"updated_by": "clay_worker"},
    }


def _write_run(root: Path, payload: dict, counts: list[list[int]] | None = None) -> Path:
    counts = counts or COUNTS
    run_dir = root / payload["run_id"]
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(json.dumps(payload), encoding="utf-8")
    artifact = {"num_classes": len(counts), "groups": {"scene-1": counts}}
    (run_dir / "confusion_per_scene.json").write_text(json.dumps(artifact), encoding="utf-8")
    return run_dir


def test_record_with_null_metrics_is_unreadable(tmp_path: Path) -> None:
    _write_run(tmp_path, _payload())

    records, errors = RunRepository(tmp_path).list_records()

    assert records == []
    assert len(errors) == 1
    assert "Input should be a valid number" in errors[0]["error"]


def test_rederive_recovers_the_record(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, _payload())

    assert rederive.main([str(tmp_path)]) == 0

    records, errors = RunRepository(tmp_path).list_records()
    assert errors == []
    assert len(records) == 1
    computed = rederive.recompute(rederive.load_counts(run_dir / "confusion_per_scene.json"))
    scored = {
        metric.name: metric.value for metric in records[0].metrics if metric.class_name is None
    }
    assert scored["frequency_weighted_iou"] == pytest.approx(computed["frequency_weighted_iou"])
    assert scored["mean_iou"] == pytest.approx(computed["mean_iou"])
    # The undefined class is dropped instead of stored as a null.
    assert all(metric.class_name != "class_0" for metric in records[0].metrics)
    assert (run_dir / f"run.json.pre-rederive-{rederive.date.today().isoformat()}").is_file()


def test_rederive_refuses_when_the_primary_disagrees(tmp_path: Path) -> None:
    payload = _payload()
    payload["metrics"][0]["value"] = 0.99  # the artifact cannot support this
    run_dir = _write_run(tmp_path, payload)
    before = (run_dir / "run.json").read_text(encoding="utf-8")

    assert rederive.main([str(tmp_path)]) == 1
    assert (run_dir / "run.json").read_text(encoding="utf-8") == before


def test_rederive_is_a_no_op_on_a_consistent_record(tmp_path: Path) -> None:
    _write_run(tmp_path, _payload())
    assert rederive.main([str(tmp_path)]) == 0
    run_json = tmp_path / "T8-clay-seed17" / "run.json"
    before = run_json.read_text(encoding="utf-8")

    assert rederive.main([str(tmp_path), "--all"]) == 0
    assert run_json.read_text(encoding="utf-8") == before


def test_every_emitted_segmentation_metric_is_recompute_aware() -> None:
    """A new stored metric must be teachable to the rederive tool (review 2026-10-27)."""
    from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
    from rs_backbone_eval.evaluation.segmentation import binary_metrics, multiclass_metrics

    binary = ConfusionMatrix(num_classes=2)
    binary.update([0, 1, 1], [0, 1, 0])
    multi = ConfusionMatrix(num_classes=3)
    multi.update([0, 1, 2, 2], [0, 1, 1, 2])

    emitted = set()
    for values in (binary_metrics(binary), multiclass_metrics(multi)):
        for key in values:
            emitted.add(key.split("/", 1)[0])

    assert emitted <= rederive.CONFUSION_METRICS, (
        "metrics emitted by the evaluator but unknown to rederive_metrics: "
        f"{sorted(emitted - rederive.CONFUSION_METRICS)}"
    )
