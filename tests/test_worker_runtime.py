"""Tests for the shared worker runtime: phase costs and confusion persistence."""

from __future__ import annotations

from pathlib import Path

from rs_backbone_eval.contracts import TrainingRecord
from rs_backbone_eval.registry import MetricSpec, ProtocolSpec, TaskSpec
from rs_backbone_eval.worker_runtime import RunWriter, load_confusion


def task_spec() -> TaskSpec:
    return TaskSpec(
        task_id="T1",
        application="Flood",
        dataset_id="Sen1Floods11",
        family="segmentation",
        sensor="sentinel-2",
        time_steps=1,
        bands=["BLUE", "GREEN"],
        num_outputs=2,
        primary_metric=MetricSpec(name="water_iou", direction="maximize"),
        official_config_url="https://example.invalid",
    )


def protocol_spec() -> ProtocolSpec:
    return ProtocolSpec(
        protocol_id="frozen_probe",
        regime="frozen_probe",
        train_backbone=False,
        train_head=True,
        train_input_adapter=False,
        seeds=[42],
    )


def test_training_record_carries_phase_costs() -> None:
    record = TrainingRecord(
        regime="frozen_probe",
        head_id="lite_spatial_head",
        phases={"extraction": 12.5, "head_training": 3.0},
        cache_bytes=2048,
    )
    assert record.phases["extraction"] == 12.5
    assert record.cache_bytes == 2048


def test_add_confusion_roundtrips(tmp_path: Path) -> None:
    writer = RunWriter(
        tmp_path,
        run_id="run-conf",
        task_spec=task_spec(),
        protocol_spec=protocol_spec(),
        model_id="anysat_base",
        seed=42,
        manifest_sha256="a" * 64,
    )
    writer.add_confusion({"g2": [[1, 0], [0, 1]], "g1": [[2, 0], [0, 2]]}, num_classes=2)

    assert (tmp_path / "confusion_per_scene.json").is_file()
    assert any(artifact.kind == "confusion_matrix" for artifact in writer.record.artifacts)

    num_classes, groups = load_confusion(tmp_path / "confusion_per_scene.json")
    assert num_classes == 2
    assert [group for group, _ in groups] == ["g1", "g2"]
    assert groups[0][1] == [[2, 0], [0, 2]]


def test_random_init_requested_reads_the_environment(monkeypatch) -> None:
    from rs_backbone_eval.repro import random_init_requested

    monkeypatch.delenv("RS_EVAL_INIT", raising=False)
    assert random_init_requested() is False
    monkeypatch.setenv("RS_EVAL_INIT", "random")
    assert random_init_requested() is True
