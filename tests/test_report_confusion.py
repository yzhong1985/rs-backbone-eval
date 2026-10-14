"""Tests for the grouped/paired confusion intervals in the report (R5 step 5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rs_backbone_eval.contracts import RunRecord
from rs_backbone_eval.evaluation.groups import GroupedSegmentation
from rs_backbone_eval.report import _confusion_section, build_markdown


def payload(run_id: str, model_id: str, *, seed: int = 42) -> dict:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "completed",
        "task_id": "T1",
        "model_id": model_id,
        "protocol_id": "full_finetune",
        "seed": seed,
        "dataset": {"dataset_id": "Sen1Floods11", "manifest_sha256": "b" * 64, "split": "test"},
        "input": {
            "sensor": "sentinel-2",
            "bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            "time_steps": 1,
            "preprocessing_id": "s2-six-band-v1",
        },
        "training": {"regime": "full_finetune", "head_id": "lite_spatial_head"},
        "metrics": [
            {
                "name": "water_iou",
                "value": 0.8,
                "split": "test",
                "direction": "maximize",
                "is_primary": True,
            }
        ],
        "artifacts": [{"kind": "confusion_matrix", "path": "confusion_per_scene.json"}],
        "tags": {"learning_rate": "1e-4", "encoder_input_px": "512"},
    }


def write_confusion(runs_root: Path, run_id: str, groups: dict[str, list[list[int]]]) -> None:
    directory = runs_root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "confusion_per_scene.json").write_text(
        json.dumps({"num_classes": 2, "groups": groups}), encoding="utf-8"
    )


def cell(model: str, run_id: str) -> dict:
    return {
        "track": "B1",
        "model": model,
        "task": "T1",
        "init": "pretrained",
        "label_fraction": "1.0",
        "runs": [run_id],
    }


def counts(tp: int, fp: int, fn: int, tn: int) -> list[list[int]]:
    # rows are ground truth, columns are predictions
    return [[tn, fp], [fn, tp]]


def test_grouped_segmentation_keeps_groups_separate() -> None:
    collector = GroupedSegmentation(num_classes=2, ignore_index=-1)
    collector.update(
        ["a", "b"],
        np.array([[[1, 1], [0, 0]], [[1, 1], [0, 0]]]),
        np.array([[[1, 1], [0, 0]], [[0, 0], [1, 1]]]),
    )
    groups = collector.group_confusion()
    assert set(groups) == {"a", "b"}
    assert groups["a"][1][1] == 2  # group a: both positives are correct
    assert groups["b"][1][1] == 0  # group b: every positive is missed
    assert groups["b"][1][0] == 2  # ...and predicted as background


def test_confusion_section_reports_grouped_and_paired_intervals(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    write_confusion(
        runs_root,
        "run-a",
        {"g0": counts(9, 1, 1, 9), "g1": counts(8, 2, 2, 8), "g2": counts(9, 1, 1, 9)},
    )
    write_confusion(
        runs_root,
        "run-b",
        {"g0": counts(7, 3, 3, 7), "g1": counts(6, 4, 4, 6), "g2": counts(7, 3, 3, 7)},
    )
    records = [
        RunRecord.model_validate(payload("run-a", "m_a")),
        RunRecord.model_validate(payload("run-b", "m_b")),
    ]
    matrix = {"cells": [cell("m_a", "run-a"), cell("m_b", "run-b")]}
    lines: list[str] = []
    _confusion_section(records, runs_root, matrix, lines.append)
    text = "\n".join(lines)
    assert "分组置信区间" not in text  # the heading is added by build_markdown
    assert "95% 区间" in text
    assert "m_a" in text and "m_b" in text
    assert "m_a − m_b" in text  # paired difference on the shared groups
    assert "| 3 |" in text  # three shared groups


def test_confusion_section_is_graceful_without_artifacts(tmp_path: Path) -> None:
    records = [RunRecord.model_validate(payload("run-a", "m_a"))]
    matrix = {"cells": [cell("m_a", "run-a")]}
    lines: list[str] = []
    _confusion_section(records, tmp_path / "runs", matrix, lines.append)
    assert any("逐组混淆矩阵" in line for line in lines)


def test_report_shows_phase_costs_and_optimizer_steps(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    directory = runs_root / "run-a"
    directory.mkdir(parents=True)
    record = payload("run-a", "m_a")
    record["training"].update(
        {
            "phases": {"train": 12.5, "eval": 3.0},
            "cache_bytes": 2_000_000_000,
            "optimizer_steps": 40,
        }
    )
    (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")
    config_root = Path(__file__).resolve().parents[1] / "configs"
    markdown = build_markdown(runs_root, tmp_path / "data", config_root, Path("."))
    assert "### 2.2.1 分阶段成本与缓存" in markdown
    assert "train=12.5" in markdown
    assert "2.00" in markdown  # cache gigabytes
    assert "| 40 |" in markdown  # optimizer steps column
