"""Tests for the grouped/paired confusion intervals in the report (R5 step 5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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


def test_confusion_section_uses_t4_mean_iou_not_the_foreground_iou(tmp_path: Path) -> None:
    # T4's headline metric is the two-class mean IoU; T1/T2 keep the foreground
    # IoU. The classes-per-matrix test alone cannot tell them apart.
    runs_root = tmp_path / "runs"
    write_confusion(runs_root, "t4-run", {"g0": counts(8, 1, 1, 90)})
    record = payload("t4-run", "m_t4")
    record["task_id"] = "T4"
    record["metrics"] = [
        {
            "name": "mean_iou",
            "value": 0.8891,
            "split": "test",
            "direction": "maximize",
            "is_primary": True,
        },
        {
            "name": "foreground_iou",
            "value": 0.8,
            "split": "test",
            "direction": "maximize",
            "is_primary": False,
        },
    ]
    cell = {
        "track": "B2",
        "model": "m_t4",
        "task": "T4",
        "init": "pretrained",
        "label_fraction": "1.0",
        "runs": ["t4-run"],
    }
    lines: list[str] = []
    _confusion_section(
        [RunRecord.model_validate(record)], runs_root, {"cells": [cell]}, lines.append
    )
    text = "\n".join(lines)
    assert "| mean_iou |" in text
    assert "0.8891" in text  # (0.8 + 0.9783) / 2
    # The headline interval must stay mean_iou; the background-excluded value is
    # reported separately in 2.3.2 and labelled as such (review 2026-10-27).
    assert "去掉背景类的 mIoU" in text
    assert "| T4 | B2 | m_t4 | 2 |" in text
    assert "0.8000" in text


def test_confusion_section_is_graceful_without_artifacts(tmp_path: Path) -> None:
    records = [RunRecord.model_validate(payload("run-a", "m_a"))]
    matrix = {"cells": [cell("m_a", "run-a")]}
    lines: list[str] = []
    _confusion_section(records, tmp_path / "runs", matrix, lines.append)
    assert any("逐组混淆矩阵" in line for line in lines)


def test_report_shows_phase_costs_and_optimizer_steps(tmp_path: Path, monkeypatch) -> None:
    from rs_backbone_eval import report as report_module

    monkeypatch.setattr(report_module, "excluded_run_ids", lambda records, repo: set())
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


def test_confusion_section_adds_a_group_level_interval(tmp_path: Path) -> None:
    """Seeds of the same model repeat groups; the grouped interval counts each once."""
    runs_root = tmp_path / "runs"
    for run_id, weight in (("run-s17", counts(9, 1, 1, 9)), ("run-s42", counts(8, 2, 2, 8))):
        write_confusion(runs_root, run_id, {"g0": weight, "g1": weight, "g2": weight})
    records = [
        RunRecord.model_validate(payload("run-s17", "m_a", seed=17)),
        RunRecord.model_validate(payload("run-s42", "m_a", seed=42)),
    ]
    matrix = {
        "cells": [
            {
                "track": "B1",
                "model": "m_a",
                "task": "T1",
                "init": "pretrained",
                "label_fraction": "1.0",
                "runs": ["run-s17", "run-s42"],
            }
        ]
    }
    lines: list[str] = []
    _confusion_section(records, runs_root, matrix, lines.append)
    text = "\n".join(lines)
    assert "95% 区间（地理组，种子合并）" in text
    # 3 groups x 2 seeds for the old unit, 3 groups for the new one
    assert "| 6 / 3 |" in text


def test_stale_runs_are_listed_separately_not_in_the_main_table(
    tmp_path: Path, monkeypatch
) -> None:
    """Review 2026-10-21: pre-fix values must not pass as final results."""
    from rs_backbone_eval import report as report_module

    runs_root = tmp_path / "runs"
    for run_id, seed in (("run-a", 17), ("run-b", 42)):
        directory = runs_root / run_id
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(
            json.dumps(payload(run_id, "m_a", seed=seed)), encoding="utf-8"
        )
    monkeypatch.setattr(report_module, "excluded_run_ids", lambda records, repo: {"run-a"})
    config_root = Path(__file__).resolve().parents[1] / "configs"
    markdown = report_module.build_markdown(runs_root, tmp_path / "data", config_root, Path("."))

    assert "不能作为最终排名" in markdown
    head, _, tail = markdown.partition("### 3.4 修复前的记录")
    assert "`run-a`" in tail  # listed with its value and commit
    assert "`run-a`" not in head  # but never in the results table
    assert "`run-b`" in head


def test_common_seed_table_uses_the_variant_the_matrix_selected(
    tmp_path: Path, monkeypatch
) -> None:
    """Review 2026-10-22: do not let the last learning-rate variant win."""
    from rs_backbone_eval import report as report_module

    monkeypatch.setattr(report_module, "excluded_run_ids", lambda records, repo: set())
    runs_root = tmp_path / "runs"
    # m1's selected variant has seeds 17/42 at 0.8; a second, smaller variant
    # (seed 73 at 0.1) must not replace it in the common-seed comparison.
    selected = payload("m1-a", "m1", seed=17)
    selected["tags"]["learning_rate"] = "1e-4"
    selected42 = payload("m1-a42", "m1", seed=42)
    selected42["tags"]["learning_rate"] = "1e-4"
    other = payload("m1-b73", "m1", seed=73)
    other["tags"]["learning_rate"] = "3e-4"
    other["metrics"][0]["value"] = 0.1
    m2 = [payload(f"m2-{seed}", "m2", seed=seed) for seed in (17, 42, 73)]
    for record in m2:
        record["metrics"][0]["value"] = 0.7

    records = [RunRecord.model_validate(item) for item in [selected, selected42, other, *m2]]
    for record in records:
        directory = runs_root / record.run_id
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(record.model_dump_json(), encoding="utf-8")
    config_root = Path(__file__).resolve().parents[1] / "configs"
    markdown = build_markdown(runs_root, tmp_path / "data", config_root, tmp_path)
    section = markdown.split("共同种子对照")[1].split("###")[0]
    assert "17,42" in section
    assert "0.8000" in section
    assert "0.1000" not in section


def test_all_background_baseline_uses_the_confusion_support(tmp_path: Path) -> None:
    from rs_backbone_eval.report import _all_background_iou

    runs_root = tmp_path / "runs"
    directory = runs_root / "t3-run"
    directory.mkdir(parents=True)
    total = [
        [704, 0, 0, 0, 0],
        [62, 0, 0, 0, 0],
        [109, 0, 0, 0, 0],
        [107, 0, 0, 0, 0],
        [18, 0, 0, 0, 0],
    ]
    (directory / "confusion_per_scene.json").write_text(
        json.dumps({"num_classes": 5, "groups": {"g0": total}}), encoding="utf-8"
    )
    record = RunRecord.model_validate(payload("t3-run", "m_t3"))
    record.task_id = "T3"
    baseline, prevalence, classes = _all_background_iou(runs_root, [record])
    assert classes == 5
    assert prevalence == pytest.approx(0.704)
    assert baseline == pytest.approx(0.1408)


def test_few_shot_curve_ignores_smoke_only_fractions(tmp_path: Path, monkeypatch) -> None:
    """Review 2026-10-23: a 0.1-label short run used to crash the report."""
    from rs_backbone_eval import report as report_module

    monkeypatch.setattr(report_module, "excluded_run_ids", lambda records, repo: set())
    runs_root = tmp_path / "runs"
    official = payload("official", "m_a", seed=42)
    official["tags"]["label_fraction"] = "1.0"
    official["training"]["epochs_completed"] = 50
    smoke = payload("smoke", "m_a", seed=17)
    smoke["tags"]["label_fraction"] = "0.1"
    smoke["training"]["epochs_completed"] = 1
    for record in (official, smoke):
        directory = runs_root / record["run_id"]
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")

    config_root = Path(__file__).resolve().parents[1] / "configs"
    text = report_module.build_markdown(runs_root, tmp_path / "data", config_root, Path("."))
    assert "标注=0.1" not in text  # the only 0.1 run is a short run
    assert "标注=1.0" in text


def test_few_shot_curve_uses_the_official_run_for_a_fraction(tmp_path: Path, monkeypatch) -> None:
    from rs_backbone_eval import report as report_module

    monkeypatch.setattr(report_module, "excluded_run_ids", lambda records, repo: set())
    runs_root = tmp_path / "runs"
    official = payload("official-01", "m_a", seed=42)
    official["tags"]["label_fraction"] = "0.1"
    official["training"]["epochs_completed"] = 50
    smoke = payload("smoke-01", "m_a", seed=17)
    smoke["tags"]["label_fraction"] = "0.1"
    smoke["training"]["epochs_completed"] = 1
    smoke["metrics"][0]["value"] = 0.99
    for record in (official, smoke):
        directory = runs_root / record["run_id"]
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")

    config_root = Path(__file__).resolve().parents[1] / "configs"
    text = report_module.build_markdown(runs_root, tmp_path / "data", config_root, Path("."))
    section = text.split("### 2.1.2")[1].split("### 2.2")[0]
    assert "标注=0.1" in section
    assert "0.8000" in section  # the official value
    assert "0.9900" not in section  # the smoke value is never averaged in
