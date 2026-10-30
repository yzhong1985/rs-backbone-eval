"""The report's "what is still missing" section must follow the ledger.

Before 2026-10-28 §3.6 was hand-written prose and still claimed T2/T5 B1 had
never run, so an exported report contradicted its own tables.
"""

from __future__ import annotations

import json
from pathlib import Path

from rs_backbone_eval import report as report_module


def payload(run_id: str, model_id: str, *, seed: int = 17) -> dict:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "completed",
        "task_id": "T1",
        "model_id": model_id,
        "protocol_id": "full_finetune",
        "seed": seed,
        "dataset": {
            "dataset_id": "Sen1Floods11",
            "manifest_sha256": "b" * 64,
            "split": "test",
        },
        "input": {
            "sensor": "sentinel-2",
            "bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            "time_steps": 1,
            "preprocessing_id": "s2-six-band-v1",
        },
        "training": {
            "regime": "full_finetune",
            "head_id": "lite_spatial_head",
            "epochs_completed": 50,
        },
        "metrics": [
            {
                "name": "water_iou",
                "value": 0.8,
                "split": "test",
                "direction": "maximize",
                "is_primary": True,
            }
        ],
        "artifacts": [],
        "tags": {"learning_rate": "1e-4", "encoder_input_px": "512"},
    }


def test_progress_section_is_derived_from_the_matrix(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(report_module, "excluded_run_ids", lambda records, repo: set())
    monkeypatch.setattr(report_module, "stale_run_ids", lambda records, repo: set())

    runs_root = tmp_path / "runs"
    record = payload("T1-run", "m_a")
    directory = runs_root / record["run_id"]
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")

    config_root = Path(__file__).resolve().parents[1] / "configs"
    text = report_module.build_markdown(runs_root, tmp_path / "data", config_root, Path("."))
    section = text.split("### 3.6 尚未完成", 1)[1].split("## ", 1)[0]

    # Coverage comes from the matrix: one seed of three for this cell.
    assert "B1（全量微调）覆盖" in section
    assert "1/3" in section
    # Tasks declared in configs but absent from the ledger are named.
    assert "未跑的任务" in section
    assert "T7" in section
    # The hand-written claim that must never come back.
    assert "T2、T5 的全量微调（B1）未运行" not in section


def test_a_corrected_tag_wording_is_reported_next_to_the_run(tmp_path: Path, monkeypatch) -> None:
    """A run started before a tag-wording fix must not be hand-edited.

    The first iSAID Prithvi run (21:43) predates 8f5df4f and still says
    "zero-filled" for the unobserved NIR/SWIR bands. The report states the
    correction instead of rewriting the record; a run that is not in the table
    must not grow a note.
    """
    monkeypatch.setattr(report_module, "excluded_run_ids", lambda records, repo: set())
    monkeypatch.setattr(report_module, "stale_run_ids", lambda records, repo: set())

    runs_root = tmp_path / "runs"
    for run_id in ("prithvi_eo_v2_300_tl-T9-upernet_frozen-train-17", "other-run"):
        record = payload(run_id, "prithvi_eo_v2_300_tl")
        directory = runs_root / run_id
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")

    config_root = Path(__file__).resolve().parents[1] / "configs"
    text = report_module.build_markdown(runs_root, tmp_path / "data", config_root, Path("."))
    section = text.split("### 3.4c 标签措辞已更正的旧运行（数值不受影响）", 1)[1]
    section = section.split("### 3.5", 1)[0]

    assert "prithvi_eo_v2_300_tl-T9-upernet_frozen-train-17" in section
    assert "标准化空间" in section
    assert "other-run" not in section


def t10_payload(run_id: str, *, seed: int = 17, steps: int = 25_000, ap: float = 0.31) -> dict:
    record = payload(run_id, "lim_mit_huge", seed=seed)
    record["task_id"] = "T10"
    record["protocol_id"] = "rotated_frozen"
    record["training"] = {
        "regime": "frozen_probe",
        "head_id": "rotated_faster_rcnn",
        "epochs_completed": 3,
        "optimizer_steps": steps,
    }
    record["metrics"] = [
        {"name": "map", "value": ap, "split": "test", "direction": "maximize", "is_primary": True},
        {"name": "map_50", "value": 0.52, "split": "test", "direction": "maximize"},
        {"name": "map_75", "value": 0.30, "split": "test", "direction": "maximize"},
        {"name": "recall_small", "value": 0.11, "split": "test", "direction": "maximize"},
        {"name": "recall_medium", "value": 0.42, "split": "test", "direction": "maximize"},
        {"name": "recall_large", "value": 0.63, "split": "test", "direction": "maximize"},
        {
            "name": "ap50",
            "class_name": "Dry Cargo Ship",
            "value": 0.5,
            "split": "test",
            "direction": "maximize",
        },
    ]
    record["tags"] = {"classes_with_targets": "37", "optimizer_steps": str(steps)}
    return record


def test_detection_metrics_get_their_own_section(tmp_path: Path, monkeypatch) -> None:
    """mAP must not be mixed into the IoU table or the D1 average.

    A short T10 run stays a diagnostic: reporting it next to the formal cell (or
    letting it replace it) is the failure this guards.
    """
    monkeypatch.setattr(report_module, "excluded_run_ids", lambda records, repo: set())
    monkeypatch.setattr(report_module, "stale_run_ids", lambda records, repo: set())

    runs_root = tmp_path / "runs"
    records = [
        t10_payload("t10-formal-17", seed=17, ap=0.31),
        t10_payload("t10-formal-42", seed=42, ap=0.35),
        t10_payload("t10-smoke", seed=73, steps=400, ap=0.01),
    ]
    for record in records:
        directory = runs_root / record["run_id"]
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")

    config_root = Path(__file__).resolve().parents[1] / "configs"
    text = report_module.build_markdown(runs_root, tmp_path / "data", config_root, Path("."))
    section = text.split("### 2.5 检测结果（T10，旋转框）", 1)[1].split("### 2.4", 1)[0]

    assert "AP50" in section and "0.5200" in section
    # Two canonical seeds of the one cell, mean over 0.31 and 0.35.
    assert "2 |" in section
    assert "0.3300" in section
    # The 400-step run is a diagnostic and must not reach the table.
    assert "0.0100" not in section
    assert "不并入 D1" in section


def t9_payload(run_id: str, *, seed: int = 17, background: float = 0.99) -> dict:
    record = payload(run_id, "lim_mit_huge", seed=seed)
    record["task_id"] = "T9"
    record["protocol_id"] = "upernet_frozen"
    record["training"] = {
        "regime": "frozen_probe",
        "head_id": "upernet_gn",
        "epochs_completed": 8,
        "optimizer_steps": 25_000,
    }
    classes = {0: background, 1: 0.60, 2: 0.30, 3: 0.10}
    record["metrics"] = [
        {
            "name": "mean_iou",
            "value": sum(classes.values()) / len(classes),
            "split": "test",
            "direction": "maximize",
            "is_primary": True,
        }
    ] + [
        {
            "name": "iou",
            "class_name": f"class_{index}",
            "value": value,
            "split": "test",
            "direction": "maximize",
        }
        for index, value in classes.items()
    ]
    record["tags"] = {"optimizer_steps": "25000"}
    return record


def test_isaid_section_states_the_background_and_foreground_means(
    tmp_path: Path, monkeypatch
) -> None:
    """0.4873 with background is not comparable to a 15-class paper number.

    The report must print both means and name the classes, so a reader cannot
    mistake a frozen-backbone mIoU for a fine-tuned one.
    """
    monkeypatch.setattr(report_module, "excluded_run_ids", lambda records, repo: set())
    monkeypatch.setattr(report_module, "stale_run_ids", lambda records, repo: set())

    runs_root = tmp_path / "runs"
    for record in (t9_payload("t9-17", seed=17), t9_payload("t9-smoke", seed=1, background=0.5)):
        record["training"]["optimizer_steps"] = 25_000 if record["run_id"] == "t9-17" else 400
        directory = runs_root / record["run_id"]
        directory.mkdir(parents=True)
        (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")

    config_root = Path(__file__).resolve().parents[1] / "configs"
    text = report_module.build_markdown(runs_root, tmp_path / "data", config_root, Path("."))
    section = text.split("### 2.6 iSAID 每类 IoU", 1)[1].split("### 2.4", 1)[0]

    assert "含背景宏平均：**0.4975**" in section  # (0.99 + 0.60 + 0.30 + 0.10) / 4
    assert "前景 3 类宏平均：**0.3333**" in section
    assert "storage_tank" in section and "Helicopter" not in section
    assert "0.5000" not in section  # the 400-step run must not be averaged in
