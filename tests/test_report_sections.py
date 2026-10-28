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
