"""Regression: a completed two-stage record must not crash the matrix/report.

Locks review P1 — ``variant_key()`` appends two fields for the two-stage track
(feature_cache_rows, stage_b_start); ``build_matrix`` and ``build_markdown``
unpacked the composite key at a fixed width of 9, so a single completed
two-stage run in the store raised ``ValueError: too many values to unpack``.
"""

from __future__ import annotations

from pathlib import Path

from rs_backbone_eval.contracts import RunRecord
from rs_backbone_eval.matrix import build_matrix
from rs_backbone_eval.report import build_markdown


def _payload(run_id: str, seed: int) -> dict:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "completed",
        "task_id": "T5",
        "model_id": "anysat",
        "protocol_id": "frozen_adapter_two_stage",
        "seed": seed,
        "dataset": {
            "dataset_id": "USMultiTemporalCrop",
            "manifest_sha256": "c" * 64,
            "split": "test",
        },
        "input": {
            "sensor": "sentinel-2",
            "bands": ["BLUE", "GREEN", "RED"],
            "time_steps": 3,
            "preprocessing_id": "s2-three-band-v1",
        },
        "training": {
            "regime": "frozen_probe",
            "head_id": "lite_spatial_head",
            "trainable_parameters": 1024,
            "total_parameters": 100001024,
            "epochs_completed": 125,
            "wall_time_seconds": 600,
            "peak_gpu_memory_mb": 4096,
        },
        "metrics": [
            {
                "name": "crop_iou",
                "value": 0.55 + 0.01 * seed,
                "split": "test",
                "direction": "maximize",
                "is_primary": True,
            }
        ],
        "tags": {
            "feature_cache_rows": "312",
            "stage_b_start": "62",
        },
    }


def test_two_stage_records_render_in_matrix_and_markdown(tmp_path: Path, monkeypatch) -> None:
    from rs_backbone_eval import report as report_module

    monkeypatch.setattr(report_module, "excluded_run_ids", lambda records, repo: set())
    records = [
        RunRecord.model_validate(_payload("two-stage-a", seed=42)),
        RunRecord.model_validate(_payload("two-stage-b", seed=17)),
    ]
    matrix = build_matrix(records, {})
    # the B2S cell carries the two-stage condition axes on the candidate
    cells = [cell for cell in matrix["cells"] if cell.get("protocol") == "frozen_adapter_two_stage"]
    assert cells, "two-stage cell missing from the matrix"
    assert cells[0]["feature_cache_rows"] == "312"
    assert cells[0]["stage_b_start"] == "62"

    # build_markdown must not raise on a two-stage record either. It reads a
    # runs_root via RunRepository, so drop the two run.json files on disk.
    runs_root = tmp_path / "runs"
    for record in records:
        run_dir = runs_root / record.run_id
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
    out = build_markdown(
        runs_root=runs_root,
        data_root=tmp_path,
        config_root=tmp_path,
        repo_root=tmp_path,
    )
    assert "B2S" in out  # the track row renders
    assert "两阶段冻结" in out
