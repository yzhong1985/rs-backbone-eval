"""Regression tests for the direction-aware, no-forced-zero scorecard.

Locks in the evaluation-system changes: D1 respects the metric direction, D3/D4
expose raw values (no "worst maps to zero"), D2 reports absolute/drop/retention
from paired perturbation runs, and the input-compatibility matrix distinguishes
native support from an explicit adapter.
"""

from __future__ import annotations

from rs_backbone_eval.contracts import RunRecord
from rs_backbone_eval.matrix import (
    build_matrix,
    compatibility_matrix,
    perturbation_summary,
    scorecard,
)


def payload(
    *,
    run_id: str,
    model_id: str,
    seed: int = 42,
    metric: str = "water_iou",
    value: float = 0.5,
    direction: str = "maximize",
    extra_metrics: list[dict] | None = None,
    tags: dict | None = None,
    status: str = "completed",
    task_id: str = "T1",
) -> dict:
    metrics = [
        {
            "name": metric,
            "value": value,
            "split": "test",
            "direction": direction,
            "is_primary": True,
        }
    ]
    metrics.extend(extra_metrics or [])
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": status,
        "task_id": task_id,
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
            "trainable_parameters": 1024,
            "total_parameters": 100001024,
            "epochs_completed": 8,
            "wall_time_seconds": 600,
            "peak_gpu_memory_mb": 4096,
        },
        "metrics": metrics,
        "tags": tags or {},
    }


def test_d1_respects_minimize_direction() -> None:
    records = [
        RunRecord.model_validate(
            payload(
                run_id="run-low", model_id="m_low", value=0.2, metric="rmse", direction="minimize"
            )
        ),
        RunRecord.model_validate(
            payload(
                run_id="run-high", model_id="m_high", value=0.4, metric="rmse", direction="minimize"
            )
        ),
    ]
    card = scorecard(build_matrix(records, {}), records, {})
    rows = {row["model"]: row for row in card["models"]}
    # lower RMSE is better: 0.2 is the reference (ratio 1.0), 0.4 is 0.5
    assert rows["m_low"]["d1_b1"]["score"] == 1.0
    assert rows["m_high"]["d1_b1"]["score"] == 0.5


def test_scorecard_has_no_forced_zero_scores_and_marks_pareto() -> None:
    records = [
        RunRecord.model_validate(payload(run_id="run-a", model_id="m_a", value=0.80)),
        RunRecord.model_validate(payload(run_id="run-b", model_id="m_b", value=0.70)),
    ]
    card = scorecard(build_matrix(records, {}), records, {})
    pareto = {row["model"]: row["pareto"] for row in card["models"]}
    for row in card["models"]:
        assert "score" not in row["d3"]
        assert "score" not in row["d4"]
        assert row["d4"]["total_runs"] == 1
    # equal cost, higher accuracy: the lower-accuracy model is dominated
    assert pareto["m_a"] is True
    assert pareto["m_b"] is False


def test_perturbation_summary_is_paired_and_direction_aware() -> None:
    perturbed = {
        "name": "water_iou",
        "value": 0.6,
        "split": "band_drop:NIR",
        "direction": "maximize",
        "is_primary": False,
    }
    record = RunRecord.model_validate(
        payload(run_id="run-p", model_id="m", value=0.8, extra_metrics=[perturbed])
    )
    rows = perturbation_summary([record])
    assert len(rows) == 1
    assert rows[0]["baseline"] == 0.8
    assert rows[0]["perturbed"] == 0.6
    assert round(rows[0]["drop"], 4) == 0.2
    assert round(rows[0]["retention"], 4) == 0.75

    # minimize metric: a larger value is worse, so drop is positive when it grows
    rmse = RunRecord.model_validate(
        payload(
            run_id="run-pr",
            model_id="m",
            metric="rmse",
            direction="minimize",
            value=0.2,
            extra_metrics=[
                {
                    "name": "rmse",
                    "value": 0.3,
                    "split": "resolution:2",
                    "direction": "minimize",
                    "is_primary": False,
                }
            ],
        )
    )
    row = perturbation_summary([rmse])[0]
    assert round(row["drop"], 4) == 0.1


def test_compatibility_matrix_separates_adapter_from_native() -> None:
    records = [
        RunRecord.model_validate(payload(run_id="run-nat", model_id="m_nat", tags={})),
        RunRecord.model_validate(
            payload(run_id="run-adp", model_id="m_adp", tags={"adapter_parameters": "1546"})
        ),
    ]
    rows = {row["model"]: row for row in compatibility_matrix(records, {})}
    assert rows["m_nat"]["support"] == "native"
    assert rows["m_adp"]["support"] == "adapter"
    assert rows["m_adp"]["adapter_parameters"] == 1546
