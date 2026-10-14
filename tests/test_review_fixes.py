"""Regression tests for the review findings R1, R3, R7 and R9."""

from __future__ import annotations

from rs_backbone_eval.contracts import RunRecord
from rs_backbone_eval.matrix import build_matrix, compatibility_matrix, perturbation_summary


def payload(
    *,
    run_id: str,
    model_id: str = "clay_v1_5",
    seed: int = 42,
    value: float = 0.8,
    lr: str = "1e-4",
    tags: dict | None = None,
    extra_metrics: list[dict] | None = None,
    metrics: list[dict] | None = None,
    status: str = "completed",
) -> dict:
    merged_tags = {"learning_rate": lr, "encoder_input_px": "512"}
    merged_tags.update(tags or {})
    if metrics is None:
        metrics = [
            {
                "name": "water_iou",
                "value": value,
                "split": "test",
                "direction": "maximize",
                "is_primary": True,
            }
        ]
    metrics = list(metrics) + list(extra_metrics or [])
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": status,
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
            "trainable_parameters": 1024,
            "total_parameters": 100001024,
            "epochs_completed": 8,
            "wall_time_seconds": 600,
            "peak_gpu_memory_mb": 4096,
        },
        "metrics": metrics,
        "tags": merged_tags,
    }


def cells_for(records: list[RunRecord]) -> list[dict]:
    return [cell for cell in build_matrix(records, {})["cells"] if cell["model"] == "clay_v1_5"]


def test_r1_init_and_label_budget_are_separate_panels() -> None:
    records = [
        RunRecord.model_validate(payload(run_id="run-pre", value=0.8)),
        RunRecord.model_validate(payload(run_id="run-rand", value=0.2, tags={"init": "random"})),
        RunRecord.model_validate(
            payload(run_id="run-10", value=0.3, tags={"label_fraction": "0.1"})
        ),
    ]
    cells = cells_for(records)
    panels = {(cell["init"], cell["label_fraction"]) for cell in cells}
    assert panels == {("pretrained", "1.0"), ("random", "1.0"), ("pretrained", "0.1")}


def test_r1_same_condition_retry_does_not_add_a_seed() -> None:
    records = [
        RunRecord.model_validate(payload(run_id="run-a", value=0.8)),
        RunRecord.model_validate(payload(run_id="run-b", value=0.6)),
    ]
    cells = cells_for(records)
    assert len(cells) == 1
    assert cells[0]["n"] == 1


def test_r3_t1_validation_tag_drives_selection_regardless_of_order() -> None:
    def candidate(run_id: str, lr: str, validation: str) -> RunRecord:
        return RunRecord.model_validate(
            payload(
                run_id=run_id,
                lr=lr,
                tags={"selection": "best_validation_water_iou", "validation_water_iou": validation},
            )
        )

    low = candidate("run-low", "1e-4", "0.5000")
    high = candidate("run-high", "3e-4", "0.9000")
    for order in ([low, high], [high, low]):
        chosen = cells_for(list(order))[0]
        assert chosen["learning_rate"] == "3e-4"


def test_r7_validation_split_is_not_a_perturbation() -> None:
    record = RunRecord.model_validate(
        payload(
            run_id="run-v",
            extra_metrics=[
                {
                    "name": "water_iou",
                    "value": 0.6,
                    "split": "validation",
                    "direction": "maximize",
                    "is_primary": False,
                }
            ],
        )
    )
    assert perturbation_summary([record]) == []


def test_r9_forward_probe_is_not_native_support() -> None:
    probe = RunRecord.model_validate(payload(run_id="run-probe", metrics=[], status="completed"))
    rows = compatibility_matrix([probe], {})
    assert rows[0]["received"] == "native"
    assert rows[0]["validated"] == "forward_only"
    assert rows[0]["support"] == "unknown"
