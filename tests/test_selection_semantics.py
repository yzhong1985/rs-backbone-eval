"""Regression tests for the P0 aggregation and metric-selection fixes.

These lock in three behaviours the review flagged: perturbation metrics must
never become the headline, duplicate seeds must not inflate the statistics, and
the headline variant must be chosen by the recorded validation score rather
than by seed count or input order.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rs_backbone_eval.contracts import RunRecord
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.matrix import build_matrix, canonical_runs


def run_payload(
    *,
    run_id: str,
    seed: int,
    value: float,
    learning_rate: str = "1e-4",
    updated_at: str = "2026-10-12T09:00:00Z",
    validation_mean_iou: str | None = None,
    protocol_id: str = "full_finetune",
    epochs: int = 50,
) -> dict:
    tags = {"learning_rate": learning_rate, "encoder_input_px": "512"}
    if validation_mean_iou is not None:
        tags["selection"] = "best_validation_mean_iou"
        tags["validation_mean_iou"] = validation_mean_iou
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": "2026-10-12T08:00:00Z",
        "updated_at": updated_at,
        "status": "completed",
        "task_id": "T1",
        "model_id": "clay_v1_5",
        "protocol_id": protocol_id,
        "seed": seed,
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
            "regime": "full_finetune",
            "head_id": "lite_spatial_head",
            "trainable_parameters": 1024,
            "total_parameters": 100001024,
            "epochs_completed": epochs,
            "wall_time_seconds": 1800,
            "peak_gpu_memory_mb": 4096,
        },
        "metrics": [
            {
                "name": "water_iou",
                "value": value,
                "split": "test",
                "direction": "maximize",
                "is_primary": True,
            },
        ],
        "tags": tags,
    }


def test_auxiliary_metrics_are_never_primary() -> None:
    records = to_metric_records(
        {"water_iou": 0.3}, primary_name="water_iou", split="band_drop:NIR", primary=False
    )
    assert all(not record.is_primary for record in records)


def test_perturbation_first_does_not_steal_the_headline() -> None:
    record = RunRecord.model_validate(run_payload(run_id="run-r", seed=42, value=0.81))
    # a perturbation sweep is appended first, the clean test result afterwards
    record.metrics = to_metric_records(
        {"water_iou": 0.3}, primary_name="water_iou", split="band_drop:NIR", primary=False
    ) + to_metric_records({"water_iou": 0.81}, primary_name="water_iou", split="test")
    assert record.primary_metric() is not None
    assert record.primary_metric().value == 0.81
    assert record.primary_metric().split == "test"


def test_canonical_runs_keeps_one_run_per_seed() -> None:
    records = [
        RunRecord.model_validate(run_payload(run_id="run-a", seed=42, value=0.75)),
        RunRecord.model_validate(
            run_payload(run_id="run-b", seed=42, value=0.77, updated_at="2026-10-12T10:00:00Z")
        ),
        RunRecord.model_validate(run_payload(run_id="run-c", seed=17, value=0.74)),
    ]
    canonical = canonical_runs(records)
    assert [record.run_id for record in canonical] == ["run-c", "run-b"]


def test_headline_variant_is_chosen_by_validation_not_seed_count() -> None:
    records = [
        # more seeds, worse validation score
        RunRecord.model_validate(
            run_payload(
                run_id="lr3e4-s42",
                seed=42,
                value=0.77,
                learning_rate="3e-4",
                validation_mean_iou="0.7000",
            )
        ),
        RunRecord.model_validate(
            run_payload(
                run_id="lr3e4-s17",
                seed=17,
                value=0.78,
                learning_rate="3e-4",
                validation_mean_iou="0.7000",
            )
        ),
        RunRecord.model_validate(
            run_payload(
                run_id="lr3e4-s73",
                seed=73,
                value=0.79,
                learning_rate="3e-4",
                validation_mean_iou="0.7000",
            )
        ),
        # one seed, better validation score
        RunRecord.model_validate(
            run_payload(
                run_id="lr1e4-s42",
                seed=42,
                value=0.80,
                learning_rate="1e-4",
                validation_mean_iou="0.7800",
            )
        ),
    ]
    matrix = build_matrix(records, {})
    cells = {(cell["track"], cell["model"], cell["task"]): cell for cell in matrix["cells"]}
    chosen = cells[("B1", "clay_v1_5", "T1")]
    assert chosen["learning_rate"] == "1e-4"
    assert chosen["n"] == 1


def test_best_checkpoint_persists_the_selected_weights(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from rs_backbone_eval.selection import BestCheckpoint

    model = torch.nn.Linear(3, 2)
    best = BestCheckpoint()
    assert best.offer(0.5, 3, model) is True
    assert best.offer(0.4, 4, model) is False  # a worse score is not selected
    path = tmp_path / "best_checkpoint.pt"
    best.save(path)
    payload = torch.load(path, weights_only=False)
    assert payload["epoch"] == 3
    assert payload["score"] == 0.5
    assert set(payload["state_dict"]) == set(model.state_dict())


def test_best_checkpoint_save_without_a_selection_writes_nothing(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from rs_backbone_eval.selection import BestCheckpoint

    path = tmp_path / "best_checkpoint.pt"
    BestCheckpoint().save(path)
    assert not path.exists()


def test_a_smoke_never_replaces_the_official_run_for_the_same_seed() -> None:
    official = RunRecord.model_validate(
        run_payload(run_id="official", seed=42, value=0.75, updated_at="2026-10-12T09:00:00Z")
    )
    smoke = RunRecord.model_validate(
        run_payload(
            run_id="smoke",
            seed=42,
            value=0.44,
            updated_at="2026-10-12T11:00:00Z",
            epochs=1,
        )
    )
    canonical = canonical_runs([official, smoke])
    assert [record.run_id for record in canonical] == ["official"]
    # A seed that only has a smoke keeps it, so the cell can still be flagged.
    assert [record.run_id for record in canonical_runs([smoke])] == ["smoke"]


def test_reduced_runs_of_other_seeds_do_not_enter_the_cell() -> None:
    records = [
        RunRecord.model_validate(run_payload(run_id="official-42", seed=42, value=0.75)),
        RunRecord.model_validate(run_payload(run_id="smoke-17", seed=17, value=0.40, epochs=1)),
        RunRecord.model_validate(run_payload(run_id="smoke-73", seed=73, value=0.45, epochs=1)),
    ]
    canonical = canonical_runs(records)
    assert [record.seed for record in canonical] == [42]

    matrix = build_matrix(records, {})
    cell = matrix["cells"][0]
    assert cell["n"] == 1
    assert cell["seeds"] == [42]
    assert cell["mean"] == pytest.approx(0.75)
    assert cell["sd"] is None  # one seed is not a standard deviation


def test_a_cell_with_only_reduced_runs_keeps_them() -> None:
    records = [
        RunRecord.model_validate(run_payload(run_id="smoke-17", seed=17, value=0.40, epochs=1)),
        RunRecord.model_validate(run_payload(run_id="smoke-73", seed=73, value=0.45, epochs=1)),
    ]
    assert [record.seed for record in canonical_runs(records)] == [17, 73]
