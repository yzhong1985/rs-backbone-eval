"""The budget gate that decides whether a run is a formal result.

Regression: the LIM worker kept its own ``OFFICIAL_EPOCHS = 50`` and stamped a
step-budget run that had completed its full 25,000 steps as
``[REDUCED-BUDGET]`` -- ``epochs_completed`` is 8 under a step budget, so the
epoch comparison could never pass. ``matrix.stopped_short_of_budget`` is now the
single implementation, and the worker imports it.
"""

from __future__ import annotations

import pytest

from rs_backbone_eval.contracts import RunRecord
from rs_backbone_eval.matrix import (
    build_matrix,
    canonical_runs,
    formal_candidates,
    is_reduced_run,
    stopped_short_of_budget,
    variant_key,
)


def payload(
    *,
    run_id: str,
    model_id: str,
    task_id: str,
    tags: dict | None = None,
    epochs_completed: int = 8,
) -> dict:
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "completed",
        "task_id": task_id,
        "model_id": model_id,
        "protocol_id": "upernet_frozen",
        "seed": 17,
        "dataset": {
            "dataset_id": "iSAID",
            "manifest_sha256": "b" * 64,
            "split": "test",
        },
        "input": {
            "sensor": "aerial-rgb",
            "bands": ["RED", "GREEN", "BLUE"],
            "time_steps": 1,
            "preprocessing_id": "isaid_rgb_train_stats",
        },
        "training": {
            "regime": "frozen_probe",
            "head_id": "upernet_gn",
            "trainable_parameters": 6_956_816,
            "total_parameters": 531_391_760,
            "epochs_completed": epochs_completed,
            "wall_time_seconds": 600,
            "peak_gpu_memory_mb": 4096,
        },
        "metrics": [
            {
                "name": "mean_iou",
                "value": 0.5,
                "split": "test",
                "direction": "maximize",
                "is_primary": True,
            }
        ],
        "tags": tags or {},
    }


def test_step_budget_is_judged_by_steps() -> None:
    # T9: 25,000 steps x 8 tiles; 8 whole epochs is the full budget
    assert not stopped_short_of_budget(
        "T9", epochs_completed=8, steps_total=25_000, total_steps=25_000
    )
    assert not stopped_short_of_budget(
        "T9", epochs_completed=8, steps_total=26_500, total_steps=25_000
    )
    assert stopped_short_of_budget("T9", epochs_completed=8, steps_total=24_000, total_steps=25_000)


def test_epoch_budget_still_applies_without_a_step_budget() -> None:
    assert not stopped_short_of_budget("T1", epochs_completed=50)
    assert stopped_short_of_budget("T1", epochs_completed=49)
    assert not stopped_short_of_budget("T1", epochs_completed=60)
    # Missing completion evidence must never enter the formal result set.
    assert stopped_short_of_budget("T1", epochs_completed=0)
    # an unknown task has no declared budget
    assert stopped_short_of_budget("T99", epochs_completed=1)


def test_is_reduced_run_reads_the_step_counters() -> None:
    """A step-budget run that met its budget must stay a formal candidate."""
    full = RunRecord.model_validate(
        payload(
            run_id="lim-full",
            model_id="lim_mit_huge",
            task_id="T9",
            tags={"train_steps": "25000", "total_steps_budget": "25000"},
        )
    )
    assert not is_reduced_run(full)

    short = RunRecord.model_validate(
        payload(
            run_id="lim-short",
            model_id="lim_mit_huge",
            task_id="T9",
            tags={"train_steps": "2000", "total_steps_budget": "25000"},
        )
    )
    assert is_reduced_run(short)


def test_is_reduced_run_keeps_the_note_and_epoch_rules() -> None:
    smoke = RunRecord.model_validate(
        payload(run_id="smoke", model_id="m", task_id="T1", tags={})
    ).model_copy(update={"notes": "mean_iou=0.2 [SMOKE]"})
    assert is_reduced_run(smoke)

    short = RunRecord.model_validate(
        payload(run_id="short", model_id="m", task_id="T1", epochs_completed=10)
    )
    assert is_reduced_run(short)

    full = RunRecord.model_validate(
        payload(run_id="full", model_id="m", task_id="T1", epochs_completed=50)
    )
    assert not is_reduced_run(full)


@pytest.mark.parametrize("declared", [None, 0, 400, 25_000, 100_000])
def test_worker_limit_cannot_define_the_official_budget(declared) -> None:
    assert stopped_short_of_budget("T9", epochs_completed=50, steps_total=400, total_steps=declared)
    assert not stopped_short_of_budget(
        "T9", epochs_completed=0, steps_total=25_000, total_steps=declared
    )
    # A step limit must not bypass an epoch-based task's requirement either.
    assert stopped_short_of_budget("T1", epochs_completed=1, steps_total=400, total_steps=400)


@pytest.mark.parametrize(
    ("task", "epochs", "tags"),
    [
        ("T9", 8, {"train_steps": "400", "total_steps_budget": "400"}),
        ("T1", 0, {}),
        ("T10", 50, {"train_steps": "25000", "total_steps_budget": "25000"}),
        ("T9", 50, {}),
        ("T9", 50, {"train_steps": "0", "total_steps_budget": "25000"}),
        ("T9", 50, {"train_steps": "invalid", "total_steps_budget": "25000"}),
    ],
)
def test_invalid_budget_evidence_is_not_formal(task, epochs, tags) -> None:
    record = RunRecord.model_validate(
        payload(run_id="diagnostic", model_id="m", task_id=task, epochs_completed=epochs, tags=tags)
    )
    assert is_reduced_run(record)
    assert formal_candidates([record]) == []
    assert canonical_runs([record]) == []
    assert build_matrix([record], {})["cells"] == []


def test_newer_400_step_run_cannot_replace_25000_step_result() -> None:
    full = RunRecord.model_validate(
        payload(
            run_id="full-25k",
            model_id="lim",
            task_id="T9",
            tags={"train_steps": "25000", "total_steps_budget": "25000"},
        )
    ).model_copy(update={"updated_at": "2026-10-29T01:00:00Z"})
    short = RunRecord.model_validate(
        payload(
            run_id="later-losscheck",
            model_id="lim",
            task_id="T9",
            tags={"train_steps": "400", "total_steps_budget": "400"},
        )
    ).model_copy(update={"updated_at": "2026-10-29T02:00:00Z"})
    assert variant_key(full) == variant_key(short)
    assert canonical_runs([full, short]) == [full]
    assert formal_candidates([full, short]) == [full]
    assert build_matrix([full, short], {})["cells"][0]["runs"] == ["full-25k"]


def test_structured_optimizer_steps_and_legacy_tags_are_supported() -> None:
    data = payload(run_id="structured", model_id="clay", task_id="T9")
    data["training"]["optimizer_steps"] = 25_000
    assert not is_reduced_run(RunRecord.model_validate(data))
    data["training"]["optimizer_steps"] = 400
    data["tags"] = {"train_steps": "25000", "total_steps_budget": "25000"}
    assert is_reduced_run(RunRecord.model_validate(data))


@pytest.mark.parametrize("task", ["T1S", "T8"])
def test_existing_50_epoch_domain_tasks_keep_their_budget(task) -> None:
    assert not stopped_short_of_budget(task, epochs_completed=50)
    assert stopped_short_of_budget(task, epochs_completed=49)
