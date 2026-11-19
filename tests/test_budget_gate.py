"""The budget gate that decides whether a run is a formal result.

Regression: the LIM worker kept its own ``OFFICIAL_EPOCHS = 50`` and stamped a
step-budget run that had completed its full 25,000 steps as
``[REDUCED-BUDGET]`` -- ``epochs_completed`` is 8 under a step budget, so the
epoch comparison could never pass. ``matrix.stopped_short_of_budget`` is now the
single implementation, and the worker imports it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rs_backbone_eval.contracts import RunRecord
from rs_backbone_eval.matrix import (
    OFFICIAL_STEPS,
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


def test_t10_detection_has_a_registered_step_budget() -> None:
    """The detection cell must be admitted by the same authority as T9.

    It stayed unregistered (fail-closed) until the throughput was measured; a
    number written before the measurement would have been self-approval.
    """
    assert OFFICIAL_STEPS["T10"] == 25_000
    assert not stopped_short_of_budget("T10", epochs_completed=3, steps_total=25_000)
    assert stopped_short_of_budget("T10", epochs_completed=3, steps_total=24_999)
    # No self-declared total_steps can substitute for the completed count.
    assert stopped_short_of_budget("T10", epochs_completed=12, steps_total=9_000, total_steps=9_000)


def test_the_t10_lane_exports_the_registered_budget() -> None:
    """A lane that stops at a different number would produce reduced runs."""
    import re

    text = (Path(__file__).resolve().parents[1] / "scripts/rerun_baselines.sh").read_text(
        encoding="utf-8"
    )
    lane = re.search(r"\n    T10B2\)\n(.*?)\n    [A-Z0-9]+\)", text, re.S)
    assert lane, "lane T10B2 is missing"
    exported = int(re.search(r"LIM_WORKER_TOTAL_STEPS=(\d+)", lane.group(1)).group(1))
    assert exported == OFFICIAL_STEPS["T10"]


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
            tags={"optimizer_steps": "25000", "total_steps_budget": "25000"},
        )
    )
    assert not is_reduced_run(full)

    short = RunRecord.model_validate(
        payload(
            run_id="lim-short",
            model_id="lim_mit_huge",
            task_id="T9",
            tags={"optimizer_steps": "2000", "total_steps_budget": "25000"},
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
        ("T9", 8, {"optimizer_steps": "400", "total_steps_budget": "400"}),
        ("T1", 0, {}),
        # T6a has no declared budget (only some tasks are graded today), so
        # even a full-looking step count cannot make it formal.
        ("T6a", 50, {"optimizer_steps": "25000", "total_steps_budget": "25000"}),
        ("T9", 50, {}),
        ("T9", 50, {"optimizer_steps": "0", "total_steps_budget": "25000"}),
        ("T9", 50, {"optimizer_steps": "invalid", "total_steps_budget": "25000"}),
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
            tags={"optimizer_steps": "25000", "total_steps_budget": "25000"},
        )
    ).model_copy(update={"updated_at": "2026-10-29T01:00:00Z"})
    short = RunRecord.model_validate(
        payload(
            run_id="later-losscheck",
            model_id="lim",
            task_id="T9",
            tags={"optimizer_steps": "400", "total_steps_budget": "400"},
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


def test_tile_count_tag_cannot_stand_in_for_optimizer_steps() -> None:
    """`train_steps` counts tiles; reading it would inflate steps 8x."""
    record = RunRecord.model_validate(
        payload(
            run_id="tiles-only",
            model_id="lim_mit_huge",
            task_id="T9",
            tags={"train_steps": "25000", "total_steps_budget": "25000"},
        )
    )
    assert is_reduced_run(record)


def test_t14_change_runs_use_the_declared_20_epoch_budget() -> None:
    """T14 is registered in OFFICIAL_EPOCHS, so a 20-epoch run is not REDUCED-BUDGET."""
    from rs_backbone_eval.matrix import OFFICIAL_EPOCHS, PROTOCOL_TRACKS, track_of

    assert OFFICIAL_EPOCHS["T14"] == 20
    assert not stopped_short_of_budget("T14", epochs_completed=20)
    assert stopped_short_of_budget("T14", epochs_completed=19)

    full = dict(
        payload(
            run_id="t14-full",
            model_id="lim_mit_huge",
            task_id="T14",
            epochs_completed=20,
        ),
        protocol_id="t14_b0",
    )
    short = dict(full, run_id="t14-short")
    short["training"] = {**full["training"], "epochs_completed": 19}

    full_record = RunRecord.model_validate(full)
    short_record = RunRecord.model_validate(short)
    assert PROTOCOL_TRACKS["t14_b0"] == "change"
    assert track_of(full_record) == "change"
    assert not is_reduced_run(full_record)
    assert is_reduced_run(short_record)
    assert formal_candidates([full_record, short_record]) == [full_record]


def test_t14_change_track_stays_out_of_the_prithvi_d1_tracks() -> None:
    """D1 is computed from B1/B2 only; the change track must not sit in either."""
    from rs_backbone_eval.matrix import DIAGNOSTIC_TRACKS, TRACK_NAMES, TRACK_ORDER

    assert "change" in TRACK_ORDER
    assert "change" in TRACK_NAMES
    assert "诊断" in TRACK_NAMES["change"]
    assert "change" in DIAGNOSTIC_TRACKS
    assert "change" not in {"B1", "B2"}


def test_t14_completed_20_epoch_run_is_diagnostic_not_headline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T14 is a diagnostic track: a completed 20-epoch run stays out of the matrix
    (headline, scorecard, Pareto, portal) and remains visible in the report."""
    from rs_backbone_eval import report as report_module
    from rs_backbone_eval.matrix import TRACK_NAMES, build_matrix

    monkeypatch.setattr(report_module, "excluded_run_ids", lambda records, repo: set())
    monkeypatch.setattr(report_module, "stale_run_ids", lambda records, repo: set())

    record = dict(
        payload(run_id="t14-b0-20ep", model_id="lim_mit_huge", task_id="T14", epochs_completed=20),
        protocol_id="t14_b0",
    )
    record["metrics"] = [
        {
            "name": "change_f1",
            "value": 0.61,
            "split": "test",
            "direction": "maximize",
            "is_primary": True,
        }
    ]
    run = RunRecord.model_validate(record)
    assert run.status == "completed" and not is_reduced_run(run)

    matrix = build_matrix([run], {})
    assert matrix["cells"] == []
    assert all("t14-b0-20ep" not in cell["runs"] for cell in matrix["cells"])

    runs_root = tmp_path / "runs"
    directory = runs_root / "t14-b0-20ep"
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")
    config_root = Path(__file__).resolve().parents[1] / "configs"
    text = report_module.build_markdown(runs_root, tmp_path / "data", config_root, Path("."))
    assert "t14-b0-20ep" in text
    assert TRACK_NAMES["change"] in text
