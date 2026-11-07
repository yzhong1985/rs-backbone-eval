"""Regression tests for the direction-aware, no-forced-zero scorecard.

Locks in the evaluation-system changes: D1 respects the metric direction, D3/D4
expose raw values (no "worst maps to zero"), D2 reports absolute/drop/retention
from paired perturbation runs, and the input-compatibility matrix distinguishes
native support from an explicit adapter.
"""

from __future__ import annotations

import pytest

from rs_backbone_eval.contracts import RunRecord
from rs_backbone_eval.matrix import (
    build_matrix,
    compatibility_matrix,
    holdout_summary,
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
            "epochs_completed": 50,
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
        assert row["d4"]["training_runs"] == 1
        assert row["d4"]["failed_runs"] == 0
        assert row["d4"]["failure_rate"] == 0.0
    # equal cost, higher accuracy: the lower-accuracy model is dominated
    assert pareto["m_a"] is True
    assert pareto["m_b"] is False


def test_perturbation_summary_is_paired_and_direction_aware() -> None:
    perturbed = {
        "name": "water_iou",
        "value": 0.6,
        "split": "test_band_drop-NIR",
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
                    "split": "test_resolutionx2",
                    "direction": "minimize",
                    "is_primary": False,
                }
            ],
        )
    )
    row = perturbation_summary([rmse])[0]
    assert round(row["drop"], 4) == 0.1


def test_perturbations_survive_a_b1_run_with_the_same_variant_key() -> None:
    perturbed = {
        "name": "water_iou",
        "value": 0.6,
        "split": "test_band_drop-NIR",
        "direction": "maximize",
        "is_primary": False,
    }
    frozen = RunRecord.model_validate(
        payload(run_id="run-b2", model_id="m", value=0.8, extra_metrics=[perturbed])
    )
    frozen.protocol_id = "frozen_probe"
    # Same model, seed and variant key, no perturbation metrics, updated later.
    finetune = RunRecord.model_validate(payload(run_id="run-b1", model_id="m", value=0.9))
    finetune.updated_at = "2099-01-01T00:00:00Z"
    rows = perturbation_summary([frozen, finetune])
    assert [row["run_id"] for row in rows] == ["run-b2"]


def holdout_metric(value: float) -> dict:
    return {
        "name": "water_iou",
        "value": value,
        "split": "bolivia_holdout",
        "direction": "maximize",
        "is_primary": False,
    }


def test_holdout_summary_pairs_headline_and_holdout_over_the_same_seeds() -> None:
    records = [
        RunRecord.model_validate(
            payload(
                run_id=f"run-{seed}",
                model_id="m",
                seed=seed,
                value=0.8,
                extra_metrics=[holdout_metric(0.6 + seed / 1000)],
            )
        )
        for seed in (17, 42, 73)
    ]
    rows = holdout_summary(records)
    assert len(rows) == 1
    row = rows[0]
    assert row["split"] == "bolivia_holdout" and row["n"] == 3
    assert row["seeds"] == [17, 42, 73]
    assert row["headline_mean"] == pytest.approx(0.8)
    assert row["holdout_mean"] == pytest.approx(0.6 + 44 / 1000)


def test_holdout_summary_skips_cells_where_some_seeds_lack_the_holdout() -> None:
    with_holdout = RunRecord.model_validate(
        payload(run_id="run-a", model_id="m", seed=17, extra_metrics=[holdout_metric(0.5)])
    )
    without = RunRecord.model_validate(payload(run_id="run-b", model_id="m", seed=42))
    assert holdout_summary([with_holdout, without]) == []


def test_holdout_is_not_a_perturbation() -> None:
    record = RunRecord.model_validate(
        payload(run_id="run-h", model_id="m", extra_metrics=[holdout_metric(0.5)])
    )
    assert perturbation_summary([record]) == []


def test_compatibility_matrix_separates_adapter_from_native() -> None:
    records = [
        RunRecord.model_validate(
            payload(
                run_id="run-nat",
                model_id="m_nat",
                tags={"true_input": "aux s2_path RGB B04/B03/B02"},
            )
        ),
        RunRecord.model_validate(
            payload(run_id="run-adp", model_id="m_adp", tags={"adapter_parameters": "1546"})
        ),
    ]
    rows = {row["model"]: row for row in compatibility_matrix(records, {})}
    assert rows["m_nat"]["support"] == "native"
    assert rows["m_adp"]["support"] == "adapter"
    assert rows["m_adp"]["adapter_parameters"] == 1546
    assert rows["m_nat"]["true_input"] == "aux s2_path RGB B04/B03/B02"


def test_superseded_runs_are_excluded_from_the_matrix() -> None:
    """A run tagged superseded must not surface as a cell (old unseeded runs)."""
    live = RunRecord.model_validate(payload(run_id="live-run", model_id="m_live", value=0.5))
    old = RunRecord.model_validate(
        payload(
            run_id="old-run",
            model_id="m_old",
            value=0.9,
            tags={"superseded": "unseeded pre-fix run"},
        )
    )
    matrix = build_matrix([live, old], {})
    models = {cell["model"] for cell in matrix["cells"]}
    assert "m_live" in models
    assert "m_old" not in models


def test_failure_rate_is_split_into_incident_classes() -> None:
    """Review 2026-10-21: run incidents must not read as model instability."""
    ok = payload(run_id="ok-run", model_id="m", value=0.5)
    guard = payload(run_id="guard-run", model_id="m", value=0.1, status="failed")
    guard["notes"] = (
        "protocol full_finetune requires train_backbone=True; the worker implements only "
        "frozen-backbone head training"
    )
    oom = payload(run_id="oom-run", model_id="m", value=0.1, status="failed")
    oom["notes"] = "RuntimeError: CUDA out of memory"
    cache = payload(run_id="cache-run", model_id="m", value=0.1, status="failed")
    cache["notes"] = "BadZipFile: Bad CRC-32 for file 'dense.npy'"
    interrupted = payload(run_id="stop-run", model_id="m", value=0.1, status="failed")
    interrupted["notes"] = "KeyboardInterrupt: interrupted by the operator"

    records = [RunRecord.model_validate(item) for item in (ok, guard, oom, cache, interrupted)]
    card = scorecard(build_matrix(records, {}), records, {})
    d4 = card["models"][0]["d4"]
    assert d4["failed_runs"] == 4
    assert d4["training_runs"] == 5
    assert d4["incident_classes"] == {
        "配置拦截": 1,
        "显存不足": 1,
        "缓存故障": 1,
        "人工中断": 1,
    }


def test_d3_cost_uses_the_b1_runs_not_the_frozen_head() -> None:
    """Review 2026-10-22: a B2 head's 2.7M params must not sit in a B1 cost cell."""
    b1 = payload(run_id="b1-run", model_id="m", value=0.8)
    b1["training"].update(
        {
            "trainable_parameters": 312_887_554,
            "total_parameters": 312_887_554,
            "wall_time_seconds": 2253.0,
            "peak_gpu_memory_mb": 9904.0,
        }
    )
    b2 = payload(run_id="b2-run", model_id="m", value=0.5)
    b2["protocol_id"] = "frozen_probe"
    b2["training"].update({"trainable_parameters": 2_671_661, "total_parameters": 314_114_605})

    records = [RunRecord.model_validate(b1), RunRecord.model_validate(b2)]
    card = scorecard(build_matrix(records, {}), records, {})
    d3 = card["models"][0]["d3"]
    assert d3["b1_trainable_params"] == 312_887_554
    assert d3["b1_total_params"] == 312_887_554
    assert d3["b2_trainable_params"] == 2_671_661
    assert d3["b1_wall_min_mean"] == pytest.approx(37.5, abs=0.2)


def test_a_fixed_band_adapter_is_not_reported_as_native_input() -> None:
    """Review 2026-10-22: OlmoEarth gets 6 of 12 bands via a constant-fill adapter."""
    olmo = payload(run_id="olmo-run", model_id="olmoearth_v1_2_tiny", value=0.6)
    olmo["protocol_id"] = "frozen_adapter"
    olmo["tags"] = {
        "band_adapter": "constant_fill_12band",
        "adapter_note": "missing bands filled with the standardised mean",
    }
    plain = payload(run_id="plain-run", model_id="anysat_base", value=0.8)

    records = [RunRecord.model_validate(olmo), RunRecord.model_validate(plain)]
    rows = compatibility_matrix(records, {})
    by_model = {row["model"]: row for row in rows}
    assert by_model["olmoearth_v1_2_tiny"]["received"] == "adapter"
    assert by_model["anysat_base"]["received"] == "native"


def test_d3_excludes_the_timing_contaminated_run() -> None:
    """Review 2026-11-02: seed 73 was timed while G1 shared the GPU.

    The metrics join the cell normally, but the wall time must not enter the
    cost dimension — only the uncontaminated seed's minutes may.
    """
    from rs_backbone_eval.matrix import TIMING_EXCLUDED, build_matrix

    normal = payload(run_id="clay_v1_5-T4-full_finetune-42-v1", model_id="m", seed=42, value=0.8)
    normal["training"].update({"wall_time_seconds": 2253.0})
    contaminated = payload(
        run_id="clay_v1_5-T4-full_finetune-73-v1", model_id="m", seed=73, value=0.7
    )
    contaminated["training"].update({"wall_time_seconds": 36000.0})
    assert "clay_v1_5-T4-full_finetune-73-v1" in TIMING_EXCLUDED

    records = [RunRecord.model_validate(r) for r in (normal, contaminated)]
    card = scorecard(build_matrix(records, {}), records, {})
    d3 = card["models"][0]["d3"]
    assert d3["b1_wall_min_mean"] == pytest.approx(2253.0 / 60, abs=0.2)


def test_build_matrix_quarantines_a_budget_mixing_cell(monkeypatch) -> None:
    """A misregistered protocol must not 500 the report or the portal.

    Review 2026-11-05 round 13: build_matrix used to raise ValueError when one
    cell held two input budgets. The report and the portal share this function,
    so a single bad registration took down every page. It now returns the cell
    in ``invalid_cells`` and leaves it out of the matrix.
    """
    from rs_backbone_eval import matrix as matrix_module

    monkeypatch.setitem(matrix_module.PROTOCOL_TRACKS, "lim_s2_20bin", "BB1")
    first = payload(run_id="t11-single", model_id="lim_mit_huge", task_id="T11", metric="mean_iou")
    first["protocol_id"] = "lim_s2_single"
    first["training"]["epochs_completed"] = 8
    second = payload(run_id="t11-20bin", model_id="lim_mit_huge", task_id="T11", metric="mean_iou")
    second["protocol_id"] = "lim_s2_20bin"
    second["training"]["epochs_completed"] = 8
    records = [RunRecord.model_validate(first), RunRecord.model_validate(second)]

    matrix = build_matrix(records, {})
    assert matrix["cells"] == []
    assert len(matrix["invalid_cells"]) == 1
    invalid = matrix["invalid_cells"][0]
    assert invalid["track"] == "BB1"
    assert invalid["model"] == "lim_mit_huge"
    assert "input budgets" in invalid["reason"]


def test_annual_embedding_products_get_separate_tracks() -> None:
    """T11 AEF and T11O OlmoEarth must never share a matrix cell.

    Both products run the same UPerNet/MLP probe on the same PASTIS-HD patches;
    only the input product differs. If they shared a track, ``_rank`` would
    silently pick one and hide the comparison (plan 2026-11-06 section 5.5).
    """
    from rs_backbone_eval.matrix import is_reduced_run

    aef = payload(run_id="aef-cell", model_id="aef_annual_2019", task_id="T11", metric="mean_iou")
    aef["protocol_id"] = "aef_probe_upernet"
    aef["training"]["epochs_completed"] = 8
    olmo = payload(
        run_id="olmo-cell", model_id="olmoearth_v13_embedding", task_id="T11O", metric="mean_iou"
    )
    olmo["protocol_id"] = "olmoearth_v13_probe_upernet"
    olmo["training"]["epochs_completed"] = 8
    records = [RunRecord.model_validate(aef), RunRecord.model_validate(olmo)]

    matrix = build_matrix(records, {})
    tracks = {(cell["task"], cell["track"]) for cell in matrix["cells"]}
    assert tracks == {("T11", "BA"), ("T11O", "BOE")}
    assert matrix["invalid_cells"] == []

    # T11O's 8-epoch budget is official; a shorter run must stay non-formal.
    assert not is_reduced_run(records[1])
    olmo["training"]["epochs_completed"] = 2
    assert is_reduced_run(RunRecord.model_validate(olmo))


def test_diagnostic_panel_runs_stay_out_of_the_matrix() -> None:
    """A 20-epoch single-seed diagnostic must not sit beside the formal cell.

    Review 2026-11-07 P2: the doc said "not headline" but BOEM20 entered the
    matrix and Portal. ``panel: diagnostic`` is written to run.json by the
    worker and ``build_matrix`` drops it; the run stays in the ledger and the
    report's run list.
    """
    formal = payload(
        run_id="olmo-8ep", model_id="olmoearth_v13_embedding", task_id="T11O", metric="mean_iou"
    )
    formal["protocol_id"] = "olmoearth_v13_probe_upernet"
    formal["training"]["epochs_completed"] = 8
    diagnostic = payload(
        run_id="olmo-20ep", model_id="olmoearth_v13_embedding", task_id="T11O", metric="mean_iou"
    )
    diagnostic["protocol_id"] = "olmoearth_v13_probe_upernet_20ep"
    diagnostic["training"]["epochs_completed"] = 20
    diagnostic["tags"] = {"panel": "diagnostic"}
    records = [RunRecord.model_validate(formal), RunRecord.model_validate(diagnostic)]

    matrix = build_matrix(records, {})
    tracks = {cell["track"] for cell in matrix["cells"]}
    assert tracks == {"BOE"}
    assert all(cell["runs"] != ["olmo-20ep"] for cell in matrix["cells"])
