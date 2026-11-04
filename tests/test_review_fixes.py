"""Regression tests for the review findings R1, R3, R7 and R9."""

from __future__ import annotations

from rs_backbone_eval.contracts import RunRecord
from rs_backbone_eval.matrix import (
    _d2_by_model,
    build_matrix,
    compatibility_matrix,
    perturbation_summary,
    scorecard,
)


def payload(
    *,
    run_id: str,
    model_id: str = "clay_v1_5",
    seed: int = 42,
    value: float = 0.8,
    lr: str = "1e-4",
    protocol_id: str = "full_finetune",
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
        "protocol_id": protocol_id,
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


def test_review_20261104_shrunken_eval_never_enters_a_headline() -> None:
    """A shrunk-test-eval run must not become a formal candidate.

    Review 2026-11-04 #3 (reviewer 2): a run can hit the full training budget
    and still evaluate only a shrunken test subset; without a guard it would
    sit in the headline. The T11 workers now record both counts in the tags,
    and matrix.formal_candidates excludes records whose tags disagree. Runs
    without the tags (every historical worker) keep their admission.
    """
    shrunken = RunRecord.model_validate(
        payload(
            run_id="shrunk-eval",
            tags={"test_samples_evaluated": "1", "test_manifest_samples": "496"},
        )
    )
    full = RunRecord.model_validate(
        payload(
            run_id="full-eval",
            tags={"test_samples_evaluated": "496", "test_manifest_samples": "496"},
        )
    )
    legacy = RunRecord.model_validate(payload(run_id="legacy-no-tag", seed=73))
    ids = [run for cell in cells_for([shrunken, full, legacy]) for run in cell["runs"]]
    assert "shrunk-eval" not in ids
    assert "full-eval" in ids and "legacy-no-tag" in ids


def test_review_20261104_rank_tiebreak_is_order_free() -> None:
    """Two variants with an identical admission must not flip by input order.

    ``_rank`` used to leave exact ties to the insertion order, so the headline
    could switch between heads when the ledger was read in a different order
    (reviewer 2026-11-04). The protocol id is now the final tiebreaker.
    """
    first = RunRecord.model_validate(
        payload(run_id="first-variant", protocol_id="b_proto", tags={"selection": "1"})
    )
    last = RunRecord.model_validate(
        payload(run_id="last-variant", protocol_id="a_proto", tags={"selection": "1"})
    )
    forward = [c["protocol"] for c in build_matrix([first, last], {})["cells"]]
    backward = [c["protocol"] for c in build_matrix([last, first], {})["cells"]]
    assert forward == backward == ["b_proto"]


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


def _perturbed(run_id: str, init: str | None = None, seed: int = 42) -> RunRecord:
    tags = {"init": init} if init else {}
    return RunRecord.model_validate(
        payload(
            run_id=run_id,
            seed=seed,
            tags=tags,
            extra_metrics=[
                {
                    "name": "water_iou",
                    "value": 0.6,
                    "split": "test_band_drop-NIR",
                    "direction": "maximize",
                    "is_primary": False,
                }
            ],
        )
    )


def test_r7_d2_keeps_init_panels_separate_and_canonicalises() -> None:
    records = [
        _perturbed("run-pre"),
        _perturbed("run-rand", init="random"),
        # a retry of the pretrained seed must not add a second observation
        _perturbed("run-pre-retry"),
    ]
    rows = _d2_by_model(perturbation_summary(records))
    panels = {(row["init"], row["label_fraction"]) for row in rows["clay_v1_5"]}
    assert panels == {("pretrained", "1.0"), ("random", "1.0")}
    pretrained = [row for row in rows["clay_v1_5"] if row["init"] == "pretrained"][0]
    assert pretrained["seeds"] == [42]  # the retry collapsed into one seed


def test_r8_failure_rate_only_counts_terminal_training_runs() -> None:
    probe = RunRecord.model_validate(payload(run_id="run-probe", metrics=[]))
    probe.training.head_id = "none"
    training = RunRecord.model_validate(payload(run_id="run-train"))
    card = scorecard(build_matrix([probe, training], {}), [probe, training], {})
    row = {r["model"]: r for r in card["models"]}["clay_v1_5"]
    assert row["d4"]["training_runs"] == 1
    assert row["d4"]["failed_runs"] == 0
    assert row["d4"]["failure_rate"] == 0.0
    assert row["d4"]["excluded_probes"] == 1


def test_review_round2_batch_override_is_not_a_formal_candidate() -> None:
    """A full-epoch run with a halved batch must not pass for formal (reviewer
    round-2 #3): 8 completed epochs with batch_override=2 hide an eighth of
    the effective budget behind unchanged step/epoch counters. The tag is
    only written on the diagnostic path, so its presence means reduced."""
    diag = RunRecord.model_validate(payload(run_id="half-batch", tags={"batch_override": "2"}))
    formal = RunRecord.model_validate(payload(run_id="plain-formal"))
    cells = cells_for([diag, formal])
    assert all("half-batch" not in cell.get("runs", []) for cell in cells)
    assert any("plain-formal" in cell.get("runs", []) for cell in cells)


def test_t3_b0_note_is_conditional_and_does_not_claim_head_causality() -> None:
    from rs_backbone_eval.report import _t3_b0_note

    within = _t3_b0_note(0.1558, 0.0298, 3, [0.1536, 0.1577, 0.1643, 0.1766])
    assert "0.1536–0.1766" in within and "落在" in within
    assert "不能把分差归因于任务头" in within
    assert "天花板不在头" not in within

    assert "高于" in _t3_b0_note(0.2, 0.01, 3, [0.15, 0.17])
    assert "低于" in _t3_b0_note(0.1, 0.01, 3, [0.15, 0.17])

    from rs_backbone_eval.report import _band_relation

    # The outstanding-work section must use the same score-derived relation.
    assert "同档" in _band_relation(0.1558, [0.1536, 0.1766])
    assert "同档" not in _band_relation(0.2, [0.15, 0.17])
    assert "高于" in _band_relation(0.2, [0.15, 0.17])
    assert "低于" in _band_relation(0.1, [0.15, 0.17])
