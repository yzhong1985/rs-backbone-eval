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
    task_id: str = "T1",
    head_id: str = "lite_spatial_head",
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
        "task_id": task_id,
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
            "head_id": head_id,
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


def test_review_round3_official_anchor_blocks_a_shrunk_manifest() -> None:
    """Round 3 #2: a shrunk manifest must not self-attest even when worker
    counts agree with it -- the official-count tag pins both sides."""
    shrunk_manifest = RunRecord.model_validate(
        payload(
            run_id="shrunk-manifest",
            tags={
                "test_samples_evaluated": "1",
                "test_manifest_samples": "1",
                "test_expected_samples": "496",
            },
        )
    )
    honest = RunRecord.model_validate(
        payload(
            run_id="honest-run",
            tags={
                "test_samples_evaluated": "496",
                "test_manifest_samples": "496",
                "test_expected_samples": "496",
            },
        )
    )
    ids = [run for cell in cells_for([shrunk_manifest, honest]) for run in cell["runs"]]
    assert "shrunk-manifest" not in ids
    assert "honest-run" in ids


def test_review_round6_a_shrunken_validation_split_blocks_admission() -> None:
    """Round 6 P2: the test fold can be complete while validation was cut to
    one row; that is still a protocol deviation and must not be formal."""
    shrunken_validation = RunRecord.model_validate(
        payload(
            run_id="val-shrunk",
            tags={
                "test_samples_evaluated": "496",
                "test_manifest_samples": "496",
                "test_expected_samples": "496",
                "validation_samples_evaluated": "1",
                "validation_manifest_samples": "1",
                "validation_expected_samples": "482",
            },
        )
    )
    honest = RunRecord.model_validate(
        payload(
            run_id="val-honest",
            tags={
                "test_samples_evaluated": "496",
                "test_manifest_samples": "496",
                "test_expected_samples": "496",
                "validation_samples_evaluated": "482",
                "validation_manifest_samples": "482",
                "validation_expected_samples": "482",
            },
        )
    )
    ids = [run for cell in cells_for([shrunken_validation, honest]) for run in cell["runs"]]
    assert "val-shrunk" not in ids
    assert "val-honest" in ids


def test_review_20261104_rank_tiebreak_is_order_free() -> None:
    """Ties must resolve by variant identity, not by protocol name.

    Two no-selection protocols in one cell are now rejected (round 10); with a
    validation score the better one wins in either input order.
    """
    first = RunRecord.model_validate(
        payload(
            run_id="first-variant",
            protocol_id="frozen_probe",
            tags={"selection": "best_validation", "validation_mean_iou": "0.7"},
        )
    )
    last = RunRecord.model_validate(
        payload(
            run_id="last-variant",
            protocol_id="frozen_adapter",
            tags={"selection": "best_validation", "validation_mean_iou": "0.9"},
        )
    )
    forward = [c["protocol"] for c in build_matrix([first, last], {})["cells"]]
    backward = [c["protocol"] for c in build_matrix([last, first], {})["cells"]]
    assert forward == backward == ["frozen_adapter"]


def test_review_round10_single_and_20bin_are_separate_cells() -> None:
    """A one-frame and a 20-frame protocol must both stay visible.

    They used to share the BB track, and the protocol-name tiebreaker picked
    ``lim_s2_single`` over ``lim_s2_20bin`` (review 2026-11-04 round 10).
    """
    single = RunRecord.model_validate(
        payload(run_id="lim-single", protocol_id="lim_s2_single", task_id="T11", value=0.19)
    )
    binned = RunRecord.model_validate(
        payload(run_id="lim-20bin", protocol_id="lim_s2_20bin", task_id="T11", value=0.41)
    )
    cells = build_matrix([single, binned], {})["cells"]
    by_track = {cell["track"]: cell for cell in cells}
    assert by_track["BB1"]["mean"] == 0.19
    assert by_track["BBT"]["mean"] == 0.41


def test_review_round10_mixed_input_budgets_quarantine_the_cell(monkeypatch) -> None:
    """A 20-bin protocol mapped onto the single-frame track is quarantined.

    Review 2026-11-05 round 13: raising ValueError here took down the whole
    report and Portal; the cell is now reported in ``invalid_cells`` instead.
    """
    from rs_backbone_eval import matrix

    monkeypatch.setitem(matrix.PROTOCOL_TRACKS, "lim_s2_20bin", "BB1")
    records = [
        RunRecord.model_validate(
            payload(run_id="lim-single", protocol_id="lim_s2_single", task_id="T11")
        ),
        RunRecord.model_validate(
            payload(run_id="lim-20bin", protocol_id="lim_s2_20bin", task_id="T11")
        ),
    ]
    result = build_matrix(records, {})
    assert result["cells"] == []
    assert "mixes input budgets" in result["invalid_cells"][0]["reason"]


def test_review_round10_validation_metric_drives_ranking() -> None:
    """Validation metrics in ``metrics`` count even without validation tags.

    T11 workers record the validation split as a metric, not a tag, so the old
    tag-only ``validation_score`` saw -inf and left the choice to the protocol
    name (review 2026-11-04 round 10).
    """

    def variant(run_id: str, head: str, validation: float) -> RunRecord:
        return RunRecord.model_validate(
            payload(
                run_id=run_id,
                head_id=head,
                extra_metrics=[
                    {
                        "name": "water_iou",
                        "value": validation,
                        "split": "validation",
                        "direction": "maximize",
                        "is_primary": False,
                    }
                ],
            )
        )

    low = variant("low", "head_a", 0.5)
    high = variant("high", "head_b", 0.7)
    for order in ([low, high], [high, low]):
        chosen = cells_for(list(order))[0]
        assert chosen["head"] == "head_b"


def test_review_round10_no_selection_protocols_in_one_cell_quarantine() -> None:
    """Multiple protocols without a validation score must not be ranked."""
    records = [
        RunRecord.model_validate(payload(run_id="probe", protocol_id="frozen_probe")),
        RunRecord.model_validate(payload(run_id="adapter", protocol_id="frozen_adapter")),
    ]
    result = build_matrix(records, {})
    assert result["cells"] == []
    assert "no validation" in result["invalid_cells"][0]["reason"]


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
