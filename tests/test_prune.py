"""Checkpoint retention: what is kept, what is dropped, what is refused."""

from __future__ import annotations

import json
from pathlib import Path

from rs_backbone_eval.prune import plan, split_keys, summarise


def test_split_keeps_the_head_and_drops_the_frozen_encoder() -> None:
    keys = [
        "detector.backbone.patch_embed1.proj.weight",
        "detector.fpn.lateral.0.weight",
        "detector.rpn_head.cls.weight",
        "detector.cls_score.weight",
    ]
    head, backbone = split_keys(keys)
    assert backbone == ["detector.backbone.patch_embed1.proj.weight"]
    assert "detector.cls_score.weight" in head and len(head) == 3

    clay = ["encoder.blocks.0.weight", "head.project.0.weight"]
    head, backbone = split_keys(clay)
    assert backbone == ["encoder.blocks.0.weight"] and head == ["head.project.0.weight"]


def run_payload(
    run_id: str,
    status: str,
    *,
    tags: dict[str, str] | None = None,
    notes: str | None = None,
    epochs: int = 50,
) -> dict:
    """A minimal run.json that passes ``RunRecord`` validation.

    T1's official budget is 50 epochs, so ``epochs=50`` is a formal-length run
    and anything shorter is a reduced-budget diagnostic.
    """
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": "2026-11-01T00:00:00Z",
        "updated_at": "2026-11-01T00:00:00Z",
        "status": status,
        "task_id": "T1",
        "model_id": "anysat_base",
        "protocol_id": "full_finetune",
        "seed": 17,
        "dataset": {
            "dataset_id": "Sen1Floods11",
            "version": "official",
            "manifest_sha256": "a" * 64,
            "split": "test",
        },
        "input": {
            "sensor": "sentinel-2",
            "bands": ["BLUE", "GREEN", "RED", "NIR_NARROW"],
            "time_steps": 1,
            "gsd_m": 10,
            "preprocessing_id": "s2-six-band-v1",
        },
        "training": {
            "regime": "full_finetune",
            "head_id": "upernet",
            "epochs_completed": epochs,
            "optimizer_steps": 320,
        },
        "metrics": [],
        "curves": [],
        "artifacts": [],
        "tags": tags or {},
        "notes": notes,
    }


def make_run(
    root: Path,
    name: str,
    status: str,
    checkpoint: bool = True,
    **record: object,
) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps(run_payload(name, status, **record)), encoding="utf-8")
    if checkpoint:
        (run / "latest.pt").write_bytes(b"x" * 1024)
    return run


def test_plan_only_touches_completed_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    make_run(runs, "done-1", "completed")
    make_run(runs, "done-2", "completed")
    make_run(runs, "running-1", "running")
    make_run(runs, "failed-1", "failed")
    make_run(runs, "no-ckpt", "completed", checkpoint=False)

    entries = {entry.run_id: entry for entry in plan(runs)}
    assert entries["done-1"].action == "export-and-delete"
    assert entries["done-2"].action == "export-and-delete"
    assert entries["running-1"].action == "skip"
    assert entries["failed-1"].action == "skip"
    assert "no-ckpt" not in entries
    assert summarise(list(entries.values()))["export_and_delete"] == 2


def test_existing_head_only_turns_into_a_plain_delete(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = make_run(runs, "done-1", "completed")
    (run / "head_only.pt").write_bytes(b"y")
    entry = next(iter(plan(runs)))
    assert entry.action == "delete"
    assert entry.has_head_only


def test_age_window_skips_young_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    make_run(runs, "done-1", "completed")
    entry = next(iter(plan(runs, older_than_days=7)))
    assert entry.action == "skip"
    assert "younger" in entry.reason


def test_missing_run_json_is_not_assumed_completed(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "weird"
    run.mkdir(parents=True)
    (run / "latest.pt").write_bytes(b"z")
    entry = next(iter(plan(runs)))
    assert entry.action == "skip" and entry.status == "unknown"


def test_diagnostic_panel_is_never_pruned(tmp_path: Path) -> None:
    """20-epoch probes are completed runs but only diagnostics (review 2026-11-10)."""
    runs = tmp_path / "runs"
    make_run(runs, "diag-20ep", "completed", tags={"panel": "diagnostic"})
    make_run(runs, "formal", "completed")

    entries = {entry.run_id: entry for entry in plan(runs)}
    assert entries["diag-20ep"].action == "skip"
    assert "panel=diagnostic" in entries["diag-20ep"].reason
    assert entries["formal"].action == "export-and-delete"


def test_reduced_budget_run_is_never_pruned(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    make_run(runs, "short-note", "completed", notes="REDUCED-BUDGET diagnostic")
    make_run(runs, "short-counter", "completed", epochs=1)

    entries = {entry.run_id: entry for entry in plan(runs)}
    assert entries["short-note"].action == "skip"
    assert "reduced-budget" in entries["short-note"].reason
    assert entries["short-counter"].action == "skip"
    assert "budget" in entries["short-counter"].reason


def test_unreadable_run_json_fails_closed(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = make_run(runs, "broken", "completed")
    (run / "run.json").write_text("not json", encoding="utf-8")

    entry = next(iter(plan(runs)))
    # Status is unknown, so the policy already refuses; the protection reason
    # only matters when a run.json validates but is not a formal candidate.
    assert entry.action == "skip"


def test_shrunken_eval_is_never_pruned(tmp_path: Path) -> None:
    """Full budget but a truncated test fold is refused by admission (2026-11-10 r2)."""
    runs = tmp_path / "runs"
    make_run(
        runs,
        "short-eval",
        "completed",
        tags={"test_samples_evaluated": "120", "test_manifest_samples": "900"},
    )
    make_run(
        runs,
        "short-eval-vs-official",
        "completed",
        tags={"test_expected_samples": "900", "test_samples_evaluated": "120"},
    )
    make_run(runs, "formal", "completed")

    entries = {entry.run_id: entry for entry in plan(runs)}
    assert entries["short-eval"].action == "skip"
    assert "shrunken test fold" in entries["short-eval"].reason
    assert "shrunken test fold" in entries["short-eval-vs-official"].reason
    assert entries["formal"].action == "export-and-delete"


def test_superseded_run_is_never_pruned(tmp_path: Path) -> None:
    """The ledger keeps superseded runs, so their resume state stays as well."""
    runs = tmp_path / "runs"
    make_run(runs, "old", "completed", tags={"superseded": "pre-fix rerun"})

    entry = next(iter(plan(runs)))
    assert entry.action == "skip"
    assert "superseded" in entry.reason
