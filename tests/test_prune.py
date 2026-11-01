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


def make_run(root: Path, name: str, status: str, checkpoint: bool = True) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": name, "status": status}), encoding="utf-8")
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
