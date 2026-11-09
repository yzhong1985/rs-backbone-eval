"""Lightning ``last.ckpt`` retention: only completed runs, only with a best."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from rs_backbone_eval.prune import plan_last_ckpt, summarise_last_ckpt

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location(
        "prune_checkpoints", ROOT / "scripts" / "prune_checkpoints.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_run(root: Path, name: str, status: str, *, best: bool = True) -> Path:
    run = root / name
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"run_id": name, "status": status}), encoding="utf-8")
    (checkpoints / "last.ckpt").write_bytes(b"x" * 1024)
    if best:
        (checkpoints / "best-epoch=01.ckpt").write_bytes(b"x" * 512)
    return run


def test_plan_drops_only_completed_runs_with_a_best(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    make_run(runs, "done", "completed")
    make_run(runs, "running", "running")
    make_run(runs, "failed", "failed")
    make_run(runs, "no-best", "completed", best=False)

    entries = {entry.run_id: entry for entry in plan_last_ckpt(runs)}
    assert entries["done"].action == "delete-last-ckpt"
    assert entries["running"].action == "skip"
    assert entries["failed"].action == "skip"
    assert entries["no-best"].action == "skip"
    assert "no best" in entries["no-best"].reason
    summary = summarise_last_ckpt(list(entries.values()))
    assert summary["delete_last_ckpt"] == 1
    assert summary["reclaimable_bytes"] == 1024


def test_apply_removes_last_and_keeps_best(tmp_path: Path, monkeypatch) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    run = make_run(runs, "done", "completed")
    module = load_script()
    monkeypatch.setattr(
        sys,
        "argv",
        ["prune_checkpoints.py", "--runs-root", str(runs), "--include-last-ckpt", "--apply"],
    )
    assert module.main() == 0
    assert not (run / "checkpoints" / "last.ckpt").exists()
    assert (run / "checkpoints" / "best-epoch=01.ckpt").is_file()
    assert (run / "run.json").is_file()
    log = (tmp_path / "checkpoint_prune.log").read_text(encoding="utf-8").splitlines()
    record = json.loads(log[-1])
    assert record["action"] == "delete-last-ckpt"
    assert record["deleted_bytes"] == 1024


def test_dry_run_keeps_everything(tmp_path: Path, monkeypatch) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    run = make_run(runs, "done", "completed")
    module = load_script()
    monkeypatch.setattr(
        sys, "argv", ["prune_checkpoints.py", "--runs-root", str(runs), "--include-last-ckpt"]
    )
    assert module.main() == 0
    assert (run / "checkpoints" / "last.ckpt").is_file()
