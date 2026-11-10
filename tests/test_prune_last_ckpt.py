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


def make_run(root: Path, name: str, status: str, *, best: bool = True, **record: object) -> Path:
    run = root / name
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps(run_payload(name, status, **record)), encoding="utf-8")
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


def test_diagnostic_run_keeps_last_ckpt(tmp_path: Path) -> None:
    """A completed 20-epoch probe is a diagnostic, not a result (2026-11-10)."""
    runs = tmp_path / "runs"
    runs.mkdir()
    make_run(runs, "diag", "completed", tags={"panel": "diagnostic"})
    make_run(runs, "short", "completed", epochs=2)
    make_run(runs, "formal", "completed")

    entries = {entry.run_id: entry for entry in plan_last_ckpt(runs)}
    assert entries["diag"].action == "skip"
    assert "panel=diagnostic" in entries["diag"].reason
    assert entries["short"].action == "skip"
    assert entries["formal"].action == "delete-last-ckpt"
