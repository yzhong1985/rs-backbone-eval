"""Tests for the final-checkpoint prune helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    target = ROOT / "scripts" / "prune_final_checkpoints.py"
    spec = importlib.util.spec_from_file_location("prune_final_checkpoints", target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_run(root: Path, run_id: str, *, status: str = "completed") -> Path:
    directory = root / run_id
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "tags": {},
                "artifacts": [
                    {"kind": "checkpoint", "path": "final_checkpoint.pt"},
                    {"kind": "log", "path": "train.log"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (directory / "final_checkpoint.pt").write_bytes(b"checkpoint-bytes")
    return directory


def test_dry_run_keeps_the_checkpoint(tmp_path: Path) -> None:
    module = load_module()
    run_dir = _write_run(tmp_path, "run")
    before = (run_dir / "final_checkpoint.pt").read_bytes()
    assert module.prune_one(
        tmp_path,
        "run",
        "keep winner",
        apply=False,
        log_path=tmp_path.parent / "checkpoint_prune.log",
    )
    assert (run_dir / "final_checkpoint.pt").read_bytes() == before
    assert json.loads((run_dir / "run.json").read_text())["artifacts"]


def test_apply_deletes_only_the_final_checkpoint_and_logs_it(tmp_path: Path) -> None:
    module = load_module()
    run_dir = _write_run(tmp_path, "run")
    log_path = tmp_path.parent / "checkpoint_prune.log"
    assert module.prune_one(tmp_path, "run", "keep winner", apply=True, log_path=log_path)
    assert not (run_dir / "final_checkpoint.pt").exists()
    record = json.loads((run_dir / "run.json").read_text())
    assert [artifact["path"] for artifact in record["artifacts"]] == ["train.log"]
    assert record["tags"]["checkpoint_pruned"] == "keep winner"
    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["run_id"] == "run"
    assert entry["bytes"] == len(b"checkpoint-bytes")


def test_running_run_is_refused(tmp_path: Path) -> None:
    module = load_module()
    run_dir = _write_run(tmp_path, "running", status="running")
    assert not module.prune_one(
        tmp_path,
        "running",
        "reason",
        apply=True,
        log_path=tmp_path.parent / "checkpoint_prune.log",
    )
    assert (run_dir / "final_checkpoint.pt").exists()


def test_missing_checkpoint_is_already_pruned(tmp_path: Path) -> None:
    module = load_module()
    run_dir = _write_run(tmp_path, "run")
    (run_dir / "final_checkpoint.pt").unlink()
    assert module.prune_one(
        tmp_path,
        "run",
        "reason",
        apply=True,
        log_path=tmp_path.parent / "checkpoint_prune.log",
    )
