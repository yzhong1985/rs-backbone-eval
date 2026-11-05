"""Tests for the ledger supersede tag helper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    target = ROOT / "scripts" / "supersede_run.py"
    spec = importlib.util.spec_from_file_location("supersede_run", target)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_run(root: Path, run_id: str) -> Path:
    directory = root / run_id
    directory.mkdir(parents=True)
    path = directory / "run.json"
    path.write_text(
        json.dumps({"run_id": run_id, "status": "completed", "tags": {}}),
        encoding="utf-8",
    )
    return path


def test_dry_run_does_not_touch_the_record(tmp_path: Path) -> None:
    module = load_module()
    path = _write_run(tmp_path, "old-run")
    before = path.read_text(encoding="utf-8")
    assert module.supersede(tmp_path, "rerun v2", ["old-run"], apply=False) == 0
    assert path.read_text(encoding="utf-8") == before


def test_apply_sets_only_the_superseded_tag(tmp_path: Path) -> None:
    module = load_module()
    path = _write_run(tmp_path, "old-run")
    original = json.loads(path.read_text(encoding="utf-8"))
    assert module.supersede(tmp_path, "rerun v2", ["old-run"], apply=True) == 0
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["tags"]["superseded"] == "rerun v2"
    assert record["status"] == original["status"]
    assert record["run_id"] == original["run_id"]
    # idempotent
    assert module.supersede(tmp_path, "rerun v2", ["old-run"], apply=True) == 0


def test_missing_run_is_reported(tmp_path: Path) -> None:
    module = load_module()
    assert module.supersede(tmp_path, "reason", ["missing"], apply=False) == 1
