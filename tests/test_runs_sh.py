"""CPU tests for scripts/runs.sh: pending-job matching for `stop`, and spec errors for `enqueue`.

Pending-job tests source the script (its dispatch is guarded) and call has_pending_job with
RS_EVAL_QUEUE_ROOT pointing at a temporary directory. The dry-run tests execute the script
with the root uv environment, which has PyYAML.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "runs.sh"
RUN_ID = "lim_mit_huge-T13-whu_b0-train-73"
QUEUED = "20261118T113516Z_lim_mit_huge-T13-whu_b0-train-73.json"


def _source_call(queue_root: Path, run_id: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", 'source "$1"; has_pending_job "$2"', "_", str(SCRIPT), run_id],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "RS_EVAL_QUEUE_ROOT": str(queue_root)},
    )


def _has_pending(queue_root: Path, run_id: str) -> int:
    return _source_call(queue_root, run_id).returncode


def _pending(tmp_path: Path, *names: str) -> Path:
    queue = tmp_path / "queue"
    pending = queue / "pending"
    pending.mkdir(parents=True)
    for name in names:
        (pending / name).write_text("{}", encoding="utf-8")
    return queue


def test_stop_finds_a_timestamped_pending_job(tmp_path: Path) -> None:
    queue = _pending(tmp_path, QUEUED)
    assert _has_pending(queue, RUN_ID) == 0


def test_stop_does_not_match_a_longer_run_id_with_the_same_suffix(tmp_path: Path) -> None:
    queue = _pending(tmp_path, "20261118T113516Z_lim_b0-train-73.json")
    assert _has_pending(queue, "b0-train-73") == 1


def test_stop_does_not_match_a_missing_run(tmp_path: Path) -> None:
    queue = _pending(tmp_path, QUEUED)
    assert _has_pending(queue, "lim_other-run") == 1


def test_stop_treats_dots_in_run_id_literally(tmp_path: Path) -> None:
    # An unescaped "." would match "runAx1"; the escaped one must not.
    queue = _pending(tmp_path, "20261118T113516Z_runAx1.json")
    assert _has_pending(queue, "run.1") == 1
    queue = _pending(tmp_path / "second", "20261118T113516Z_run.1.json")
    assert _has_pending(queue, "run.1") == 0


def test_run_id_with_regex_metacharacters_is_refused(tmp_path: Path) -> None:
    queue = _pending(tmp_path, QUEUED)
    result = _source_call(queue, "run*")
    assert result.returncode != 0
    assert "characters outside" in result.stderr


def _enqueue_dry(tmp_path: Path, body: str) -> subprocess.CompletedProcess:
    spec = tmp_path / "spec.yaml"
    spec.write_text(body, encoding="utf-8")
    return subprocess.run(
        ["bash", str(SCRIPT), "enqueue", str(spec), "--dry-run"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_dry_run_with_a_missing_seed_fails_and_says_so(tmp_path: Path) -> None:
    body = "task: T13\nmodel: lim_mit_huge\nprotocol: frozen_probe\n"
    result = _enqueue_dry(tmp_path, body)
    assert result.returncode != 0
    assert "spec missing" in result.stdout + result.stderr
    assert "rs-eval queue add" not in result.stdout


def test_dry_run_with_all_required_fields_prints_the_command(tmp_path: Path) -> None:
    body = "task: T13\nmodel: lim_mit_huge\nprotocol: frozen_probe\nseed: 73\n"
    result = _enqueue_dry(tmp_path, body)
    assert result.returncode == 0, result.stderr
    assert "rs-eval queue add" in result.stdout
    assert "--seed 73" in result.stdout
