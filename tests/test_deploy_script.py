"""CPU-only tests for scripts/deploy.sh, run against a throwaway git repo.

--skip-sync is used so no uv environments are built.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy.sh"


def _clean_env() -> dict[str, str]:
    """Drop GIT_* variables. Git sets GIT_DIR for pre-push hooks, and inherited, it would
    point these commands at the pushing repo instead of the temporary ones."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=_clean_env()
    ).stdout.strip()


def _source(tmp_path: Path) -> tuple[str, Path]:
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    _git(src, "config", "user.email", "t@example.com")
    _git(src, "config", "user.name", "t")
    (src / "a.txt").write_text("1", encoding="utf-8")
    _git(src, "add", "a.txt")
    _git(src, "commit", "-q", "-m", "one")
    bundle = tmp_path / "src.bundle"
    _git(src, "bundle", "create", str(bundle), "--all")
    return _git(src, "rev-parse", "HEAD"), bundle


def _empty_queue(tmp_path: Path) -> Path:
    queue = tmp_path / "queue"
    for state in ("running", "pending"):
        (queue / state).mkdir(parents=True, exist_ok=True)
    return queue


def _deploy(
    tmp_path: Path, target: Path, *args: str, queue: Path | None = None
) -> subprocess.CompletedProcess:
    env = {**_clean_env(), "RS_EVAL_QUEUE_ROOT": str(queue or _empty_queue(tmp_path))}
    return subprocess.run(
        ["bash", str(SCRIPT), "--target", str(target), "--skip-sync", *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=env,
    )


def test_deploys_pinned_commit_and_writes_record(tmp_path: Path) -> None:
    commit, bundle = _source(tmp_path)
    target = tmp_path / "deploy"
    result = _deploy(tmp_path, target, "--ref", commit, "--source", str(bundle))
    assert result.returncode == 0, result.stderr
    assert _git(target, "rev-parse", "HEAD") == commit
    assert _git(target, "status", "--porcelain") == ""
    record = json.loads((tmp_path / "deploy.deploy.json").read_text(encoding="utf-8"))
    assert record["commit"] == commit
    assert record["source"] == str(bundle)


def test_refuses_target_with_local_changes(tmp_path: Path) -> None:
    commit, bundle = _source(tmp_path)
    target = tmp_path / "deploy"
    assert _deploy(tmp_path, target, "--ref", commit, "--source", str(bundle)).returncode == 0
    (target / "a.txt").write_text("edited", encoding="utf-8")
    result = _deploy(tmp_path, target, "--ref", commit, "--source", str(bundle))
    assert result.returncode != 0
    assert "local changes" in result.stderr


def test_missing_arguments_exit_2(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT), "--target", str(tmp_path / "x")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


def test_ref_branch_name_resolves_to_fetched_source_not_stale_local_branch(
    tmp_path: Path,
) -> None:
    # The target keeps a local "main" at the old commit. --ref main must pick the
    # freshly fetched source main instead.
    old, bundle_old = _source(tmp_path)
    src = tmp_path / "src"
    (src / "a.txt").write_text("2", encoding="utf-8")
    _git(src, "commit", "-q", "-am", "two")
    new = _git(src, "rev-parse", "HEAD")
    bundle_new = tmp_path / "src2.bundle"
    _git(src, "bundle", "create", str(bundle_new), "--all")
    target = tmp_path / "deploy"
    assert _deploy(tmp_path, target, "--ref", old, "--source", str(bundle_old)).returncode == 0
    _git(target, "branch", "-f", "main", old)
    assert _git(target, "rev-parse", "main") == old
    result = _deploy(tmp_path, target, "--ref", "main", "--source", str(bundle_new))
    assert result.returncode == 0, result.stderr
    assert _git(target, "rev-parse", "HEAD") == new


def test_unknown_ref_fails_even_when_a_local_branch_matches(tmp_path: Path) -> None:
    # "side" exists only as a local branch in the target, not in the source.
    commit, bundle = _source(tmp_path)
    target = tmp_path / "deploy"
    assert _deploy(tmp_path, target, "--ref", commit, "--source", str(bundle)).returncode == 0
    _git(target, "branch", "side", commit)
    result = _deploy(tmp_path, target, "--ref", "side", "--source", str(bundle))
    assert result.returncode != 0
    assert "not found" in result.stderr


def test_refuses_while_a_job_is_pending(tmp_path: Path) -> None:
    commit, bundle = _source(tmp_path)
    queue = _empty_queue(tmp_path)
    (queue / "pending" / "20261119T000000Z_x.json").write_text("{}", encoding="utf-8")
    target = tmp_path / "deploy"
    result = _deploy(tmp_path, target, "--ref", commit, "--source", str(bundle), queue=queue)
    assert result.returncode != 0
    assert "queue has 1" in result.stderr
    assert not target.exists()


def test_allow_busy_overrides_the_queue_check(tmp_path: Path) -> None:
    commit, bundle = _source(tmp_path)
    queue = _empty_queue(tmp_path)
    (queue / "running" / "20261119T000000Z_x.json").write_text("{}", encoding="utf-8")
    result = _deploy(
        tmp_path,
        tmp_path / "deploy",
        "--ref",
        commit,
        "--source",
        str(bundle),
        "--allow-busy",
        queue=queue,
    )
    assert result.returncode == 0, result.stderr
