"""CPU tests for the git version recorded in run.json (worker_runtime._code_version)."""

from __future__ import annotations

import subprocess

from rs_backbone_eval import worker_runtime


def test_git_tree_wins_over_build_environment(monkeypatch) -> None:
    monkeypatch.setenv("RS_EVAL_GIT_VERSION", "from-build")

    def clean_tree(cmd, **kw):
        out = "" if "status" in cmd else "abc123\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    monkeypatch.setattr(worker_runtime.subprocess, "run", clean_tree)
    assert worker_runtime._code_version() == "abc123"


def test_image_without_git_uses_build_version(monkeypatch) -> None:
    monkeypatch.setenv("RS_EVAL_GIT_VERSION", "def456-dirty")

    def no_git(cmd, **kw):
        raise subprocess.CalledProcessError(128, cmd)

    monkeypatch.setattr(worker_runtime.subprocess, "run", no_git)
    assert worker_runtime._code_version() == "def456-dirty"


def test_no_git_and_no_build_version_is_none(monkeypatch) -> None:
    monkeypatch.delenv("RS_EVAL_GIT_VERSION", raising=False)

    def no_git(cmd, **kw):
        raise OSError("no git")

    monkeypatch.setattr(worker_runtime.subprocess, "run", no_git)
    assert worker_runtime._code_version() is None
