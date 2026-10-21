"""Version-ancestry audit for run records.

A fix that changed stored metric values makes every run that predates it
incomparable with a run that includes it. Commit *time* is not enough (two
commits can share a timestamp), so the check asks git whether the fix is an
ancestor of the run's recorded commit. Records with no recorded commit are
"unknown" and never folded into a passing group.

Used by ``scripts/audit_runs.py`` (CLI) and by the report, which must not
present a cell built from pre-fix runs as a finished result.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Fixes that changed stored metric values, in landing order.
P0 = "ffdc6db"  # seed_everything + metric selection/aggregation
P0_TRAIN = "7377fb6"  # adapter gradient, Clay B1 unfreeze
B0_RESTORE = "ea9e8a4"  # B0 best-checkpoint restore + day-of-year
T5_GRID = "3b4460a"  # T5 head trained on the full label grid
R6_GRID = "35a972e"  # frozen-path loss on the full label grid

UNKNOWN = "unknown"
OK = "ok"
STALE = "stale"


def required_fixes(task: str, model: str, regime: str) -> list[str]:
    """Fixes whose absence makes a run's stored metrics incomparable."""
    fixes = [P0, P0_TRAIN]
    if regime == "reference":
        # B0 is the Prithvi reference reproduction; it needs the checkpoint
        # restore fix on top of the training fixes.
        fixes.append(B0_RESTORE)
    if task == "T5":
        # day-of-year encoding (ea9e8a4) plus the full-label grid for the
        # per-date encoders (Prithvi/Clay).
        fixes.append(B0_RESTORE)
        if model in {"prithvi_eo_v2_300_tl", "clay_v1_5"}:
            fixes.append(T5_GRID)
    if task == "T3":
        fixes.append(B0_RESTORE)
    if task in {"T1", "T2", "T3"} and model == "prithvi_eo_v2_300_tl" and regime == "frozen_probe":
        fixes.append(R6_GRID)
    if task == "T3" and model == "clay_v1_5" and regime == "frozen_probe":
        fixes.append(R6_GRID)
    # De-duplicate while keeping order.
    seen: list[str] = []
    for fix in fixes:
        if fix not in seen:
            seen.append(fix)
    return seen


class Ancestry:
    """Cached ``git merge-base --is-ancestor`` lookups."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self._cache: dict[tuple[str, str], bool] = {}

    def has(self, fix: str, commit: str) -> bool:
        key = (fix, commit)
        if key not in self._cache:
            result = subprocess.run(
                ["git", "merge-base", "--is-ancestor", fix, commit],
                cwd=self.repo,
                capture_output=True,
                check=False,
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(
                    f"git merge-base failed for {fix} -> {commit}: "
                    f"{result.stderr.decode(errors='replace').strip()}"
                )
            self._cache[key] = result.returncode == 0
        return self._cache[key]

    def verdict(self, commit: str | None, fixes: list[str]) -> str:
        if not commit:
            return UNKNOWN
        # ``_code_version`` appends ``-dirty`` when the tree had uncommitted
        # changes; ancestry is checked against the base commit.
        base = commit.split("-dirty", 1)[0]
        return OK if all(self.has(fix, base) for fix in fixes) else STALE


def stale_run_ids(records, repo: Path) -> set[str]:
    """Run ids whose recorded commit predates a fix that changed their metric.

    Unknown-commit records are not reported here: they are already handled by
    the superseded tag and by the CLI's own reporting.
    """
    ancestry = Ancestry(repo)
    stale: set[str] = set()
    for record in records:
        training = record.training
        fixes = required_fixes(record.task_id, record.model_id, training.regime)
        if ancestry.verdict(record.git_commit, fixes) == STALE:
            stale.add(record.run_id)
    return stale
