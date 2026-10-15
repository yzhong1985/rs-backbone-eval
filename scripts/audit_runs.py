#!/usr/bin/env python3
"""Audit historical runs against the current evaluation protocol.

For every run record, check whether each fix that changed that path's stored
metrics is an **ancestor** of the run's recorded commit. Commit *time* is not
used: two commits can share a timestamp (for example 1445a0b and the later
fixes), so only ancestry is reliable. A record with no recorded commit is
reported as "unknown" and never folded into a passing group.

Usage: ``python3 scripts/audit_runs.py [runs_root] [repo_root]``
Exit code is 0 only when no record is stale (unknown records do not fail the
run, but they are counted and listed).
"""

from __future__ import annotations

import collections
import glob
import json
import subprocess
import sys
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


def main() -> int:
    runs_root = Path(sys.argv[1] if len(sys.argv) > 1 else "/geodata/rs-eval/runs")
    repo = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    ancestry = Ancestry(repo)

    runs = []
    for path in sorted(glob.glob(str(runs_root / "*" / "run.json"))):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        training = record.get("training") or {}
        runs.append(
            {
                "run_id": record.get("run_id"),
                "task": record.get("task_id"),
                "model": record.get("model_id"),
                "regime": training.get("regime"),
                "seed": record.get("seed"),
                "status": record.get("status"),
                "commit": record.get("git_commit"),
            }
        )

    cells: dict[tuple, list[dict]] = collections.defaultdict(list)
    for run in runs:
        cells[(run["task"], run["model"], run["regime"])].append(run)

    unknown_total = 0
    stale_total = 0
    print(f"总运行 {len(runs)} 条\n")
    print("| 任务 | 模型 | 轨道 | 运行数 | 种子 | 状态 | ✅ | ❌ | ？ | 判定 |")
    print("|---|---|---|---:|---|---|---:|---:|---:|---|")
    for key in sorted(cells, key=lambda k: (k[0] or "", k[1] or "", k[2] or "")):
        task, model, regime = key
        group = cells[key]
        seeds = sorted({r["seed"] for r in group})
        status = ",".join(sorted({str(r["status"]) for r in group}))
        fixes = required_fixes(task or "", model or "", regime or "")
        verdicts = [ancestry.verdict(r["commit"], fixes) for r in group]
        n_ok = verdicts.count(OK)
        n_stale = verdicts.count(STALE)
        n_unknown = verdicts.count(UNKNOWN)
        unknown_total += n_unknown
        stale_total += n_stale
        if n_ok == len(group):
            verdict = "✅ 满足"
        elif n_stale == len(group):
            verdict = f"❌ 需重跑（{'/'.join(fixes)}）"
        elif n_stale and n_ok:
            verdict = f"⚠️ 混合（{'/'.join(fixes)}）"
        elif n_unknown and n_ok:
            verdict = "⚠️ 含未记录代码版本"
        else:
            verdict = "❌ 无法判定（未记录代码版本）"
        print(
            f"| {task} | {model} | {regime} | {len(group)} | {seeds} | {status} "
            f"| {n_ok} | {n_stale} | {n_unknown} | {verdict} |"
        )
    print(f"\n汇总：需重跑 {stale_total} 条，未记录代码版本 {unknown_total} 条。")
    return 1 if stale_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
