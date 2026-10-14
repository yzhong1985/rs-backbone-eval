#!/usr/bin/env python3
"""Audit historical runs against the current evaluation protocol.

Classifies every run record under ``RS_EVAL_RUNS`` (or the first argument) by
whether its code version is at or after the commit that last changed the
semantics of its path. Report-only fixes (experiment identity, D2 split
classification, D1 common-task set) change aggregation but not the stored
metric values, so they are not part of the threshold.

Usage: ``python3 scripts/audit_runs.py [runs_root] [repo_root]``
"""

from __future__ import annotations

import collections
import datetime as dt
import glob
import json
import subprocess
import sys
from pathlib import Path

# Commits that changed stored metric values, by path.
P0 = "ffdc6db"  # seed_everything + adapter gradient + Clay B1 unfreeze
T5 = "3b4460a"  # T5 head trained on the full label grid
R6 = "35a972e"  # frozen-path loss on the full label grid (T1/T2/T3, Clay T3)


def commit_date(repo: Path, sha: str | None) -> dt.datetime | None:
    if not sha or sha == "none":
        return None
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", sha],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return dt.datetime.fromisoformat(out)
    except (subprocess.CalledProcessError, ValueError):
        return None


def required(
    task: str, model: str, regime: str, dates: dict[str, dt.datetime]
) -> tuple[str, dt.datetime]:
    """(reason, earliest acceptable commit date) for one path."""
    if regime == "reference":
        return "B0 参考复现（已知选模/权重恢复缺陷）", dates[P0]
    need, reason = dates[P0], "P0（seed/适配器梯度）"
    if task == "T5" and model in {"prithvi_eo_v2_300_tl", "clay_v1_5"}:
        need, reason = max(need, dates[T5]), "T5 完整标签网格"
    if task in {"T1", "T2", "T3"} and model == "prithvi_eo_v2_300_tl" and regime == "frozen_probe":
        need, reason = max(need, dates[R6]), "R6 冻结路径完整标签网格"
    if task == "T3" and model == "clay_v1_5" and regime == "frozen_probe":
        need, reason = max(need, dates[R6]), "R6 冻结路径完整标签网格"
    return reason, need


def main() -> int:
    runs_root = Path(sys.argv[1] if len(sys.argv) > 1 else "/geodata/rs-eval/runs")
    repo = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    dates = {sha: commit_date(repo, sha) for sha in (P0, T5, R6)}

    runs = []
    for path in sorted(glob.glob(str(runs_root / "*" / "run.json"))):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        training = record.get("training") or {}
        metrics = record.get("metrics") or []
        runs.append(
            {
                "run_id": record.get("run_id"),
                "task": record.get("task_id"),
                "model": record.get("model_id"),
                "regime": training.get("regime"),
                "seed": record.get("seed"),
                "status": record.get("status"),
                "commit": record.get("git_commit"),
                "value": metrics[0].get("value") if metrics else None,
            }
        )

    cells: dict[tuple, list[dict]] = collections.defaultdict(list)
    for run in runs:
        cells[(run["task"], run["model"], run["regime"])].append(run)

    print(f"总运行 {len(runs)} 条；未记录代码版本 {sum(1 for r in runs if not r['commit'])} 条\n")
    print("| 任务 | 模型 | 轨道 | 运行数 | 种子 | 状态 | 代码版本 | 判定 |")
    print("|---|---|---|---:|---|---|---|---|")
    for key in sorted(cells, key=lambda k: (k[0] or "", k[1] or "", k[2] or "")):
        task, model, regime = key
        group = cells[key]
        seeds = sorted({r["seed"] for r in group})
        status = ",".join(sorted({str(r["status"]) for r in group}))
        known = [commit_date(repo, r["commit"]) for r in group]
        known = [d for d in known if d]
        reason, need = required(task or "", model or "", regime or "", dates)
        if not known:
            verdict = f"❌ 无法判定（{len(group)} 条均未记录代码版本）"
        elif min(known) >= need:
            verdict = f"✅ 满足（{reason}）"
        elif max(known) < need:
            verdict = f"❌ 需重跑（{reason}）"
        else:
            verdict = f"⚠️ 混合（{reason}）"
        span = "无" if not known else f"{min(known).date()}~{max(known).date()}"
        print(
            f"| {task} | {model} | {regime} | {len(group)} | {seeds} | {status} "
            f"| {span} | {verdict} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
