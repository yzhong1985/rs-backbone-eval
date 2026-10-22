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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rs_backbone_eval.audit import (  # noqa: E402
    B0_RESTORE,
    OK,
    P0,
    P0_TRAIN,
    R6_GRID,
    STALE,
    T4_METRIC,
    T5_GRID,
    UNKNOWN,
    Ancestry,
    excluded_run_ids,
    required_fixes,
    stale_run_ids,
)

# The checks live in the package (the report uses them too); this script keeps
# re-exporting the names its tests and the review notes refer to.
__all__ = [
    "P0",
    "P0_TRAIN",
    "B0_RESTORE",
    "T4_METRIC",
    "T5_GRID",
    "R6_GRID",
    "OK",
    "UNKNOWN",
    "STALE",
    "Ancestry",
    "excluded_run_ids",
    "required_fixes",
    "stale_run_ids",
    "main",
]


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
                "trainable": training.get("trainable_parameters"),
                "total": training.get("total_parameters"),
                "superseded": bool((record.get("tags") or {}).get("superseded")),
            }
        )

    # A superseded run is already excluded from every conclusion, so it is not
    # re-judged here; the ledger keeps it for the audit trail.
    live = [run for run in runs if not run["superseded"]]

    # A record whose regime is full_finetune but which only trained the head is
    # internally inconsistent: the label promises a B1 run while the weights say
    # frozen backbone. That defect produced a head-only Clay cell that looked
    # like a full fine-tune, so it now fails the audit.
    inconsistent = [
        run
        for run in live
        if run["regime"] == "full_finetune"
        and run["trainable"]
        and run["total"]
        and run["trainable"] < 0.5 * run["total"]
    ]

    cells: dict[tuple, list[dict]] = collections.defaultdict(list)
    for run in live:
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
    if inconsistent:
        print(
            f"\n❌ 记录与训练语义不一致 {len(inconsistent)} 条"
            "（regime=full_finetune，但训练参数 < 50%）："
        )
        for run in inconsistent:
            share = 100 * run["trainable"] / run["total"]
            print(
                f"- `{run['run_id']}`：{run['task']} {run['model']} "
                f"trainable={run['trainable']:,} / {run['total']:,}（{share:.1f}%）"
            )
    else:
        print("\n✅ 没有记录与训练语义不一致的运行。")
    return 1 if stale_total or inconsistent else 0


if __name__ == "__main__":
    raise SystemExit(main())
