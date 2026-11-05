#!/usr/bin/env python3
"""Mark ledger runs as superseded without editing ``run.json`` by hand.

A rerun of a seed is not a new observation. ``canonical_runs`` already keeps the
newest formal record, but the old one should still carry a reason so the report
can show why it was replaced. This script is the supported way to add that tag:
it only touches ``tags.superseded`` and leaves every metric untouched.

Usage::

    python3 scripts/supersede_run.py /geodata/rs-eval/runs "reason" RUN_ID [RUN_ID ...]
    python3 scripts/supersede_run.py /geodata/rs-eval/runs "reason" RUN_ID --apply

The default is a dry run; ``--apply`` writes the tag.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def supersede(runs_root: Path, reason: str, run_ids: list[str], *, apply: bool) -> int:
    failures = 0
    for run_id in run_ids:
        path = runs_root / run_id / "run.json"
        if not path.is_file():
            print(f"missing: {path}")
            failures += 1
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        tags = record.setdefault("tags", {})
        if tags.get("superseded") == reason:
            print(f"already superseded: {run_id}")
            continue
        if apply:
            tags["superseded"] = reason
            path.write_text(
                json.dumps(record, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"superseded: {run_id}")
        else:
            print(f"would supersede: {run_id}")
    return 0 if failures == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("reason")
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.reason.strip():
        parser.error("reason must not be empty")
    return supersede(args.runs_root, args.reason, args.run_ids, apply=args.apply)


if __name__ == "__main__":
    raise SystemExit(main())
