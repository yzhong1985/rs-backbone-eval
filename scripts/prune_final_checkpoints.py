#!/usr/bin/env python3
"""Prune ``final_checkpoint.pt`` from completed runs, with an audit trail.

Only the fixed file ``final_checkpoint.pt`` is ever removed. ``run.json``,
logs, ``best_checkpoint.pt``, ``head_only.pt``, ``latest.pt`` and the run
directory itself are left in place. The script records the sha256/size before
deleting, removes the artifact entry from ``run.json`` and adds a
``checkpoint_pruned`` tag, so a pruned run no longer advertises a missing file.

Dry-run is the default; pass ``--apply`` to delete.

    python3 scripts/prune_final_checkpoints.py /geodata/rs-eval/runs \
        --reason "LR screen: keep the lr5e5 winner" \
        --remove RUN_ID [RUN_ID ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

FILENAME = "final_checkpoint.pt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def prune_one(
    runs_root: Path,
    run_id: str,
    reason: str,
    *,
    apply: bool,
    log_path: Path,
) -> bool:
    run_dir = runs_root / run_id
    record_path = run_dir / "run.json"
    if not record_path.is_file():
        print(f"missing: {record_path}")
        return False
    record = json.loads(record_path.read_text(encoding="utf-8"))
    status = record.get("status")
    if status not in {"completed", "failed"}:
        print(f"refusing {run_id}: status is {status!r}, not completed/failed")
        return False
    checkpoint = run_dir / FILENAME
    if not checkpoint.is_file():
        print(f"already pruned: {run_id}")
        return True
    digest = _sha256(checkpoint)
    size = checkpoint.stat().st_size
    if not apply:
        print(f"would prune: {run_id} ({size / 1e9:.2f} GB, sha256 {digest[:12]}…)")
        return True
    checkpoint.unlink()
    artifacts = [
        artifact for artifact in record.get("artifacts", []) if artifact.get("path") != FILENAME
    ]
    record["artifacts"] = artifacts
    record.setdefault("tags", {})["checkpoint_pruned"] = reason
    _write_json_atomic(record_path, record)
    entry = {
        "time": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "file": FILENAME,
        "bytes": size,
        "sha256": digest,
        "reason": reason,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"pruned: {run_id} ({size / 1e9:.2f} GB)")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--remove", nargs="+", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if not args.reason.strip():
        parser.error("--reason must not be empty")
    log_path = args.runs_root.parent / "checkpoint_prune.log"
    failures = 0
    for run_id in args.remove:
        if not prune_one(
            args.runs_root,
            run_id,
            args.reason,
            apply=args.apply,
            log_path=log_path,
        ):
            failures += 1
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
