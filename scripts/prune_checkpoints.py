#!/usr/bin/env python3
"""Export head-only weights for completed runs, then delete their `latest.pt`.

Dry-run by default; `--apply` performs the export+delete. Every action goes to
`<runs_root>/../checkpoint_prune.log` (JSONL) as the audit trail. Never touches
`run.json`, `best_checkpoint.pt`, logs or any other file.

    python scripts/prune_checkpoints.py --runs-root /geodata/rs-eval/runs
    python scripts/prune_checkpoints.py --runs-root /geodata/rs-eval/runs --apply

Exit codes: 0 ok, 1 nothing to do, 2 a run needed manual review (kept intact).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from rs_backbone_eval.prune import MAX_HEAD_PARAMS, plan, split_keys, summarise

FORMAT = "head-only-v1"


def log(path: Path, payload: dict) -> None:
    payload = {"time": datetime.now(UTC).isoformat(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def export_head_only(run_dir: Path) -> dict:
    """Write ``head_only.pt`` and return a summary; raises on anything odd."""
    import torch

    source = run_dir / "latest.pt"
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError(f"{source}: unsupported format_version {payload.get('format_version')}")
    state = payload["model"]
    head_keys, backbone_keys = split_keys(list(state.keys()))
    kept = {key: state[key] for key in head_keys}
    kept_params = sum(tensor.numel() for tensor in kept.values())
    total_params = sum(tensor.numel() for tensor in state.values())
    if not kept:
        raise ValueError("no head keys found")
    if kept_params > MAX_HEAD_PARAMS:
        raise ValueError(
            f"kept {kept_params:,} params (> {MAX_HEAD_PARAMS:,}); the backbone prefix "
            "was probably not recognised"
        )
    record = payload.get("run_record", {})
    export = {
        "format": FORMAT,
        "run_id": record.get("run_id", run_dir.name),
        "identity": payload.get("identity", {}),
        "task_id": record.get("task_id"),
        "model_id": record.get("model_id"),
        "protocol_id": record.get("protocol_id"),
        "seed": record.get("seed"),
        "git_commit": record.get("git_commit"),
        "pretrained_checkpoint": record.get("model_checkpoint_sha256"),
        "head_state_dict": kept,
        "head_params": kept_params,
        "total_params": total_params,
        "dropped_prefixes": sorted({key.split(".")[0] for key in backbone_keys}),
        "note": "Rebuild the frozen backbone from identity.pretrained_checkpoint (hash "
        "check included) and load head_state_dict to re-evaluate this run.",
    }
    target = run_dir / "head_only.pt"
    torch.save(export, target)
    return {
        "head_params": kept_params,
        "total_params": total_params,
        "head_keys": len(kept),
        "backbone_keys": len(backbone_keys),
        "head_bytes": target.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("/geodata/rs-eval/runs"))
    parser.add_argument("--older-than-days", type=float, default=0.0)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    entries = plan(args.runs_root, args.older_than_days)
    summary = summarise(entries)
    print(json.dumps(summary, ensure_ascii=False))
    audit = args.runs_root.parent / "checkpoint_prune.log"
    review_needed = False
    for entry in entries:
        if entry.action == "skip":
            print(f"skip   {entry.run_id}: {entry.reason}")
            continue
        run_dir = args.runs_root / entry.run_id
        if not args.apply:
            print(f"would  {entry.run_id}: {entry.action} ({entry.size_bytes / 1e9:.2f} GB)")
            log(audit, {"action": "dry-run", "run_id": entry.run_id, "bytes": entry.size_bytes})
            continue
        started = time.time()
        try:
            stats = export_head_only(run_dir) if entry.action == "export-and-delete" else {}
        except Exception as error:  # keep the run intact when anything is unclear
            review_needed = True
            print(f"ERROR  {entry.run_id}: {error} (latest.pt kept)")
            log(audit, {"action": "error", "run_id": entry.run_id, "error": str(error)})
            continue
        (run_dir / "latest.pt").unlink()
        for tmp in run_dir.glob("latest.pt.tmp*"):
            tmp.unlink()
        log(
            audit,
            {
                "action": "export-and-delete",
                "run_id": entry.run_id,
                "deleted_bytes": entry.size_bytes,
                **stats,
                "seconds": round(time.time() - started, 1),
            },
        )
        print(
            f"done   {entry.run_id}: head {stats.get('head_params', 0):,} params, "
            f"{stats.get('head_bytes', 0) / 1e6:.0f} MB kept; latest.pt removed"
        )
    return 2 if review_needed else 0


if __name__ == "__main__":
    raise SystemExit(main())
