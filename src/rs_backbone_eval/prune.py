"""Checkpoint retention policy: export head-only weights, then drop `latest.pt`.

`latest.pt` is the resume state (full state_dict incl. the frozen backbone,
optimizer, RNG). Once a run is `completed` it is no longer needed for resuming,
but it is the only copy of that model's weights — so deleting it would also
remove the ability to *re-evaluate* the run (new holdout, fixed evaluator)
without retraining.

Lightning runs (Prithvi B0) keep the same state in ``checkpoints/last.ckpt`` with
the model in ``best-*.ckpt``; ``plan_last_ckpt`` applies the same policy there
(2026-11-09: 23 completed runs, 88 GB).

Policy (2026-11-01, option 2):
1. export the trainable part (`head_only.pt`, tens of MB) with enough identity to
   rebuild the frozen backbone from its pretrained checkpoint;
2. only then delete `latest.pt`;
3. never touch `run.json`, `best_checkpoint.pt`, logs or any other artifact;
4. dry-run by default, one line per run in `checkpoint_prune.log`.

This module holds the decisions; `scripts/prune_checkpoints.py` does the I/O.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Module paths under which a frozen encoder lives. LimDetector exposes
# `detector.backbone.*`; the Clay/Prithvi/AnySat probes register `encoder.*`; a
# wrapped variant would use `model.backbone.*`/`model.encoder.*`.
BACKBONE_PREFIXES = (
    "detector.backbone.",
    "model.backbone.",
    "model.encoder.",
    "encoder.",
    "backbone.",
)
# A kept subset larger than this is suspicious: it means the backbone was not
# recognised, and exporting it would defeat the purpose.
MAX_HEAD_PARAMS = 80_000_000


@dataclass
class Entry:
    run_id: str
    status: str
    size_bytes: int
    has_head_only: bool
    action: str  # "export-and-delete" | "skip" | "dry-run"
    reason: str = ""


def split_keys(keys: list[str]) -> tuple[list[str], list[str]]:
    """(head keys, backbone keys) by prefix; the head keeps everything else."""
    head, backbone = [], []
    for key in keys:
        if key.startswith(BACKBONE_PREFIXES):
            backbone.append(key)
        else:
            head.append(key)
    return head, backbone


def is_smoke(run_dir: Path) -> bool:
    """Diagnostic short runs stay: their checkpoints are small and useful."""
    try:
        payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    notes = (payload.get("notes") or "").upper()
    return "SMOKE" in notes or "SMOKE" in run_dir.name.upper()


def read_status(run_dir: Path) -> str | None:
    try:
        payload = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return payload.get("status")


def plan(runs_root: Path, older_than_days: float = 0.0) -> list[Entry]:
    """Every ``latest.pt`` under ``runs_root`` with the action it would get."""
    import time

    now = time.time()
    entries: list[Entry] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        checkpoint = run_dir / "latest.pt"
        if not checkpoint.is_file():
            continue
        status = read_status(run_dir) or "unknown"
        age_days = (now - checkpoint.stat().st_mtime) / 86400
        head_only = (run_dir / "head_only.pt").is_file()
        if status != "completed":
            action, reason = "skip", f"status is {status}, may still need to resume"
        elif is_smoke(run_dir):
            action, reason = "skip", "smoke/diagnostic run"
        elif age_days < older_than_days:
            action, reason = "skip", f"completed {age_days:.1f} d ago, younger than the window"
        elif head_only:
            action, reason = "delete", "head_only.pt already exported"
        else:
            action, reason = "export-and-delete", ""
        entries.append(
            Entry(run_dir.name, status, checkpoint.stat().st_size, head_only, action, reason)
        )
    return entries


def summarise(entries: list[Entry]) -> dict:
    reclaimable = sum(e.size_bytes for e in entries if e.action in ("delete", "export-and-delete"))
    return {
        "runs_with_latest": len(entries),
        "export_and_delete": sum(e.action == "export-and-delete" for e in entries),
        "delete": sum(e.action == "delete" for e in entries),
        "skip": sum(e.action == "skip" for e in entries),
        "reclaimable_bytes": reclaimable,
        "reclaimable_gb": round(reclaimable / 1e9, 2),
    }


def plan_last_ckpt(runs_root: Path, older_than_days: float = 0.0) -> list[Entry]:
    """Lightning ``checkpoints/last.ckpt`` of completed runs.

    Prithvi B0 runs save through Lightning: ``last.ckpt`` is the resume state
    and ``best-*.ckpt`` the model artifact. Once a run is ``completed`` the
    resume state has the same status as ``latest.pt``: it is not an experiment
    result and the audit trail does not reference it (checked 2026-11-09: no
    run.json mentions ``last.ckpt``). It is only removed when a best checkpoint
    is present, and never for failed/running/smoke runs.
    """
    import time

    now = time.time()
    entries: list[Entry] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        checkpoints = run_dir / "checkpoints"
        resume = checkpoints / "last.ckpt"
        if not resume.is_file():
            continue
        status = read_status(run_dir) or "unknown"
        age_days = (now - resume.stat().st_mtime) / 86400
        best = sorted(checkpoints.glob("best-*.ckpt"))
        if status != "completed":
            action, reason = "skip", f"status is {status}, may still need to resume"
        elif is_smoke(run_dir):
            action, reason = "skip", "smoke/diagnostic run"
        elif age_days < older_than_days:
            action, reason = "skip", f"completed {age_days:.1f} d ago, younger than the window"
        elif not best:
            action, reason = "skip", "no best-*.ckpt to keep"
        else:
            action, reason = "delete-last-ckpt", ""
        entries.append(Entry(run_dir.name, status, resume.stat().st_size, False, action, reason))
    return entries


def summarise_last_ckpt(entries: list[Entry]) -> dict:
    reclaimable = sum(e.size_bytes for e in entries if e.action == "delete-last-ckpt")
    return {
        "runs_with_last_ckpt": len(entries),
        "delete_last_ckpt": sum(e.action == "delete-last-ckpt" for e in entries),
        "skip": sum(e.action == "skip" for e in entries),
        "reclaimable_bytes": reclaimable,
        "reclaimable_gb": round(reclaimable / 1e9, 2),
    }
