#!/usr/bin/env python3
"""Generate prediction visualisations for a completed run, read-only.

Runs in the worker environment because it loads the checkpoint:

    uv run --project workers/lim python scripts/visualize_run.py \
        --run-id lim_mit_huge-T10-rotated_frozen-train-17-v7 --per-bucket 2 --mode reevaluate

Writes ``<run>/visualization/{manifest.json,samples/*.png}`` and never touches
``run.json`` or the metrics. Design: docs/design/Portal可视化_2026-11-02.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from rs_backbone_eval.visualization import (
    BUCKETS,
    LEGEND,
    MAX_SAMPLES,
    SCHEMA_VERSION,
    compute_fingerprint,
    detection_bucket,
    replay_mode,
    select_bucket_samples,
    take_pool,
    validate_manifest,
    window_digest,
)

REPO = Path(__file__).resolve().parents[1]

#: Inference inputs that a replay must hold fixed. Every one of these is hashed
#: at the run commit and in the working tree; a difference means "re-evaluated",
#: never "original protocol". Config files are included on purpose: a YAML edit
#: changes preprocessing just as much as a Python edit changes decoding.
INFERENCE_FILES = (
    "scripts/visualize_run.py",
    "src/rs_backbone_eval/visualization.py",
    "src/rs_backbone_eval/detection/detector.py",
    "src/rs_backbone_eval/detection/nms.py",
    "src/rs_backbone_eval/detection/ap.py",
    "src/rs_backbone_eval/detection/geometry.py",
    "src/rs_backbone_eval/detection/classes.py",
    "src/rs_backbone_eval/data/isaid_io.py",
    "src/rs_backbone_eval/data/tiling.py",
    "workers/lim/src/lim_worker/replay.py",
    "workers/lim/src/lim_worker/t10.py",
    "workers/lim/src/lim_worker/mit.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> bytes | None:
    result = subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False)
    return result.stdout if result.returncode == 0 else None


def _blob_at(repo: Path, commit: str, relative: str) -> bytes | None:
    return _git(repo, "cat-file", "blob", f"{commit}:{relative}")


def _relative(repo: Path, path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo))
    except ValueError:
        return str(path)


def _worktree_clean(repo: Path) -> bool:
    status = _git(repo, "status", "--porcelain")
    return status is not None and not status.strip()


def _file_hashes(repo: Path, relative: list[str], commit: str | None) -> tuple[dict, dict]:
    """Hashes now and at the run commit; a missing file yields ``None`` there."""
    now: dict[str, str] = {}
    then: dict[str, str | None] = {}
    for name in relative:
        path = repo / name
        now[name] = _sha256(path) if path.is_file() else "<missing>"
        if commit:
            blob = _blob_at(repo, commit, name)
            then[name] = hashlib.sha256(blob).hexdigest() if blob is not None else None
        else:
            then[name] = None
    return now, then


def _environment() -> dict:
    import torch

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "cpu",
        "cudnn": str(torch.backends.cudnn.version()),
        "numpy": np.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def _scene_rows(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    return groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("/geodata/rs-eval/runs"))
    parser.add_argument("--per-bucket", type=int, default=2)
    parser.add_argument("--pool", type=int, default=40)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split", default=None)
    parser.add_argument("--mode", choices=("original", "reevaluate"), default="original")
    parser.add_argument(
        "--display-score",
        type=float,
        default=0.3,
        help="draw boxes at or above this score (the card keeps the protocol threshold)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    run_dir = args.runs_root / args.run_id
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    if record["status"] != "completed":
        print(f"{args.run_id}: status {record['status']}, refusing to visualise", file=sys.stderr)
        return 2
    if record["task_id"] != "T10":
        print("only T10 is implemented in P0", file=sys.stderr)
        return 2

    output = run_dir / "visualization"
    if output.exists() and not args.force:
        print(f"{output} exists; pass --force to replace it", file=sys.stderr)
        return 2

    split = args.split or "test"
    rows = [
        json.loads(line)
        for line in Path(request["manifest_path"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row["split"] == split]
    if not rows:
        print(f"manifest has no {split} rows", file=sys.stderr)
        return 2

    from lim_worker import replay  # noqa: PLC0415 - worker env only

    context = replay.load_for_eval(run_dir, device=args.device)

    # ------------------------------------------------------------- identity
    config_paths = {
        "task": request["task_config_path"],
        "model": request["model_config_path"],
        "protocol": request["protocol_config_path"],
    }
    relative_files = list(INFERENCE_FILES) + [
        _relative(REPO, path) for path in config_paths.values()
    ]
    now_hashes, then_hashes = _file_hashes(REPO, relative_files, context.run_commit)
    params = context.params()
    env = _environment()
    fingerprint_now = compute_fingerprint(now_hashes, params, env)
    fingerprint_at_run = (
        compute_fingerprint({k: v for k, v in then_hashes.items() if v}, params, env)
        if all(then_hashes.get(name) for name in INFERENCE_FILES)
        else None
    )
    from rs_backbone_eval.audit import excluded_run_ids  # noqa: PLC0415
    from rs_backbone_eval.results import RunRepository  # noqa: PLC0415

    records, _ = RunRepository(args.runs_root).list_records()
    excluded = excluded_run_ids(records, REPO)
    stale_flags = [args.run_id] if args.run_id in excluded else []
    mode, reason = replay_mode(
        fingerprint_at_run=fingerprint_at_run,
        fingerprint_now=fingerprint_now,
        worktree_clean=_worktree_clean(REPO),
        stale_flags=stale_flags,
    )
    if mode != "original_protocol" and args.mode == "original":
        print(
            f"not an original-protocol replay: {reason}\n"
            "Re-run with --mode reevaluate if a re-evaluation under the current code is intended.",
            file=sys.stderr,
        )
        return 3

    # -------------------------------------------------------------- sampling
    pool_rows, candidates_sha256 = take_pool(rows, args.seed, args.pool)
    groups = _scene_rows(pool_rows)
    annotations = replay.AnnotationCache(context.data_root, context.keep_ratio)
    scored: list[dict] = []
    matches: dict[str, replay.SceneMatch] = {}
    started = time.time()
    inference_windows = 0
    for scene_id in sorted(groups):
        scene_rows = groups[scene_id]
        match = replay.match_scene(
            replay.predict_scene(context, scene_rows, annotations),
            replay.scene_targets(context, scene_rows[0], annotations),
        )
        matches[scene_id] = match
        inference_windows += len(scene_rows)
        counts = match.counts
        bucket = detection_bucket(
            tp=counts["tp"],
            fp=counts["fp"],
            fn=counts["fn"],
            gt=counts["gt"],
            predictions=counts["pred"],
        )
        # One window per scene: two windows of the same scene would repeat the
        # same scene counts and the same focused object.
        row = min(
            scene_rows,
            key=lambda item: window_digest(
                args.seed,
                scene_id,
                tuple(int(value) for value in item["window"]),
            ),
        )
        x, y, width, height = (int(value) for value in row["window"])
        scored.append(
            {
                "row": row,
                "scene_id": scene_id,
                "digest": window_digest(args.seed, scene_id, (x, y, width, height)),
                "bucket": bucket,
            }
        )
    wall = time.time() - started

    chosen = select_bucket_samples(scored, args.per_bucket)
    if len(chosen) > MAX_SAMPLES:
        raise SystemExit("selection exceeds the sample cap")

    # --------------------------------------------------------------- render
    staging = Path(tempfile.mkdtemp(prefix=f".visualization-{os.getpid()}-", dir=run_dir))
    samples_dir = staging / "samples"
    samples_dir.mkdir(parents=True)
    entries = []
    for item in chosen:
        row = item["row"]
        scene_id = item["scene_id"]
        match = matches[scene_id]
        x, y, width, height = (int(value) for value in row["window"])
        sample_id = f"{scene_id}@{x}_{y}_{width}_{height}"
        image_path = samples_dir / f"{sample_id}.png"
        focus = replay.choose_focus(
            match,
            (x, y, width, height),
            item["digest"],
            display_score=args.display_score,
        )
        payload = annotations.get(row["annotation_path"])
        valid_w = min(context.crop, int(payload["width"]) - x)
        valid_h = min(context.crop, int(payload["height"]) - y)
        outside = replay.render_window(
            context,
            row,
            match,
            image_path,
            display_score=args.display_score,
            focus=focus,
            valid_size=(valid_w, valid_h),
        )
        classes = sorted(
            {replay.FAIR1M_ID_TO_CLASS.get(label, str(label)) for _, label in match.targets}
            | {
                replay.FAIR1M_ID_TO_CLASS.get(detection.class_id, str(detection.class_id))
                for detection in match.detections
            }
        )
        entries.append(
            {
                "sample_id": sample_id,
                "scene_id": scene_id,
                "window": [x, y, width, height],
                "bucket": item["bucket"],
                "diagnostic": False,
                "image": f"samples/{sample_id}.png",
                "metrics": {"status": "ok"},
                "objects": {
                    "tp": match.counts["tp"],
                    "fp": match.counts["fp"],
                    "fn": match.counts["fn"],
                },
                "gt_objects": match.counts["gt"],
                "pred_objects": match.counts["pred"],
                "outside_window": outside,
                "focus": focus,
                "classes_present": classes,
            }
        )

    bucket_counts = {
        bucket: sum(1 for item in scored if item["bucket"] == bucket) for bucket in BUCKETS
    }
    selection = {
        "seed": args.seed,
        "display_score_threshold": args.display_score,
        "pool_windows": len(pool_rows),
        "pool_scenes": len(groups),
        "bucket_counts": bucket_counts,
        "per_bucket": args.per_bucket,
        "bucket_metric": "detection_counts",
        "algorithm": "sha256(seed:group_id:window) asc",
        "candidates_sha256": candidates_sha256,
        "buckets": [bucket for bucket in BUCKETS if any(e["bucket"] == bucket for e in entries)],
        "inference_windows": inference_windows,
        "inference_scenes": len(groups),
        "wall_seconds": round(wall, 1),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": record["run_id"],
        "task_id": record["task_id"],
        "model_id": record["model_id"],
        "protocol_id": record["protocol_id"],
        "run_seed": int(record["seed"]),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "unit": "scene-window",
        "split": split,
        "class_space": {
            "num_classes": context.num_classes,
            "background_class": None,
            "ignore_values": [],
            "source": f"{config_paths['task']} (no background) + min_box_side filter",
        },
        "selection": selection,
        "legend": dict(LEGEND),
        "identity": {
            "run": {
                "run_id": record["run_id"],
                "git_commit": context.run_commit,
                "dirty": record["git_commit"].endswith("-dirty"),
            },
            "data": {
                "manifest_path": str(context.manifest_path),
                "manifest_sha256": record["dataset"]["manifest_sha256"],
                "split": split,
                "windows": len(rows),
                "scenes": len({row["group_id"] for row in rows}),
            },
            "config": {
                key: {"path": _relative(REPO, path), "sha256": _sha256(Path(path))}
                for key, path in config_paths.items()
            }
            | {"request_json_sha256": _sha256(run_dir / "request.json")},
            "backbone": {
                "sha256": record.get("model_checkpoint_sha256"),
                "path": None,
            },
            "head": {
                "path": context.checkpoint_path.name,
                "kind": context.checkpoint_kind,
                "sha256": context.checkpoint_sha256,
            },
            "input": {
                "sensor": record["input"]["sensor"],
                "bands": record["input"]["bands"],
                "preprocessing_id": record["input"]["preprocessing_id"],
                "time_steps": record["input"]["time_steps"],
            },
            "inference": {
                "generator_version": "visualize-run/1",
                "files": now_hashes,
                "files_sha256": hashlib.sha256(
                    json.dumps(now_hashes, sort_keys=True).encode("utf-8")
                ).hexdigest(),
                "params": params,
                "env": env,
            },
            "replay": {
                "mode": mode,
                "reason": reason,
                "run_git_commit": context.run_commit,
                "generator_git_commit": (_git(REPO, "rev-parse", "HEAD") or b"").decode().strip(),
                "fingerprint": fingerprint_now,
                "fingerprint_at_run": fingerprint_at_run,
                "fingerprint_now": fingerprint_now,
                "worktree_clean": _worktree_clean(REPO),
                "stale_flags": stale_flags,
            },
        },
        "samples": entries,
    }
    validate_manifest(manifest)
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if output.exists():
        shutil.rmtree(output)
    staging.replace(output)
    print(
        f"{record['run_id']}: {len(entries)} samples ({', '.join(selection['buckets'])}), "
        f"{selection['inference_scenes']} scenes / {inference_windows} windows in {wall:.1f}s, "
        f"buckets={bucket_counts}, "
        f"mode={mode} -> {output}"
    )
    if reason:
        print(f"  reason: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
