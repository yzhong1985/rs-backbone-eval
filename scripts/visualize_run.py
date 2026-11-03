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
import fcntl
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
    candidate_window_per_scene,
    compute_fingerprint,
    detection_bucket,
    identity_drift,
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
#: The CLI driver (scripts/visualize_run.py itself) is deliberately excluded:
#: rendering/presentation changes must not flip historical runs out of
#: original-protocol replay (review 2026-11-03, R1-3). Files that carry
#: inference numerics (replay, detection) stay here, from-now-on strict.
INFERENCE_FILES = (
    "src/rs_backbone_eval/visualization.py",
    "src/rs_backbone_eval/detection/detector.py",
    "src/rs_backbone_eval/detection/nms.py",
    "src/rs_backbone_eval/detection/ap.py",
    "src/rs_backbone_eval/detection/geometry.py",
    "src/rs_backbone_eval/detection/classes.py",
    # Anchor sizes/ratios and the decode weights: changing them changes the
    # boxes without touching detector.py, so they belong to the fingerprint
    # (review 2026-11-03).
    "src/rs_backbone_eval/detection/anchors.py",
    # FPN structure and the GroupNorm factory: both change the predictions
    # without touching detector.py (review 2026-11-03, third round).
    "src/rs_backbone_eval/detection/fpn.py",
    "src/rs_backbone_eval/heads.py",
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
    # -uno: untracked reports/tmp must not read as "dirty" — the gate protects
    # inference code, not the presence of output files (review 2026-11-03).
    status = _git(repo, "status", "--porcelain", "-uno")
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
        "torchvision": __import__("torchvision").__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


#: Same idle rule as ``scripts/rerun_baselines.sh``: a GPU using more than this
#: is running something (a formal lane), and a replay must not share it.
GPU_IDLE_MIB = 2000
GPU_POLL_SECONDS = 300


def _check_arguments(args: argparse.Namespace) -> str | None:
    """Reject a bad request before anything is loaded."""
    if args.pool < 1:
        return "--pool must be at least 1"
    if not 1 <= args.per_bucket <= MAX_SAMPLES:
        return f"--per-bucket must be between 1 and {MAX_SAMPLES}"
    if not 0.0 <= args.display_score <= 1.0:
        return "--display-score must be between 0 and 1"
    if args.device not in (None, "cpu", "cuda") and not str(args.device).startswith("cuda:"):
        return "--device must be cpu, cuda or cuda:N"
    return None


def _acquire_lock(runs_root: Path):
    """Process-wide generator lock inside the runs root; ``None`` when held.

    Placed inside ``runs_root`` (gitignored) rather than its parent: at the
    repo root it would be an untracked file and poison the clean-tree check
    (review 2026-11-03, R1-2).
    """
    path = Path(runs_root) / ".visualize_run.lock"
    handle = path.open("a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def _is_cuda_device(device: str) -> bool:
    """``cuda`` and ``cuda:N`` both need the idle check; ``cuda:0`` must not bypass it."""
    return str(device).split(":")[0] == "cuda"


def _gpu_used_mib(device: str) -> int | None:
    """Memory used in MiB on *the requested device only*, or None on failure."""
    index = str(device).split(":")[1] if ":" in str(device) else "0"
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", index, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if result.returncode != 0 or not result.stdout.strip().isdigit():
        return None
    return int(result.stdout.strip())


def _wait_for_idle_gpu(wait: bool, device: str) -> tuple[bool, str | None]:
    """Wait until the requested device is idle; ``(False, reason)`` on refuse.

    The refusal reason distinguishes an absent nvidia-smi from a genuinely busy
    GPU so the caller can print the right advice (review 2026-11-03).
    """
    while True:
        used = _gpu_used_mib(device)
        if used is None:
            return False, "GPU idle check unavailable (nvidia-smi missing or failed)"
        if used <= GPU_IDLE_MIB:
            return True, None
        if not wait:
            return False, f"GPU {device or 'cuda'} busy ({used} MiB)"
        print(f"GPU busy ({used} MiB); waiting {GPU_POLL_SECONDS}s", file=sys.stderr)
        time.sleep(GPU_POLL_SECONDS)


def _training_identity(run_dir: Path) -> dict | None:
    """The identity training wrote into its own checkpoint (latest or head_only).

    ``best_checkpoint.pt`` carries no identity, so the run's ``latest.pt`` (or the
    ``head_only.pt`` exported from it) is the source even when best weights are
    replayed. Loaded memory-mapped: only the small identity dict is read.
    """
    import torch  # noqa: PLC0415 - worker env only

    for name in ("latest.pt", "head_only.pt"):
        path = run_dir / name
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        identity = payload.get("identity")
        if identity:
            return dict(identity)
    return None


def _scene_rows(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    return groups


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--per-bucket", type=int, default=2)
    parser.add_argument("--pool", type=int, default=120)
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
    parser.add_argument(
        "--wait",
        action="store_true",
        help="wait for the GPU to be idle instead of refusing (same rule as the rerun lanes)",
    )
    args = parser.parse_args(argv)
    problem = _check_arguments(args)
    if problem:
        print(problem, file=sys.stderr)
        return 2
    if args.runs_root is None:
        from rs_backbone_eval.paths import RuntimePaths  # noqa: PLC0415

        args.runs_root = RuntimePaths.from_environment().runs_root

    run_dir = args.runs_root / args.run_id
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    if record["status"] != "completed":
        print(f"{args.run_id}: status {record['status']}, refusing to visualise", file=sys.stderr)
        return 2
    if record["task_id"] != "T10":
        print("only T10 is implemented in P0", file=sys.stderr)
        return 2

    # One generator at a time (design §10). The lock is held until the process
    # exits, so the output check below and the final directory swap cannot race
    # with a second generator writing the same run.
    lock = _acquire_lock(args.runs_root)
    if lock is None:
        print("another visualize_run.py is running (lock held); try again later", file=sys.stderr)
        return 4

    output = run_dir / "visualization"
    if output.exists() and not args.force:
        print(f"{output} exists; pass --force to replace it", file=sys.stderr)
        return 2

    split = args.split or "test"
    manifest_path = Path(request["manifest_path"])
    manifest_sha_now = _sha256(manifest_path)
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row["split"] == split]
    if not rows:
        print(f"manifest has no {split} rows", file=sys.stderr)
        return 2

    import torch  # noqa: PLC0415 - worker env only
    from lim_worker import replay  # noqa: PLC0415 - worker env only

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------- identity
    # Everything here is decided before the model is built, so a request that
    # would be refused never allocates GPU memory.
    config_paths = {
        "task": request["task_config_path"],
        "model": request["model_config_path"],
        "protocol": request["protocol_config_path"],
    }
    relative_files = list(INFERENCE_FILES) + [
        _relative(REPO, path) for path in config_paths.values()
    ]
    run_commit = str(record.get("git_commit") or "")
    now_hashes, then_hashes = _file_hashes(REPO, relative_files, run_commit)
    training_identity = _training_identity(run_dir)
    replay_identity = {
        "task": record["task_id"],
        "model": record["model_id"],
        "protocol": record["protocol_id"],
        "seed": record["seed"],
        "manifest": manifest_sha_now,
        "pretrained_checkpoint": record.get("model_checkpoint_sha256"),
        "git_commit": run_commit,
        "torch_version": str(torch.__version__),
        "device_type": torch.device(device).type,
        "configs": {key: _sha256(Path(path)) for key, path in config_paths.items()},
    }
    drift = identity_drift(training_identity, replay_identity)
    recorded_manifest = record["dataset"]["manifest_sha256"]
    if manifest_sha_now != recorded_manifest:
        drift.insert(0, "data manifest content differs from run.json dataset.manifest_sha256")
    # Environment on each side: the run's from its own checkpoint, the replay's
    # from this process. Never fill the run side with current values.
    env_at_run = (
        {
            "torch_version": training_identity.get("torch_version"),
            "device_type": training_identity.get("device_type"),
        }
        if training_identity
        and training_identity.get("torch_version")
        and training_identity.get("device_type")
        else None
    )
    env_now = {
        "torch_version": replay_identity["torch_version"],
        "device_type": replay_identity["device_type"],
    }
    from rs_backbone_eval.audit import excluded_run_ids  # noqa: PLC0415
    from rs_backbone_eval.results import RunRepository  # noqa: PLC0415

    records, _ = RunRepository(args.runs_root).list_records()
    excluded = excluded_run_ids(records, REPO)
    stale_flags = [args.run_id] if args.run_id in excluded else []
    files_at_run = (
        {k: v for k, v in then_hashes.items() if v}
        if all(then_hashes.get(name) for name in INFERENCE_FILES)
        else None
    )
    # Inference parameters are a function of the hashed files and the run's own
    # tags, so they are not compared separately; the full params go into the
    # fingerprint below once the model is built.
    pre_at_run = (
        compute_fingerprint(files_at_run, {}, env_at_run)
        if files_at_run is not None and env_at_run is not None
        else None
    )
    pre_now = compute_fingerprint(now_hashes, {}, env_now)
    mode, reason = replay_mode(
        fingerprint_at_run=pre_at_run,
        fingerprint_now=pre_now,
        worktree_clean=_worktree_clean(REPO),
        stale_flags=stale_flags,
        drift=drift,
    )
    if mode != "original_protocol" and args.mode == "original":
        print(
            f"not an original-protocol replay: {reason}\n"
            "Re-run with --mode reevaluate if a re-evaluation under the current code is intended.",
            file=sys.stderr,
        )
        return 3

    # ------------------------------------------------------------------ GPU
    if _is_cuda_device(device):
        idle, reason = _wait_for_idle_gpu(args.wait, device)
        if not idle:
            print(
                f"{reason} (idle threshold {GPU_IDLE_MIB} MiB, the rerun lanes' threshold). "
                "Pass --wait to queue, or --device cpu.",
                file=sys.stderr,
            )
            return 5

    context = replay.load_for_eval(run_dir, device=device)
    params = context.params()
    fingerprint_now = compute_fingerprint(now_hashes, params, env_now)
    fingerprint_at_run = (
        compute_fingerprint(files_at_run, params, env_at_run) if pre_at_run is not None else None
    )
    env = _environment()

    # -------------------------------------------------------------- sampling
    pool_rows, candidates_sha256 = take_pool(rows, args.seed, args.pool)
    # The pool sizes the *scene* sample; every window of a chosen scene is then
    # scored. Scoring only the sampled windows left the other windows' ground
    # truth unmatched, so TP/FP/FN (and the bucket) were not the scene protocol
    # (review 2026-11-03: scene 578 used 2 of its 25 windows and was labelled
    # miss with TP=0/FN=1).
    candidates = candidate_window_per_scene(pool_rows, args.seed)
    all_windows = _scene_rows(rows)
    groups = {scene: all_windows[scene] for scene in candidates}
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
        # The displayed window is the candidate window (smallest digest for a
        # scene that appears more than once in the pool); the counts above use
        # every window of the scene.
        row = candidates[scene_id]
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
    replaced = False
    try:
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
            "full_scene_inference": True,
            "scene_windows_total": sum(len(items) for items in groups.values()),
            "bucket_counts": bucket_counts,
            "per_bucket": args.per_bucket,
            "bucket_metric": "detection_counts",
            "algorithm": "sha256(seed:group_id:window) asc",
            "candidates_sha256": candidates_sha256,
            "buckets": [
                bucket for bucket in BUCKETS if any(e["bucket"] == bucket for e in entries)
            ],
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
                    "git_commit": run_commit,
                    "dirty": (record.get("git_commit") or "").endswith("-dirty"),
                },
                "data": {
                    "manifest_path": str(context.manifest_path),
                    "manifest_sha256": manifest_sha_now,
                    "manifest_sha256_recorded": recorded_manifest,
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
                    "generator_git_commit": (_git(REPO, "rev-parse", "HEAD") or b"")
                    .decode()
                    .strip(),
                    "fingerprint": fingerprint_now,
                    "fingerprint_at_run": fingerprint_at_run,
                    "fingerprint_now": fingerprint_now,
                    "worktree_clean": _worktree_clean(REPO),
                    "stale_flags": stale_flags,
                    "drift": drift,
                    "env_at_run": env_at_run,
                    "env_now": env_now,
                    # recorded by this generator but not by training, so never
                    # part of the original-protocol decision
                    "env_unverified": ["cuda", "cudnn", "numpy", "torchvision", "device"],
                },
            },
            "samples": entries,
        }
        validate_manifest(manifest)
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Swap directories with two renames: keep the previous gallery until the
        # new one is in place, and restore it if the swap fails. The old code
        # deleted first, so a failure in between left no gallery at all.
        backup = None
        if output.exists():
            backup = run_dir / f".visualization-old-{os.getpid()}"
            output.rename(backup)
        try:
            staging.replace(output)
            replaced = True
        except BaseException:
            if backup is not None and backup.exists() and not output.exists():
                backup.rename(output)
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    finally:
        if not replaced:
            shutil.rmtree(staging, ignore_errors=True)
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
