"""Deterministic sampling, bucketing and manifest validation for run visualisation.

This module is deliberately torch-free: the Portal, the tests and the report can
import it without a GPU. The worker adapter (``lim_worker.replay``) does the
inference; ``scripts/visualize_run.py`` glues the two together.

Design: ``docs/design/Portal可视化_2026-11-02.md``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SCHEMA_VERSION = "1.0"

BUCKETS: tuple[str, ...] = (
    "hit",
    "partial",
    "miss",
    "empty_correct",
    "empty_false_alarm",
)
# Not sampled: a window whose ground truth is entirely ignore carries no signal.
IGNORED_BUCKET = "ignored"
ALL_BUCKETS = BUCKETS + (IGNORED_BUCKET,)

LEGEND = {
    "tp": "#3ddc97",
    "fp": "#ff6b6b",
    "fn": "#ffb454",
    "ignore": "#777777",
    "switch": "#7ab8ff",
}

#: Per-bucket samples are capped so an accidental ``--per-bucket 9`` cannot
#: write hundreds of images; over the cap is an error, never a silent truncation.
MAX_SAMPLES = 12


def window_digest(seed: int, scene_id: str, window: Sequence[int]) -> str:
    """Stable ordering key.

    Python's built-in ``hash`` is randomised per process, so the same seed would
    pick different samples on a later run. SHA-256 over an explicit, UTF-8
    serialisation is stable across processes and hosts.
    """
    x, y, width, height = (int(value) for value in window)
    payload = f"{int(seed)}:{scene_id}:{x}:{y}:{width}:{height}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def scene_id_of(row: Mapping[str, Any]) -> str:
    return str(row["group_id"])


def window_of(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    x, y, width, height = (int(value) for value in row["window"])
    return x, y, width, height


def sample_id_of(row: Mapping[str, Any]) -> str:
    x, y, width, height = window_of(row)
    return f"{scene_id_of(row)}@{x}_{y}_{width}_{height}"


def order_candidates(
    rows: Iterable[Mapping[str, Any]], seed: int
) -> list[tuple[str, Mapping[str, Any], str]]:
    """``[(digest, row, sample_id)]`` ordered by the stable digest."""
    keyed = []
    for row in rows:
        digest = window_digest(seed, scene_id_of(row), window_of(row))
        keyed.append((digest, row, sample_id_of(row)))
    keyed.sort(key=lambda item: item[0])
    return keyed


def take_pool(
    rows: Iterable[Mapping[str, Any]], seed: int, pool: int
) -> tuple[list[Mapping[str, Any]], str]:
    """First ``pool`` windows in digest order plus a hash of the whole candidate list."""
    ordered = order_candidates(rows, seed)
    candidate_hash = hashlib.sha256(
        "\n".join(sample_id for _, _, sample_id in ordered).encode("utf-8")
    ).hexdigest()
    return [row for _, row, _ in ordered[:pool]], candidate_hash


def detection_bucket(*, tp: int, fp: int, fn: int, gt: int, predictions: int) -> str:
    """Bucket for one scene/window under the detection contract (design §5.4)."""
    if gt == 0:
        return "empty_correct" if predictions == 0 else "empty_false_alarm"
    if tp == 0:
        return "miss"
    if fp == 0 and fn == 0:
        return "hit"
    return "partial"


def segmentation_bucket(
    *,
    iou: float | None,
    gt_foreground: int,
    predicted_foreground: int,
    has_background: bool,
) -> str:
    """Bucket for one scene/window under the segmentation contract (design §5.2)."""
    if iou is None:
        return IGNORED_BUCKET
    if has_background and gt_foreground == 0:
        return "empty_correct" if predicted_foreground == 0 else "empty_false_alarm"
    if iou >= 0.7:
        return "hit"
    if iou >= 0.3:
        return "partial"
    return "miss"


def present_buckets(scored: Iterable[Mapping[str, Any]]) -> list[str]:
    seen = {item["bucket"] for item in scored}
    return [bucket for bucket in BUCKETS if bucket in seen]


def select_bucket_samples(
    scored: Sequence[Mapping[str, Any]],
    per_bucket: int,
    *,
    limit: int = MAX_SAMPLES,
) -> list[Mapping[str, Any]]:
    """Pick ``per_bucket`` items from each non-empty bucket, digest order within it.

    Raises when the request would exceed ``limit``: the caller must lower
    ``--per-bucket`` rather than have the tool silently write a subset.
    """
    if per_bucket < 1:
        raise ValueError("per_bucket must be at least 1")
    buckets = present_buckets(scored)
    wanted = per_bucket * len(buckets)
    if wanted > limit:
        raise ValueError(
            f"{len(buckets)} non-empty buckets x {per_bucket} = {wanted} images exceeds "
            f"the hard cap of {limit}; lower --per-bucket"
        )
    chosen: list[Mapping[str, Any]] = []
    for bucket in buckets:
        items = sorted(
            (item for item in scored if item["bucket"] == bucket),
            key=lambda item: item["digest"],
        )
        chosen.extend(items[:per_bucket])
    return chosen


# --------------------------------------------------------------------- identity


def compute_fingerprint(
    files: Mapping[str, str], params: Mapping[str, Any], env: Mapping[str, Any]
) -> str:
    """Hash of the inference inputs: file contents, parameters and numeric environment."""
    payload = {
        "schema": 1,
        "files": dict(sorted(files.items())),
        "params": dict(params),
        "env": dict(sorted(env.items())),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def replay_mode(
    *,
    fingerprint_at_run: str | None,
    fingerprint_now: str,
    worktree_clean: bool,
    stale_flags: Sequence[str],
) -> tuple[str, str | None]:
    """``(mode, reason)`` under the tightened rule in design §7.

    A missing fingerprint, a dirty tree, an audit flag, or any difference means
    the images are a re-evaluation, not a replay of the original protocol.
    """
    if stale_flags:
        return "reevaluated", f"audit flag(s): {', '.join(stale_flags)}"
    if not worktree_clean:
        return "reevaluated", "working tree is dirty"
    if fingerprint_at_run is None:
        return "reevaluated", "inference files do not exist at the run commit"
    if fingerprint_at_run != fingerprint_now:
        return "reevaluated", "inference fingerprint differs from the run commit"
    return "original_protocol", None


# ------------------------------------------------------------------- validation

_REQUIRED_TOP = (
    "schema_version",
    "run_id",
    "task_id",
    "model_id",
    "protocol_id",
    "run_seed",
    "generated_at",
    "unit",
    "split",
    "class_space",
    "selection",
    "legend",
    "samples",
)
_REQUIRED_SELECTION = (
    "seed",
    "pool_windows",
    "pool_scenes",
    "per_bucket",
    "bucket_metric",
    "algorithm",
    "candidates_sha256",
    "buckets",
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_manifest(payload: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` on a manifest that the Portal could not render honestly.

    Used by the generator before it writes, by the tests, and implicitly by the
    documented example: the design's sample is the test input, so the docs cannot
    drift from the schema.
    """
    problems: list[str] = []
    for key in _REQUIRED_TOP:
        if key not in payload:
            problems.append(f"missing {key}")
    if problems:
        raise ValueError("; ".join(problems))

    if payload["schema_version"] != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    if payload["unit"] not in {"scene-window", "tile"}:
        problems.append("unit must be scene-window or tile")
    if not _is_int(payload["run_seed"]):
        problems.append("run_seed must be an integer")

    class_space = payload["class_space"]
    for key in ("num_classes", "background_class", "ignore_values", "source"):
        if key not in class_space:
            problems.append(f"class_space missing {key}")
    if not _is_int(class_space.get("num_classes")):
        problems.append("class_space.num_classes must be an integer")

    selection = payload["selection"]
    for key in _REQUIRED_SELECTION:
        if key not in selection:
            problems.append(f"selection missing {key}")
    buckets = selection.get("buckets") or []
    unknown = sorted(set(buckets) - set(BUCKETS))
    if unknown:
        problems.append(f"unknown buckets {unknown}")
    if (
        selection.get("per_bucket")
        and buckets
        and selection["per_bucket"] * len(buckets) > MAX_SAMPLES
    ):
        problems.append("selection exceeds the per-run sample cap")

    samples = payload["samples"]
    if not samples:
        problems.append("samples is empty")
    for index, sample in enumerate(samples):
        where = f"samples[{index}]"
        for key in ("sample_id", "scene_id", "window", "bucket", "image", "metrics"):
            if key not in sample:
                problems.append(f"{where} missing {key}")
                continue
        if problems:
            continue
        window = sample["window"]
        if not (isinstance(window, list) and len(window) == 4 and all(_is_int(v) for v in window)):
            problems.append(f"{where}.window must be four integers")
        bucket = sample["bucket"]
        if bucket not in ALL_BUCKETS:
            problems.append(f"{where}.bucket {bucket!r} is not a bucket")
        if bucket not in buckets:
            problems.append(f"{where}.bucket {bucket!r} is not in selection.buckets")
        image = str(sample["image"])
        if not image.startswith("samples/") or ".." in image.split("/"):
            problems.append(f"{where}.image must stay under samples/")
        focus = sample.get("focus")
        if focus is not None:
            if focus.get("kind") not in {"tp", "fp", "fn"}:
                problems.append(f"{where}.focus.kind must be tp, fp or fn")
            if not isinstance(focus.get("class"), str):
                problems.append(f"{where}.focus.class must be a string")
        problems.extend(_validate_counts(where, sample, selection.get("bucket_metric")))
    if problems:
        raise ValueError("; ".join(problems))


def _validate_counts(where: str, sample: Mapping[str, Any], bucket_metric: str | None) -> list[str]:
    problems: list[str] = []
    metrics = sample.get("metrics") or {}
    if "objects" in sample:
        counts = sample["objects"]
        for key in ("tp", "fp", "fn"):
            if key not in counts:
                problems.append(f"{where}.objects missing {key}")
        if problems:
            return problems
        tp, fp, fn = (int(counts[key]) for key in ("tp", "fp", "fn"))
        expected = detection_bucket(
            tp=tp,
            fp=fp,
            fn=fn,
            gt=int(sample.get("gt_objects", tp + fn)),
            predictions=int(sample.get("pred_objects", tp + fp)),
        )
        if expected != sample["bucket"]:
            problems.append(f"{where}.bucket {sample['bucket']!r} contradicts counts ({expected})")
        if metrics.get("status") not in (None, "ok"):
            problems.append(f"{where}.metrics.status must be ok for detection")
        return problems

    counts = sample.get("pixels")
    if not isinstance(counts, Mapping):
        problems.append(f"{where} needs objects (detection) or pixels (segmentation)")
        return problems
    for key in ("tp", "fp", "fn"):
        if key not in counts:
            problems.append(f"{where}.pixels missing {key}")
    if problems:
        return problems

    status = metrics.get("status")
    if status == "ignored":
        if sample["bucket"] != IGNORED_BUCKET:
            problems.append(f"{where}: ignored metrics must use the ignored bucket")
        return problems
    if sample["bucket"] == IGNORED_BUCKET:
        problems.append(f"{where}: ignored bucket requires metrics.status == 'ignored'")
        return problems

    metric = bucket_metric or "foreground_binary_iou"
    iou = metrics.get(metric)
    if iou is None:
        problems.append(f"{where}.metrics.{metric} is missing")
        return problems
    iou = float(iou)
    if sample["bucket"] == "hit" and iou < 0.7:
        problems.append(f"{where}: hit needs {metric} >= 0.7, got {iou}")
    if sample["bucket"] == "partial" and not (0.3 <= iou < 0.7):
        problems.append(f"{where}: partial needs 0.3 <= {metric} < 0.7, got {iou}")
    if sample["bucket"] == "miss" and iou >= 0.3:
        problems.append(f"{where}: miss needs {metric} < 0.3, got {iou}")
    if sample["bucket"] in {"empty_correct", "empty_false_alarm"}:
        foreground = int(sample.get("gt_foreground_pixels", counts["tp"] + counts["fn"]))
        if foreground != 0:
            problems.append(f"{where}: empty bucket requires gt_foreground_pixels == 0")
        if sample["bucket"] == "empty_correct" and (counts["tp"] + counts["fp"]) != 0:
            problems.append(f"{where}: empty_correct requires no predicted foreground")
        if sample["bucket"] == "empty_false_alarm" and (counts["tp"] + counts["fp"]) == 0:
            problems.append(f"{where}: empty_false_alarm requires predicted foreground")
    return problems
