#!/usr/bin/env python
"""Recompute stored segmentation metrics from a run's confusion-matrix artifact.

Background (review 2026-10-27)
------------------------------
Multi-class metrics used to emit per-class IoU for classes that appear in
neither the ground truth nor the prediction (0/0). That NaN was written to
``run.json`` as ``null`` by the JSON serializer, so the whole :class:`RunRecord`
failed validation on read -- a run that trained fine became unreadable. The
same NaN also poisoned ``frequency_weighted_iou`` through ``0 * NaN``.

The per-group confusion counts are kept as an artifact
(``confusion_per_scene.json``), so the metrics can be recomputed **without
retraining**. This script does exactly that, as a sanctioned code path instead
of hand-editing a ledger file:

* it never touches ``status``, ``training``, ``dataset``, ``git_commit`` or the
  checkpoint hash -- only the ``metrics`` list and one provenance tag;
* it refuses a run when the recomputed primary metric does not reproduce the
  stored value (that would mean the artifact and the record disagree);
* it writes ``run.json.pre-rederive-<date>`` next to the file before replacing
  it, so the previous state stays auditable.

Usage
-----
    python scripts/rederive_metrics.py /geodata/rs-eval/runs --dry-run
    python scripts/rederive_metrics.py /geodata/rs-eval/runs T8-run-id-1 ...
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

from rs_backbone_eval.contracts import RunRecord
from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
from rs_backbone_eval.evaluation.segmentation import binary_metrics, multiclass_metrics

# Metric names that are derived from the confusion matrix. Anything else in a
# record is left untouched, so rederiving can never invent or drop a metric
# that came from a different source.
CONFUSION_METRICS = {
    "foreground_iou",
    "foreground_f1",
    "precision",
    "recall",
    "mean_iou",
    "macro_f1",
    "frequency_weighted_iou",
    "iou",
}

PRIMARY_TOLERANCE = 1e-6


class RefusedError(RuntimeError):
    """The artifact and the stored record disagree; refuse to rewrite."""


def load_counts(path: Path) -> np.ndarray:
    """Sum the per-group count matrices stored in a confusion artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("confusion artifact has no groups")
    num_classes = int(payload["num_classes"])
    total = np.zeros((num_classes, num_classes), dtype=np.int64)
    for group in groups.values():
        counts = np.asarray(group, dtype=np.int64)
        if counts.shape != (num_classes, num_classes):
            raise ValueError(f"group matrix shape {counts.shape} != {(num_classes, num_classes)}")
        total += counts
    return total


def recompute(counts: np.ndarray) -> dict[str, float]:
    cm = ConfusionMatrix.from_counts(counts)
    if cm.num_classes == 2:
        return binary_metrics(cm)
    return multiclass_metrics(cm)


def _key(metric: dict) -> str:
    class_name = metric.get("class_name")
    return metric["name"] if class_name is None else f"{metric['name']}/{class_name}"


def rederive_payload(payload: dict, run_dir: Path) -> tuple[dict, list[str]]:
    """Return a new run payload and a human-readable change list."""
    metrics = payload.get("metrics") or []
    splits = {metric.get("split") for metric in metrics}
    if len(splits) != 1:
        raise ValueError(f"metrics span several splits {sorted(splits)}; rederive is ambiguous")

    artifacts = payload.get("artifacts") or []
    confusion = next((item for item in artifacts if item.get("kind") == "confusion_matrix"), None)
    if confusion is None:
        raise ValueError("no confusion_matrix artifact")

    counts = load_counts(run_dir / confusion["path"])
    computed = recompute(counts)

    primaries = [metric for metric in metrics if metric.get("is_primary")]
    if len(primaries) != 1:
        raise ValueError(f"expected exactly one primary metric, found {len(primaries)}")
    primary = primaries[0]
    primary_key = _key(primary)
    expected = computed.get(primary_key)
    if expected is None or not math.isfinite(expected):
        raise ValueError(f"recomputed primary {primary_key!r} is undefined")
    stored = primary.get("value")
    if stored is not None and not math.isclose(
        float(stored), expected, rel_tol=0.0, abs_tol=PRIMARY_TOLERANCE
    ):
        raise RefusedError(f"stored primary {stored} != recomputed {expected}")

    changes: list[str] = []
    rebuilt: list[dict] = []
    for metric in metrics:
        if metric["name"] not in CONFUSION_METRICS:
            rebuilt.append(metric)
            continue
        key = _key(metric)
        value = computed.get(key)
        if value is None or not math.isfinite(value):
            changes.append(f"drop   {key}: undefined (was {metric.get('value')})")
            continue
        if metric.get("value") != value:
            changes.append(f"fix    {key}: {metric.get('value')} -> {value:.6f}")
            metric = {**metric, "value": value}
        rebuilt.append(metric)

    for key, value in computed.items():
        if not math.isfinite(value):
            continue
        if any(_key(metric) == key for metric in rebuilt):
            continue
        template = dict(primary)
        name, _, class_name = key.partition("/")
        rebuilt.append(
            {
                **template,
                "name": name,
                "class_name": class_name or None,
                "value": value,
                "is_primary": key == primary_key,
            }
        )
        changes.append(f"add    {key}: {value:.6f}")

    new_payload = {**payload, "metrics": rebuilt}
    tags = dict(new_payload.get("tags") or {})
    tags["metrics_rederived"] = f"{date.today().isoformat()}; source=confusion_per_scene.json"
    new_payload["tags"] = tags
    return new_payload, changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs_root", type=Path)
    parser.add_argument(
        "run_ids", nargs="*", help="default: every run whose metrics contain a null value"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--all", action="store_true", help="also rederive records that have no null metric"
    )
    args = parser.parse_args(argv)

    if not args.runs_root.is_dir():
        print(f"runs root not found: {args.runs_root}", file=sys.stderr)
        return 2

    candidates = (
        [args.runs_root / run_id for run_id in args.run_ids]
        if args.run_ids
        else sorted(path for path in args.runs_root.iterdir() if path.is_dir())
    )

    changed = skipped = failed = 0
    for run_dir in candidates:
        record_path = run_dir / "run.json"
        if not record_path.is_file():
            continue
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        has_null = any(metric.get("value") is None for metric in payload.get("metrics") or [])
        if not has_null and not args.all:
            continue
        try:
            new_payload, changes = rederive_payload(payload, run_dir)
        except RefusedError as error:
            print(f"REFUSE {run_dir.name}: {error}")
            failed += 1
            continue
        except (ValueError, KeyError, FileNotFoundError) as error:
            print(f"SKIP   {run_dir.name}: {error}")
            skipped += 1
            continue

        if not changes:
            print(f"OK     {run_dir.name}: already consistent")
            continue

        try:
            RunRecord.model_validate(new_payload)
        except Exception as error:  # noqa: BLE001 - report and keep going
            print(f"FAIL   {run_dir.name}: schema still rejects the record: {error}")
            failed += 1
            continue

        for line in changes:
            print(f"  {run_dir.name}: {line}")
        changed += 1
        if args.dry_run:
            continue

        backup = run_dir / f"run.json.pre-rederive-{date.today().isoformat()}"
        if not backup.exists():
            backup.write_bytes(record_path.read_bytes())
        temporary = record_path.with_suffix(".json.tmp")
        temporary.write_text(
            RunRecord.model_validate(new_payload).model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(record_path)

    mode = "dry-run" if args.dry_run else "written"
    print(f"{mode}: {changed} changed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
