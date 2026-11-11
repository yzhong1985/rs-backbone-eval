#!/usr/bin/env python3
"""P0 acceptance for the RGB+NIR (B02/B03/B04/B08) PASTIS-HD distillation track.

Read-only audit of the four things the handoff asks for before any training:

1. the T11/T11O/T11O128 manifests (sample ids, splits, folds, labels, paths,
   content hashes) and the fact that they are the same 2,433 patches;
2. the raw S2 stack: band order, dtype/DN range, acquisition dates, and the
   four band indices that the student will read;
3. the teachers: AEF 2019 (already dequantised float16) and OlmoEarth v1.3
   2019 (int8, ``sign(v)*(|v|/127.5)^2``), their valid masks, the shared
   valid intersection, and how they aggregate onto Clay's stride-8 token grid;
4. the label rasters (channel 0, classes 0-19, void 19) and the fold split.

No GPU, no torch, no writes into the data tree. The report is JSON; the
summary printed to stdout is what goes into the P0 document.

Usage::

    python scripts/pastis_rgbn_p0.py --out /home/yzhong/rs-eval/audits/pastis-rgbn/p0.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

DATA_ROOT = Path("/geodata/rs-eval/data")
MANIFEST_ROOT = DATA_ROOT / "manifests"
TASKS = ("T11", "T11O", "T11O128")
# PASTIS-HD 10-band order (README + documentation/pastis-documentation.pdf):
# B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12. The 4-band student keeps the
# visible + NIR bands: blue, green, red, nir.
S2_BANDS = ["B02", "B03", "B04", "B08"]
S2_INDICES = [0, 1, 2, 6]
GRID = 128
PATCH = 8
TOKENS = GRID // PATCH
VOID = 19


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def latest_manifest(task: str) -> Path:
    candidates = sorted(
        (MANIFEST_ROOT / task).glob("*.jsonl"), key=lambda path: path.stat().st_mtime
    )
    if not candidates:
        raise SystemExit(f"no manifest for {task}")
    return candidates[-1]


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit_manifests() -> tuple[dict, dict[str, list[dict]]]:
    rows_by_task: dict[str, list[dict]] = {}
    report: dict = {"tasks": {}}
    for task in TASKS:
        path = latest_manifest(task)
        rows = load_rows(path)
        rows_by_task[task] = rows
        ids = [row["sample_id"] for row in rows]
        splits = Counter(row["split"] for row in rows)
        groups = {}
        for row in rows:
            groups.setdefault(row["group_id"], set()).add(row["split"])
        leaks = {group: sorted(found) for group, found in groups.items() if len(found) > 1}
        missing_labels = missing_images = missing_s2 = 0
        for row in rows:
            if not (DATA_ROOT / row["label_path"]).is_file():
                missing_labels += 1
            if not (DATA_ROOT / row["image_paths"][0]).is_file():
                missing_images += 1
            s2 = (row.get("aux_paths") or {}).get("s2_path")
            if not s2 or not (DATA_ROOT / s2).is_file():
                missing_s2 += 1
        report["tasks"][task] = {
            "manifest": str(path),
            "sha256": sha256(path),
            "rows": len(rows),
            "unique_sample_ids": len(set(ids)),
            "split_counts": dict(splits),
            "group_leaks": len(leaks),
            "missing_labels": missing_labels,
            "missing_images": missing_images,
            "missing_s2": missing_s2,
            "sensor": sorted({row["sensor"] for row in rows}),
            "gsd_m": sorted({row["gsd_m"] for row in rows}),
        }
    base = {row["sample_id"] for row in rows_by_task["T11"]}
    report["sample_id_sets_identical"] = all(
        {row["sample_id"] for row in rows_by_task[task]} == base for task in TASKS
    )
    report["split_identical"] = all(
        {row["sample_id"]: row["split"] for row in rows_by_task[task]}
        == {row["sample_id"]: row["split"] for row in rows_by_task["T11"]}
        for task in TASKS
    )
    return report, rows_by_task


def audit_s2(rows: list[dict]) -> tuple[dict, list[int]]:
    """Header check for every patch, pixel read for a stratified sample."""
    lengths = Counter()
    mismatched_dates = 0
    for row in rows:
        s2 = DATA_ROOT / row["aux_paths"]["s2_path"]
        sidecar = (DATA_ROOT / row["image_paths"][0]).with_suffix(".json")
        dates = json.loads(sidecar.read_text(encoding="utf-8"))["dates_s2"]
        stack = np.load(s2, mmap_mode="r")
        lengths[int(stack.shape[0])] += 1
        if stack.shape[0] != len(dates):
            mismatched_dates += 1
        if stack.shape[1:] != (10, GRID, GRID):
            raise SystemExit(f"{s2}: unexpected shape {stack.shape}")
        if stack.dtype != np.int16:
            raise SystemExit(f"{s2}: unexpected dtype {stack.dtype}")
        del stack

    rng = random.Random(17)
    sample = rng.sample(rows, 48)
    band_stats = {name: [] for name in S2_BANDS}
    negative_share = {name: [] for name in S2_BANDS}
    years = Counter()
    months_2019 = Counter()
    for row in sample:
        stack = np.load(DATA_ROOT / row["aux_paths"]["s2_path"], mmap_mode="r")
        dates = json.loads(
            (DATA_ROOT / row["image_paths"][0]).with_suffix(".json").read_text(encoding="utf-8")
        )["dates_s2"]
        for date in dates:
            years[str(date)[:4]] += 1
            if str(date).startswith("2019"):
                months_2019[str(date)[4:6]] += 1
        pixels = np.asarray(stack[:, S2_INDICES], dtype=np.float32)
        for position, name in enumerate(S2_BANDS):
            band = pixels[:, position]
            band_stats[name].append(
                {
                    "min": float(band.min()),
                    "max": float(band.max()),
                    "mean": float(band.mean()),
                    "p01": float(np.percentile(band, 1)),
                    "p99": float(np.percentile(band, 99)),
                }
            )
            negative_share[name].append(float((band < 0).mean()))

    summary = {
        "stack_lengths": dict(sorted(lengths.items())),
        "stack_length_range": [min(lengths), max(lengths)],
        "date_mismatches": mismatched_dates,
        "sampled_patches": [row["sample_id"] for row in sample],
        "acquisition_years": dict(sorted(years.items())),
        "months_2019": dict(sorted(months_2019.items())),
        "bands": S2_BANDS,
        "band_indices": S2_INDICES,
        "band_stats": {
            name: {
                key: float(np.mean([entry[key] for entry in values]))
                for key in ("min", "max", "mean", "p01", "p99")
            }
            for name, values in band_stats.items()
        },
        "negative_share": {name: float(np.mean(values)) for name, values in negative_share.items()},
    }
    return summary, [row["sample_id"] for row in sample]


def audit_teachers(sample_ids: list[str]) -> dict:
    annotations = DATA_ROOT / "derived" / "aef-2019" / "PASTIS-HD" / "ANNOTATIONS"
    aef_dir = DATA_ROOT / "derived" / "aef-2019" / "features" / "2019"
    oe_dir = DATA_ROOT / "derived" / "olmoearth-2019" / "features" / "2019"
    oe128_dir = DATA_ROOT / "derived" / "olmoearth-2019-128" / "features" / "2019"

    shapes = {"aef": set(), "oe64": set(), "oe128": set()}
    valid_equal = 0
    aef_valid_share = []
    oe_valid_share = []
    intersection_share = []
    aef_norm = []
    oe_norm = []
    oe128_norm = []
    prefix_consistency = []
    token_full = []
    token_zero = []
    token_partial = []
    token_support = []
    label_classes = Counter()
    label_shapes = set()
    token_diff = []
    token_cos = []

    for sample_id in sample_ids:
        aef_npz = np.load(aef_dir / f"{sample_id}.npz")
        oe_npz = np.load(oe_dir / f"{sample_id}.npz")
        oe128_npz = np.load(oe128_dir / f"{sample_id}.npz")
        aef = aef_npz["features"].astype(np.float32)
        oe_raw = oe_npz["features"]
        oe128_raw = oe128_npz["features"]
        oe = oe_raw.astype(np.float32)
        oe128 = oe128_raw.astype(np.float32)
        aef_valid = aef_npz["valid"]
        oe_valid = oe_npz["valid"]
        oe128_valid = oe128_npz["valid"]

        shapes["aef"].add(aef.shape)
        shapes["oe64"].add(oe.shape)
        shapes["oe128"].add(oe128.shape)

        # OlmoEarth int8 -> real value: sign(v) * (|v|/127.5)^2, NODATA -128.
        oe = np.where(oe_valid, np.sign(oe) * (np.abs(oe) / 127.5) ** 2, 0.0)
        oe128_decoded = np.where(oe128_valid, np.sign(oe128) * (np.abs(oe128) / 127.5) ** 2, 0.0)

        if np.array_equal(aef_valid, oe_valid) and np.array_equal(aef_valid, oe128_valid):
            valid_equal += 1
        intersection = aef_valid & oe_valid & oe128_valid
        aef_valid_share.append(float(aef_valid.mean()))
        oe_valid_share.append(float(oe_valid.mean()))
        intersection_share.append(float(intersection.mean()))

        norm_aef = np.linalg.norm(aef, axis=0)[aef_valid]
        norm_oe = np.linalg.norm(oe, axis=0)[oe_valid]
        norm_oe128 = np.linalg.norm(oe128_decoded, axis=0)[oe128_valid]
        aef_norm.append(float(norm_aef.mean()))
        oe_norm.append(float(norm_oe.mean()))
        oe128_norm.append(float(norm_oe128.mean()))

        # The 64-dim product is the Matryoshka prefix of the same 128-dim
        # product: values must match exactly, but the *norm* of the 64-dim
        # prefix is not the norm of the full vector, so both are reported.
        # Compare the stored int8 payloads, before dequantisation: the 64-dim
        # file must be a literal prefix of the 128-dim file.
        prefix = oe128_raw[0:64][:, oe_valid]
        prefix_consistency.append(float((prefix == oe_raw[:, oe_valid]).mean()))

        # Teacher -> Clay token grid: mean over the 8x8 pixel support, only
        # pixels both products observe (the training contract uses their
        # intersection).
        support = intersection.reshape(TOKENS, PATCH, TOKENS, PATCH).sum(axis=(1, 3))
        token_support.append(float((support > 0).mean()))
        token_full.append(float((support == PATCH * PATCH).mean()))
        token_zero.append(float((support == 0).mean()))
        token_partial.append(float(((support > 0) & (support < PATCH * PATCH)).mean()))

        aef_grid = _token_mean(aef, intersection)
        # Compare the same-width products: AEF64 against the OlmoEarth64
        # prefix (the 128-dim vector is a different output space).
        oe_grid = _token_mean(oe, intersection)
        keep = (
            (support > 0)
            & (np.linalg.norm(aef_grid, axis=0) > 0)
            & (np.linalg.norm(oe_grid, axis=0) > 0)
        )
        token_diff.append(float(np.linalg.norm(aef_grid - oe_grid, axis=0)[keep].mean()))
        unit_aef = aef_grid / np.maximum(np.linalg.norm(aef_grid, axis=0), 1e-6)
        unit_oe = oe_grid / np.maximum(np.linalg.norm(oe_grid, axis=0), 1e-6)
        cosine = (unit_aef * unit_oe).sum(axis=0)
        token_cos.append(float(cosine[keep].mean()))

        labels = np.load(annotations / f"TARGET_{sample_id}.npy")
        label_shapes.add(tuple(labels.shape))
        semantic = labels[0]
        for value in np.unique(semantic):
            label_classes[int(value)] += 1

    return {
        "sampled_patches": len(sample_ids),
        "feature_shapes": {name: sorted(map(list, values)) for name, values in shapes.items()},
        "valid_masks_identical": f"{valid_equal}/{len(sample_ids)}",
        "valid_share": {
            "aef": float(np.mean(aef_valid_share)),
            "oe64": float(np.mean(oe_valid_share)),
            "intersection": float(np.mean(intersection_share)),
            "min_intersection": float(np.min(intersection_share)),
        },
        "mean_norm": {
            "aef": float(np.mean(aef_norm)),
            "oe64": float(np.mean(oe_norm)),
            "oe128": float(np.mean(oe128_norm)),
        },
        "oe64_is_prefix_of_128": float(np.mean(prefix_consistency)),
        "token_support": {
            "has_any": float(np.mean(token_support)),
            "full": float(np.mean(token_full)),
            "partial": float(np.mean(token_partial)),
            "zero": float(np.mean(token_zero)),
            "min_zero_share": float(np.min(token_zero)),
        },
        "teacher_pair_on_tokens": {
            "mean_l2": float(np.mean(token_diff)),
            "mean_cosine": float(np.mean(token_cos)),
        },
        "label_shapes": sorted(label_shapes),
        "label_classes_seen": sorted(label_classes),
        "label_missing_void": int(VOID not in label_classes),
    }


def _token_mean(features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    channels = features.shape[0]
    values = features.reshape(channels, TOKENS, PATCH, TOKENS, PATCH)
    mask = valid.reshape(1, TOKENS, PATCH, TOKENS, PATCH).astype(np.float32)
    total = (values * mask).sum(axis=(2, 4))
    count = mask.sum(axis=(2, 4))
    return total / np.maximum(count, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest_report, rows_by_task = audit_manifests()
    s2_report, sample_ids = audit_s2(rows_by_task["T11"])
    teacher_report = audit_teachers(sample_ids)

    report = {
        "data_root": str(DATA_ROOT),
        "manifests": manifest_report,
        "s2": s2_report,
        "teachers": teacher_report,
        "labels": {"void": VOID, "classes": list(range(20))},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"manifests: {manifest_report['sample_id_sets_identical']=} "
        f"{manifest_report['split_identical']=}"
    )
    for task, entry in manifest_report["tasks"].items():
        print(
            f"  {task}: {entry['rows']} rows {entry['split_counts']} "
            f"leaks={entry['group_leaks']} missing/s2={entry['missing_s2']} "
            f"sha={entry['sha256'][:12]}"
        )
    print(
        f"S2 stacks: lengths {s2_report['stack_length_range']}, "
        f"date mismatches {s2_report['date_mismatches']}"
    )
    print(f"2019 months sampled: {s2_report['months_2019']}")
    for name, stats in s2_report["band_stats"].items():
        print(
            f"  {name}: min {stats['min']:.0f} max {stats['max']:.0f} "
            f"mean {stats['mean']:.0f} p01 {stats['p01']:.0f} p99 {stats['p99']:.0f} "
            f"negative {s2_report['negative_share'][name]:.3f}"
        )
    print(
        f"valid masks identical: {teacher_report['valid_masks_identical']}, "
        f"intersection share {teacher_report['valid_share']['intersection']:.4f}"
    )
    print(
        f"norms AEF {teacher_report['mean_norm']['aef']:.2f} "
        f"OE64 {teacher_report['mean_norm']['oe64']:.2f} "
        f"OE128 {teacher_report['mean_norm']['oe128']:.2f}"
    )
    print(f"token support: {teacher_report['token_support']}")
    print(
        f"labels: shapes {teacher_report['label_shapes']}, "
        f"classes {teacher_report['label_classes_seen']}"
    )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
