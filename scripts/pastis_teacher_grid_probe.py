#!/usr/bin/env python3
"""Same-grid teacher reference for the T11D four-band student (diagnostic only).

The distilled probes read the teacher through an 80 m token grid. To separate
"the student did not learn the teacher" from "the teacher itself loses accuracy
at that support", this script trains the *same* MLP head and the *same* budget
directly on the aggregated teacher vectors and evaluates on the same 10 m label
grid.

It is **not** a model run and writes no ``run.json``: there is no student
forward, so recording it as a Clay result would mislabel the ledger. The output
is a reference table stored next to the audit artifacts.

Usage::

    workers/prithvi/.venv/bin/python scripts/pastis_teacher_grid_probe.py \\
        --out /home/yzhong/rs-eval/audits/pastis-rgbn/teacher_grid.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers" / "clay" / "src"))

import clay_worker.rgbn as rgbn  # noqa: E402

MANIFEST = Path("/geodata/rs-eval/data/manifests/T11D")
DATA_ROOT = Path("/geodata/rs-eval/data")
TARGETS = ("aef64", "oe64", "oe128")
EPOCHS = 8
LEARNING_RATE = 3e-3
BATCH = 2


def groups(rows: list[dict]) -> dict[str, list[dict]]:
    return {
        split: sorted((row for row in rows if row["split"] == split), key=lambda r: r["sample_id"])
        for split in ("train", "validation", "test")
    }


def _features(target: str, row: dict, normalize: bool) -> np.ndarray:
    grid = rgbn.teacher_tokens(target, row, DATA_ROOT)[0]
    if normalize:
        norm = np.linalg.norm(grid, axis=0, keepdims=True)
        grid = grid / np.maximum(norm, 1e-6)
    return grid


def train_target(
    target: str, seed: int, rows: list[dict], device: str, normalize: bool = False
) -> dict:
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    dims = rgbn.TARGET_TREE_DIMS[target][1]
    split_rows = groups(rows)
    head = rgbn.build_head(dims, 20).to(device)
    optimiser = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    for epoch in range(EPOCHS):
        order = torch.randperm(
            len(split_rows["train"]), generator=torch.Generator().manual_seed(1_000 * seed + epoch)
        ).tolist()
        head.train()
        total, steps = 0.0, 0
        for start in range(0, len(order), BATCH):
            chosen = [split_rows["train"][index] for index in order[start : start + BATCH]]
            features = torch.stack(
                [torch.from_numpy(_features(target, row, normalize)) for row in chosen]
            ).to(device)
            labels = torch.stack(
                [
                    torch.from_numpy(rgbn._labels(row, DATA_ROOT, rgbn.VOID_CLASS)).to(device)
                    for row in chosen
                ]
            ).long()
            logits = head(features)
            logits = F.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = F.cross_entropy(logits, labels, ignore_index=rgbn.VOID_CLASS)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            steps += 1
        print(f"  {target} seed {seed} epoch {epoch}: loss {total / max(steps, 1):.4f}", flush=True)

    result = {
        "target": target,
        "seed": seed,
        "dims": dims,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "normalized_features": normalize,
    }
    for split in ("validation", "test"):
        head.eval()
        confusion = np.zeros((20, 20), dtype=np.int64)
        with torch.no_grad():
            for start in range(0, len(split_rows[split]), BATCH):
                chosen = split_rows[split][start : start + BATCH]
                features = torch.stack(
                    [
                        torch.from_numpy(rgbn.teacher_tokens(target, row, DATA_ROOT)[0])
                        for row in chosen
                    ]
                ).to(device)
                labels = torch.stack(
                    [
                        torch.from_numpy(rgbn._labels(row, DATA_ROOT, rgbn.VOID_CLASS)).to(device)
                        for row in chosen
                    ]
                ).long()
                logits = head(features)
                logits = F.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                prediction = logits.argmax(dim=1)
                mask = labels != rgbn.VOID_CLASS
                flat = (labels[mask] * 20 + prediction[mask]).cpu().numpy()
                confusion += np.bincount(flat, minlength=400).reshape(20, 20)
        tp = np.diag(confusion)
        fp = confusion.sum(axis=0) - tp
        fn = confusion.sum(axis=1) - tp
        present = confusion.sum(axis=1) > 0
        iou = np.where(present, tp / np.maximum(tp + fp + fn, 1), np.nan)
        result[f"{split}_samples"] = len(split_rows[split])
        result[f"{split}_mean_iou"] = float(np.nanmean(iou))
        result[f"{split}_iou"] = {
            str(index): float(iou[index]) for index in range(20) if present[index]
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="feed unit-norm teacher tokens (the direction-only target the cosine "
        "distillation actually asks the student to predict)",
    )
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = sorted(MANIFEST.glob("*.jsonl"), key=lambda path: path.stat().st_mtime)[-1]
    rows = rgbn.read_rows(manifest)
    print(f"manifest {manifest.name}, {len(rows)} rows, device {device}", flush=True)
    runs = []
    for target in args.targets:
        for seed in args.seeds:
            runs.append(train_target(target, seed, rows, device, normalize=args.normalize))
    payload = {
        "note": "same-grid teacher reference (diagnostic, not a model run): the teacher "
        "vectors are aggregated to Clay's 16x16 token grid and read by the same MLP head, "
        "lr 3e-3 and 8 epochs as the distilled probes; evaluated at 10 m with bilinear "
        "upsampling of the token logits.",
        "manifest": str(manifest),
        "manifest_sha256": rgbn.hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for run in runs:
        print(
            f"{run['target']} seed {run['seed']}: "
            f"val {run['validation_mean_iou']:.4f} test {run['test_mean_iou']:.4f}"
        )
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
