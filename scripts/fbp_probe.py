#!/usr/bin/env python3
"""Temporal-mismatch probe on the cached FBP windows (assessment §5.5).

Trains the same MLP head as T11O on frozen embeddings and reports macro IoU on
the footprint-group holdout. This is the exploratory panel from the assessment:
labels are 2014-2016, embeddings are 2017, so the numbers are **not** comparable
with same-year results and must not be ranked against PASTIS/T12.

Protocol (assessment §5.2/§5.5):

* split by RPC-footprint group, 60/20/20 by scene count, largest group first,
  seed 17 — the official 30-scene test is *not* used (it shares footprints);
* four 512x512 windows per scene, chosen from the labels only;
* dense windows, ``Conv1x1 -> GroupNorm(8,128) -> GELU -> Conv1x1``
  (``workers/aef/src/aef_worker/heads.py``), lr 3e-3, 8 epochs, batch 2;
* ignore = 0 (unlabeled or >50% unlabeled) and any AEF/OlmoEarth fill pixel;
  a window with fewer than ``MIN_WINDOW_VALID`` usable pixels is dropped and
  counted (review 2026-11-11: two scenes have windows with no 2017 observation).

Usage::

    workers/prithvi/.venv/bin/python scripts/fbp_probe.py --product olmo \\
        --cache /home/yzhong/rs-eval/rs-eval-cache/fbp2017 --out results/olmo.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

CACHE_DEFAULT = Path("/home/yzhong/rs-eval/rs-eval-cache/fbp2017")
FBP_ROOT = Path("/geodata/Five-Billion-Pixels")
FILL = -128
CLASSES = 24
IGNORE = 0
WINDOW = 512
MIN_WINDOW_VALID = 0.05
SPLIT = {"train": 0.6, "val": 0.2, "test": 0.2}


def footprints() -> dict[str, str]:
    from rs_backbone_eval.data.datasets.five_billion_pixels import (
        COORDINATE_DIR,
        IMAGE_DIR,
        dataset_root,
        footprint_groups,
        rpc_box,
    )

    root = dataset_root()
    images = sorted(path.name[:-5] for path in (root / IMAGE_DIR).glob("*.tiff"))
    boxes = {name: rpc_box(root / COORDINATE_DIR / f"{name}.rpb") for name in images}
    return footprint_groups(boxes)


def group_split(groups: dict[str, str], seed: int = 17) -> dict[str, str]:
    """Whole footprint groups into train/val/test, largest first, deterministic."""
    members: dict[str, list[str]] = {}
    for scene, group in groups.items():
        members.setdefault(group, []).append(scene)
    order = sorted(members, key=lambda group: (-len(members[group]), group))
    rng = random.Random(seed)
    rng.shuffle(order)
    order.sort(key=lambda group: -len(members[group]))
    total = len(groups)
    counts = dict.fromkeys(SPLIT, 0)
    split: dict[str, str] = {}
    for group in order:
        best = max(SPLIT, key=lambda name: SPLIT[name] * total - counts[name])
        for scene in members[group]:
            split[scene] = best
        counts[best] += len(members[group])
    return split


def load(scene: str, cache: Path, product: str) -> list[dict]:
    labels = np.load(cache / "labels" / f"{scene}.npz")
    name = "aef" if product == "aef" else "olmo"
    features = np.load(cache / name / f"{scene}.npz")["features"]
    raster = labels["labels"]
    out = []
    for index, row in enumerate(labels["windows"]):
        top, left = int(row[0]), int(row[1])
        target = raster[top : top + WINDOW, left : left + WINDOW]
        source = features[index]
        usable = (target != IGNORE) & (source[0] != FILL)
        if usable.mean() < MIN_WINDOW_VALID:
            continue
        out.append(
            {
                "x": source.astype(np.float32),
                "y": target.astype(np.int64),
                "usable": usable,
            }
        )
    return out


def build_head(channels: int, hidden: int = 128):
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(channels, hidden, 1),
        nn.GroupNorm(8, hidden),
        nn.GELU(),
        nn.Conv2d(hidden, CLASSES + 1, 1),
    )


def evaluate(model, samples: list[dict], device) -> dict:
    import torch

    confusion = np.zeros((CLASSES + 1, CLASSES + 1), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for sample in samples:
            x = torch.from_numpy(sample["x"]).unsqueeze(0).to(device)
            prediction = model(x).argmax(1)[0].cpu().numpy()
            mask = sample["usable"]
            target = sample["y"][mask]
            predicted = prediction[mask]
            keep = target != IGNORE
            np.add.at(confusion, (target[keep], predicted[keep]), 1)
    ious = {}
    for cls in range(1, CLASSES + 1):
        tp = confusion[cls, cls]
        fp = confusion[:, cls].sum() - tp
        fn = confusion[cls, :].sum() - tp
        if tp + fp + fn:
            ious[cls] = float(tp / (tp + fp + fn))
    total = confusion.sum()
    return {
        "macro_iou": float(np.mean(list(ious.values()))) if ious else 0.0,
        "pixel_accuracy": float(np.trace(confusion) / total) if total else 0.0,
        "class_iou": ious,
        "pixels": int(total),
    }


def train_once(
    samples: dict[str, list[dict]],
    channels: int,
    seed: int,
    epochs: int,
    lr: float,
    batch: int,
    device: str,
) -> dict:
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_head(channels).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    train_samples = samples["train"]
    for epoch in range(epochs):
        model.train()
        order = np.random.permutation(len(train_samples))
        running = 0.0
        for start in range(0, len(order), batch):
            batch_samples = [train_samples[i] for i in order[start : start + batch]]
            xs = torch.from_numpy(np.stack([s["x"] for s in batch_samples])).to(device)
            ys = torch.from_numpy(np.stack([s["y"] for s in batch_samples])).to(device)
            masks = torch.from_numpy(np.stack([s["usable"] for s in batch_samples])).to(device)
            logits = model(xs).permute(0, 2, 3, 1)  # (batch, H, W, classes)
            loss = loss_fn(logits[masks], ys[masks])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            running += loss.item()
        print(
            f"  seed {seed} epoch {epoch + 1}/{epochs} loss {running / max(1, len(order)):.4f}",
            flush=True,
        )
    return evaluate(model, samples["test"], device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", choices=["aef", "olmo"], required=True)
    parser.add_argument("--cache", type=Path, default=CACHE_DEFAULT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 73])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}", flush=True)
    groups = footprints()
    split = group_split(groups)
    samples: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for scene in sorted(split):
        part = load(scene, args.cache, args.product)
        samples[split[scene]].extend(part)
    print(
        f"{args.product}: scenes {len(split)} | windows train {len(samples['train'])} "
        f"val {len(samples['val'])} test {len(samples['test'])}",
        flush=True,
    )
    channels = samples["train"][0]["x"].shape[0]
    runs = []
    for seed in args.seeds:
        metrics = train_once(samples, channels, seed, args.epochs, args.lr, args.batch, device)
        metrics["seed"] = seed
        runs.append(metrics)
        print(
            f"seed {seed}: macro IoU {metrics['macro_iou']:.4f} "
            f"acc {metrics['pixel_accuracy']:.4f}",
            flush=True,
        )

    values = [run["macro_iou"] for run in runs]
    summary = {
        "product": args.product,
        "channels": channels,
        "epochs": args.epochs,
        "lr": args.lr,
        "seeds": args.seeds,
        "macro_iou_mean": float(np.mean(values)),
        "macro_iou_std": float(np.std(values)),
        "window_counts": {k: len(v) for k, v in samples.items()},
        "runs": runs,
        "note": "temporal-mismatch panel (labels 2014-2016 vs embeddings 2017); "
        "not comparable with same-year results and not for ranking",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"{args.product}: macro IoU {summary['macro_iou_mean']:.4f} "
        f"± {summary['macro_iou_std']:.4f} -> {args.out}"
    )


if __name__ == "__main__":
    main()
