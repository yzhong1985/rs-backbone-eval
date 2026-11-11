#!/usr/bin/env python3
"""Temporal-mismatch probe on the cached FBP windows (assessment §5.5).

Trains the T11O MLP head on frozen embeddings and reports macro IoU on the
footprint-group holdout. This is the exploratory panel from the assessment:
labels are 2014-2016 and embeddings are 2017, so the numbers are **not**
comparable with same-year results and must not be ranked against PASTIS/T12.

Protocol (assessment §5.2/§5.5):

* split by RPC-footprint group, 60/20/20 by scene count, largest group first,
  seed 17 — the official 30-scene test is *not* used (it shares footprints);
* four 512x512 windows per scene, chosen from the labels only;
* dense windows, ``Conv1x1 -> GroupNorm(8,128) -> GELU -> Conv1x1``
  (``workers/aef/src/aef_worker/heads.py``), lr 3e-3, 8 epochs, batch 2;
* ignore = 0 (unlabeled or >50% unlabeled) and any AEF/OlmoEarth fill pixel;
  a window with fewer than ``MIN_WINDOW_VALID`` usable pixels is dropped and
  counted, per window rather than per scene (review 2026-11-11: three scenes
  have two windows with no 2017 AEF observation).

Scenes are loaded lazily one at a time: the full cache is 5 GB (AEF) / 14 GB
(OlmoEarth) on disk and does not fit in memory as arrays (the first version
OOM-killed the OlmoEarth run).

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
    order = sorted(members, key=lambda group: -len(members[group]))
    random.Random(seed).shuffle(order)
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


class Windows:
    """Lazy per-scene window loader with a small cache.

    A single OlmoEarth scene is four windows of 128x512x512 float16 (268 MB);
    holding all 150 scenes at once is what OOM-killed the first run.

    Both products are loaded for every window so the usable mask is the
    **intersection** of their observations. Evaluating AEF on pixels OlmoEarth
    does not cover (and vice versa) would mix the product difference with a
    difference in evaluation positions (review 2026-11-11, P2: 251,627 test
    pixels differed).
    """

    def __init__(self, cache: Path, product: str, keep: int = 4) -> None:
        self.cache = cache
        self.product = "aef" if product == "aef" else "olmo"
        self.other = "olmo" if self.product == "aef" else "aef"
        self.keep = keep
        self._cache: dict[str, list[dict]] = {}
        self._seen: set[str] = set()
        self.dropped = 0
        self.count = 0

    def scene(self, name: str) -> list[dict]:
        if name in self._cache:
            return self._cache[name]
        labels = np.load(self.cache / "labels" / f"{name}.npz")
        features = np.load(self.cache / self.product / f"{name}.npz")["features"]
        other = np.load(self.cache / self.other / f"{name}.npz")["features"]
        raster = labels["labels"]
        samples = []
        for index, row in enumerate(labels["windows"]):
            top, left = int(row[0]), int(row[1])
            target = raster[top : top + WINDOW, left : left + WINDOW]
            source = features[index]
            usable = (target != IGNORE) & (source[0] != FILL) & (other[index][0] != FILL)
            if usable.mean() < MIN_WINDOW_VALID:
                if name not in self._seen:
                    self.dropped += 1
                continue
            samples.append(
                {
                    "x": source.astype(np.float16),
                    "y": target.astype(np.int64),
                    "usable": usable,
                }
            )
        if name not in self._seen:  # count each scene once, not once per epoch
            self._seen.add(name)
            self.count += len(samples)
        if len(self._cache) >= self.keep:
            self._cache.pop(next(iter(self._cache)))
        self._cache[name] = samples
        return samples


def build_head(channels: int, hidden: int = 128):
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(channels, hidden, 1),
        nn.GroupNorm(8, hidden),
        nn.GELU(),
        nn.Conv2d(hidden, CLASSES + 1, 1),
    )


def _batch(samples: list[dict], device):
    import torch

    xs = torch.from_numpy(np.stack([s["x"] for s in samples])).to(device).float()
    ys = torch.from_numpy(np.stack([s["y"] for s in samples])).to(device)
    masks = torch.from_numpy(np.stack([s["usable"] for s in samples])).to(device)
    return xs, ys, masks


def evaluate(model, scenes: list[str], loader: Windows, device) -> dict:
    import torch

    confusion = np.zeros((CLASSES + 1, CLASSES + 1), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for name in scenes:
            for sample in loader.scene(name):
                x = torch.from_numpy(sample["x"]).unsqueeze(0).to(device).float()
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
        "confusion": confusion.tolist(),
    }


def train_once(
    scenes: dict[str, list[str]],
    loader: Windows,
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
    train_scenes = list(scenes["train"])
    for epoch in range(epochs):
        model.train()
        random.Random(seed + epoch).shuffle(train_scenes)
        running, steps = 0.0, 0
        for name in train_scenes:
            samples = loader.scene(name)
            for start in range(0, len(samples), batch):
                xs, ys, masks = _batch(samples[start : start + batch], device)
                logits = model(xs).permute(0, 2, 3, 1)  # (batch, H, W, classes)
                loss = loss_fn(logits[masks], ys[masks])
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                running += loss.item()
                steps += 1
        print(
            f"  seed {seed} epoch {epoch + 1}/{epochs} loss {running / max(1, steps):.4f}",
            flush=True,
        )
    return evaluate(model, scenes["test"], loader, device)


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
    split = group_split(footprints())
    scenes: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for scene, part in sorted(split.items()):
        scenes[part].append(scene)
    print(f"{args.product}: scenes {[len(v) for v in scenes.values()]}", flush=True)

    loader = Windows(args.cache, args.product)
    channels = 64 if args.product == "aef" else 128
    runs = []
    for seed in args.seeds:
        metrics = train_once(
            scenes, loader, channels, seed, args.epochs, args.lr, args.batch, device
        )
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
        "scenes": {k: len(v) for k, v in scenes.items()},
        "usable_windows": loader.count,
        "shared_mask": True,
        "dropped_windows": loader.dropped,
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
