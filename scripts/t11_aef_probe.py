#!/usr/bin/env python3
"""Frozen-AEF segmentation probes for T11 (PASTIS-HD, 2019).

Two declared heads, same data and folds:

* ``linear``  - 1x1 convolution, the representation-quality baseline;
* ``upernet`` - a small UPerNet-style decoder. AEF is a single 10 m feature
  map, so the pyramid is built from it with strided convolutions; the point is
  to match the T9 head family's capacity, not to claim a native pyramid.

Run in the LIM worker environment (CPU by default so it does not compete with a
training lane):

    uv run --project workers/lim python scripts/t11_aef_probe.py \
        --head linear --train-patches 300 --val-patches 100 --test-patches 150 --epochs 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path("/home/yzhong/rs-eval/data-aef")
MANIFEST = ROOT / "manifests/T11/pastis_hd_2019.jsonl"
MANIFEST_SHA256 = "2be347dc1215762d7d73578eb6c960cd815efbc85d1e67dcc99d715bdcc3eec4"
IGNORE = 19
CLASSES = 20


class LinearHead(nn.Module):
    def __init__(self, in_ch: int = 64, classes: int = CLASSES) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, classes, 1)

    def forward(self, x):
        return self.conv(x)


class MLPHead(nn.Module):
    """Two-layer 1x1 MLP: per-pixel nonlinearity, still no spatial context."""

    def __init__(self, in_ch: int = 64, hidden: int = 128, classes: int = CLASSES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, classes, 1),
        )

    def forward(self, x):
        return self.net(x)


class ContextHead(nn.Module):
    """Stacked 3x3 convs with dilation 1/2/4: cheap local context (~15 px)."""

    def __init__(self, in_ch: int = 64, width: int = 64, classes: int = CLASSES) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_ch
        for dilation in (1, 2, 4):
            layers.extend(
                (
                    nn.Conv2d(channels, width, 3, padding=dilation, dilation=dilation, bias=False),
                    nn.GroupNorm(8, width),
                    nn.GELU(),
                )
            )
            channels = width
        self.body = nn.Sequential(*layers)
        self.out = nn.Conv2d(width, classes, 1)

    def forward(self, x):
        return self.out(self.body(x))


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.norm = nn.GroupNorm(8, out_ch)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))


class UPerNetLite(nn.Module):
    """PPM + FPN over a synthetic 1/2/4/8 pyramid from the AEF map."""

    def __init__(self, in_ch: int = 64, classes: int = CLASSES, width: int = 128) -> None:
        super().__init__()
        self.stem = ConvBlock(in_ch, width)
        self.down2 = ConvBlock(width, width * 2, 2)
        self.down3 = ConvBlock(width * 2, width * 2, 2)
        self.down4 = ConvBlock(width * 2, width * 2, 2)
        self.ppm = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(size),
                    nn.Conv2d(width * 2, width, 1, bias=False),
                    nn.GroupNorm(8, width),
                    nn.GELU(),
                )
                for size in (1, 2, 3, 6)
            ]
        )
        self.ppm_fuse = nn.Sequential(
            nn.Conv2d(width * 6, width * 2, 1, bias=False),
            nn.GroupNorm(8, width * 2),
            nn.GELU(),
        )
        self.lateral = nn.ModuleList(
            [nn.Conv2d(ch, width, 1) for ch in (width, width * 2, width * 2, width * 2)]
        )
        self.smooth = nn.ModuleList([ConvBlock(width, width) for _ in range(4)])
        self.head = nn.Sequential(
            ConvBlock(width * 4, width * 2),
            nn.Conv2d(width * 2, classes, 1),
        )

    def forward(self, x):
        size = x.shape[-2:]
        l1 = self.stem(x)
        l2 = self.down2(l1)
        l3 = self.down3(l2)
        l4 = self.down4(l3)
        pooled = [module(l4) for module in self.ppm]
        pooled = [
            F.interpolate(item, size=l4.shape[-2:], mode="bilinear", align_corners=False)
            for item in pooled
        ]
        l4 = self.ppm_fuse(torch.cat([l4, *pooled], dim=1))
        levels = [l1, l2, l3, l4]
        laterals = [module(level) for module, level in zip(self.lateral, levels, strict=True)]
        for index in range(2, -1, -1):
            laterals[index] = laterals[index] + F.interpolate(
                laterals[index + 1],
                size=laterals[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        merged = [module(item) for module, item in zip(self.smooth, laterals, strict=True)]
        merged = [
            F.interpolate(item, size=size, mode="bilinear", align_corners=False) for item in merged
        ]
        return self.head(torch.cat(merged, dim=1))


HEAD_BUILDERS = {
    "linear": LinearHead,
    "mlp": MLPHead,
    "conv": ContextHead,
    "upernet": UPerNetLite,
}
HEAD_DEFAULT_LR = {"linear": 1e-2, "mlp": 3e-3, "conv": 1e-3, "upernet": 1e-3}


def load_rows() -> dict[str, list[dict]]:
    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["split"], []).append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["sample_id"]))
    return grouped


def stratified(rows: list[dict], n: int) -> list[dict]:
    """Deterministic tile-stratified subset.

    The sample_id prefix encodes the tile (10000s = t30uxv, ...), so a sorted
    prefix is a single-region sample (review 2026-11-03, #2: the fixed-budget
    subset was 297/300 t30uxv in train and 100% t30uxv in validation). Tiles
    get a largest-remainder quota and the selection alternates tiles so the
    prefix order does not reintroduce the bias.
    """
    ordered = sorted(rows, key=lambda row: int(row["sample_id"]))
    if n >= len(ordered):
        return ordered
    by_tile: dict[str, list[dict]] = {}
    for row in ordered:
        by_tile.setdefault(row["tile"], []).append(row)
    tiles = sorted(by_tile)
    quotas = {tile: n * len(by_tile[tile]) // len(rows) for tile in tiles}
    remainders = sorted(tiles, key=lambda tile: (-(n * len(by_tile[tile]) % len(rows)), tile))
    while sum(quotas.values()) < n:
        progressed = False
        for tile in remainders:
            if quotas[tile] < len(by_tile[tile]):
                quotas[tile] += 1
                progressed = True
                break
        if not progressed:
            break
    picked: list[dict] = []
    while sum(quotas[tile] for tile in tiles) > 0:
        progressed = False
        for tile in tiles:
            if quotas.get(tile, 0) > 0 and by_tile[tile]:
                picked.append(by_tile[tile].pop(0))
                quotas[tile] -= 1
                progressed = True
        if not progressed:
            break
    return picked


def load_patch(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(ROOT / row["aef_path"])
    features = archive["features"].astype(np.float32)
    valid = archive["valid"]
    labels = np.load(ROOT / row["label_path"])[0].astype(np.int64)
    return features, valid, labels


def feature_stats(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(64, dtype=np.float64)
    squares = np.zeros(64, dtype=np.float64)
    count = 0
    for row in rows:
        features, valid, _ = load_patch(row)
        values = np.sign(features) * (features / 127.5) ** 2
        values = values[:, valid]
        total += values.sum(axis=1)
        squares += (values**2).sum(axis=1)
        count += values.shape[1]
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0)) + 1e-3
    return mean.astype(np.float32), std.astype(np.float32)


def to_tensor(batch: list[dict], mean: np.ndarray, std: np.ndarray):
    features, labels = [], []
    for row in batch:
        values, valid, target = load_patch(row)
        values = np.sign(values) * (values / 127.5) ** 2
        values = (values - mean[:, None, None]) / std[:, None, None]
        values[:, ~valid] = 0
        features.append(torch.from_numpy(values))
        labels.append(torch.from_numpy(target))
    return torch.stack(features), torch.stack(labels)


def evaluate(
    model, rows, mean, std, batch_size: int, device: str = "cpu"
) -> tuple[float, float, dict[int, float]]:
    model.eval()
    confusion = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            x, y = to_tensor(rows[start : start + batch_size], mean, std)
            # Inputs must move to the model's device and the prediction back to
            # the CPU-side confusion matrix (review 2026-11-03, round 5, P2:
            # --device cuda crashed at the first validation pass).
            predicted = model(x.to(device)).argmax(dim=1).cpu()
            mask = y != IGNORE
            flat = (y[mask] * CLASSES + predicted[mask]).numpy()
            confusion += np.bincount(flat, minlength=CLASSES * CLASSES).reshape(CLASSES, CLASSES)
    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    present = confusion.sum(axis=1) > 0
    iou = np.where(present, tp / np.maximum(tp + fp + fn, 1), np.nan)
    foreground = [float(iou[index]) for index in range(1, 19) if present[index]]
    per_class = {index: round(float(value), 4) for index, value in enumerate(iou) if present[index]}
    return (
        float(np.nanmean(iou)),
        float(np.mean(foreground)) if foreground else float("nan"),
        per_class,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=tuple(HEAD_BUILDERS), default="linear")
    parser.add_argument("--train-patches", type=int, default=300)
    parser.add_argument("--val-patches", type=int, default=100)
    parser.add_argument("--test-patches", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(17)
    grouped = load_rows()
    train = stratified(grouped["train"], args.train_patches)
    val = stratified(grouped["validation"], args.val_patches)
    test = stratified(grouped["test"], args.test_patches)
    # Same stratification for the statistics sample, independently of the
    # train budget (review 2026-11-03, #2).
    mean, std = feature_stats(stratified(grouped["train"], 80))

    model = HEAD_BUILDERS[args.head]().to(args.device)
    lr = args.lr if args.lr is not None else HEAD_DEFAULT_LR[args.head]
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    started = time.time()
    history = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        batches = 0
        # Per-epoch shuffle with a manual, epoch-bound seed: iterating a
        # tile-ordered list without shuffling trained one region at a time and
        # skewed the big-head comparison (review 2026-11-03, #3).
        order = torch.randperm(
            len(train), generator=torch.Generator().manual_seed(1_000 + epoch)
        ).tolist()
        for start in range(0, len(train), args.batch):
            batch_rows = [train[index] for index in order[start : start + args.batch]]
            x, y = to_tensor(batch_rows, mean, std)
            x, y = x.to(args.device), y.to(args.device)
            loss = F.cross_entropy(model(x), y, ignore_index=IGNORE)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            batches += 1
        history.append(round(total / max(batches, 1), 4))
        print(f"epoch {epoch} loss {history[-1]}", flush=True)

    val_all, val_fg, val_classes = evaluate(model, val, mean, std, args.batch, device=args.device)
    test_all, test_fg, test_classes = evaluate(
        model, test, mean, std, args.batch, device=args.device
    )
    seconds = round(time.time() - started, 1)
    result = {
        "head": args.head,
        "params": sum(p.numel() for p in model.parameters()),
        "sampling": "tile_stratified_subset+per_epoch_shuffle",
        "manifest_sha256": MANIFEST_SHA256,
        "train_patches": len(train),
        "val_patches": len(val),
        "test_patches": len(test),
        "epochs": args.epochs,
        "lr": lr,
        "history": history,
        "val_miou_all": round(val_all, 4),
        "val_miou_foreground": round(val_fg, 4),
        "test_miou_all": round(test_all, 4),
        "test_miou_foreground": round(test_fg, 4),
        "test_per_class_iou": test_classes,
        "seconds": seconds,
        "note": "technical probe; labels are 2019 pending the LPIS-year confirmation",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
