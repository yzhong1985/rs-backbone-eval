#!/usr/bin/env python3
"""Diagnostic probe: does AEF context help the FAIR1M detector?

Three variants share the same tiles, budget and frozen LIM encoder:

    none      LIM only
    real      LIM + the tile's AEF window (64 bands at 10 m, /8 grid)
    shuffled  LIM + the same AEF values, spatially permuted (capacity control)

Tile-level AP only, one seed, a few thousand steps: this is a direction check for
the P2 design, not a reportable result. Run it in the LIM worker environment:

    workers/lim/.venv/bin/python scripts/aef_fusion_probe.py \
        --features /home/yzhong/rs-eval/aef-features --variant real --steps 2000

The AEF year is an engineering choice, never the image acquisition year.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "workers/lim/src")

from lim_worker.mit import load_encoder, mit_huge  # noqa: E402
from lim_worker.t10 import (  # noqa: E402
    AnnotationCache,
    Fair1mTiles,
    LimDetector,
)

from rs_backbone_eval.detection.aef_context import AefContext, shuffle_within_tile  # noqa: E402
from rs_backbone_eval.detection.detector import GroundTruth  # noqa: E402

MANIFEST = Path("/geodata/rs-eval/data/manifests/T10/holdout_1024_s824_seed17.jsonl")
DATA_ROOT = Path("/geodata/rs-eval/data")
CHECKPOINT = Path("/geodata/rs-eval/weights/lim/mit_huge_600.pth")
MEAN = np.array([69.11, 73.86, 63.45], dtype=np.float32)
STD = np.array([60.88, 57.94, 52.91], dtype=np.float32)


def select_rows(
    manifest: Path, train_images: int, val_images: int
) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in manifest.open(encoding="utf-8")]

    def first_images(split: str, count: int) -> list[dict]:
        picked, groups = [], []
        for row in rows:
            if row["split"] != split:
                continue
            if row["group_id"] not in groups:
                if len(groups) >= count:
                    continue
                groups.append(row["group_id"])
            picked.append(row)
        return picked

    return first_images("train", train_images), first_images("validation", val_images)


def target_of(item: dict, device: str) -> GroundTruth:
    return GroundTruth(
        boxes=item["boxes"].to(device),
        labels=item["labels"].to(device),
        ignored_boxes=item["ignored_boxes"].to(device),
        valid_region=(
            float(item["valid"][0]),
            float(item["valid"][1]),
            float(item["valid"][0] + item["valid"][2]),
            float(item["valid"][1] + item["valid"][3]),
        ),
    )


def load_context(features: Path, sample_id: str, variant: str, device: str) -> torch.Tensor:
    payload = np.load(features / f"{sample_id}.npz")
    tensor = torch.from_numpy(payload["features"].astype(np.float32))[None].to(device)
    if variant == "shuffled":
        generator = torch.Generator(device=device).manual_seed(abs(hash(sample_id)) % (2**31))
        tensor = shuffle_within_tile(tensor, generator)
    return tensor


def tile_ap(predictions, targets, num_classes: int) -> tuple[float, float]:
    from rs_backbone_eval.detection.ap import evaluate

    metrics = evaluate(predictions, targets, num_classes)
    return float(metrics["map"]), float(metrics["map_50"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--features", type=Path, default=Path("/home/yzhong/rs-eval/aef-features"))
    parser.add_argument("--variant", choices=("none", "real", "shuffled"), required=True)
    parser.add_argument(
        "--anchors",
        choices=("default", "small"),
        default="default",
        help="default = 32-256/r0.5-2 (locked); small = 16-256/r0.5-4 (diagnostic)",
    )
    parser.add_argument("--train-images", type=int, default=40)
    parser.add_argument("--val-images", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=Path, default=Path("/home/yzhong/rs-eval/aef-fusion"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_rows, val_rows = select_rows(args.manifest, args.train_images, args.val_images)
    if args.variant != "none":
        missing = [
            row["sample_id"]
            for row in train_rows + val_rows
            if not (args.features / f"{row['sample_id']}.npz").is_file()
        ]
        if missing:
            raise SystemExit(f"{len(missing)} tiles have no AEF features, e.g. {missing[0]}")
    annotations = AnnotationCache(DATA_ROOT, 0.7)
    train_ds = Fair1mTiles(train_rows, DATA_ROOT, 1024, MEAN, STD, annotations)
    val_ds = Fair1mTiles(val_rows, DATA_ROOT, 1024, MEAN, STD, annotations)

    encoder = mit_huge(in_chans=3)
    load_encoder(CHECKPOINT, encoder)
    anchor_sets = {
        "default": (None, None),
        "small": ((16, 32, 64, 128, 256), (0.5, 1.0, 2.0, 4.0)),
    }
    anchor_sizes, anchor_ratios = anchor_sets[args.anchors]
    model = LimDetector(encoder, 37, anchor_sizes=anchor_sizes, anchor_ratios=anchor_ratios).to(
        device
    )
    for parameter in model.detector.backbone.parameters():
        parameter.requires_grad_(False)
    context_head = None if args.variant == "none" else AefContext(256, level=1).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if context_head is not None:
        params += list(context_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    model.train()

    from rs_backbone_eval.detection.ap import Predictions, Targets
    from rs_backbone_eval.detection.geometry import poly2obb

    started = time.time()
    history = []
    for step in range(1, args.steps + 1):
        index = step % len(train_ds)
        item = train_ds[index]
        pixels = item["image"][None].to(device)
        target = target_of(item, device)
        features = model.detector.pyramid(pixels)
        if context_head is not None:
            context = load_context(
                args.features, train_rows[index]["sample_id"], args.variant, device
            )
            features = context_head(features, context)
        losses = model.detector.losses(features, [target])
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 35.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            predictions, targets = {}, {}
            with torch.inference_mode():
                for val_index, row in enumerate(val_rows):
                    item = val_ds[val_index]
                    pixels = item["image"][None].to(device)
                    features = model.detector.pyramid(pixels)
                    if context_head is not None:
                        context = load_context(
                            args.features, row["sample_id"], args.variant, device
                        )
                        features = context_head(features, context)
                    out = model.detector.detections(features, (1024, 1024))[0]
                    scene = Predictions()
                    for polygon, score, label in zip(
                        out["polygons"].cpu().numpy(),
                        out["scores"].tolist(),
                        out["labels"].tolist(),
                        strict=True,
                    ):
                        scene.append(polygon, int(label), float(score))
                    predictions[row["sample_id"]] = scene
                    ground = Targets()
                    for polygon, class_id in annotations.split_tile(
                        row["annotation_path"], [int(v) for v in row["window"]]
                    )[0]:
                        _, _, w, h, _ = poly2obb(polygon)
                        if min(w, h) >= 0.5:
                            ground.append(polygon, class_id)
                    targets[row["sample_id"]] = ground
            m, m50 = tile_ap(predictions, targets, 37)
            model.train()
            history.append(
                {
                    "step": step,
                    "map": round(m, 5),
                    "map_50": round(m50, 5),
                    "loss": round(float(losses["loss"].detach()), 4),
                }
            )
            print(
                json.dumps(
                    {
                        "variant": args.variant,
                        **history[-1],
                        "seconds": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "variant": args.variant,
        "anchors": args.anchors,
        "steps": args.steps,
        "train_images": args.train_images,
        "val_images": args.val_images,
        "train_tiles": len(train_rows),
        "val_tiles": len(val_rows),
        "tile_level_ap": True,
        "note": "tile-level AP on a few images; diagnostic direction check, not a formal result",
        "history": history,
    }
    (args.out / f"probe_{args.variant}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
