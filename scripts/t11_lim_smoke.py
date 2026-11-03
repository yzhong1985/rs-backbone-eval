#!/usr/bin/env python3
"""LIM on PASTIS S2: 20-bin forwards, memory and throughput smoke (T11-V1 pre-check).

Intended for the 20 GB host *before* wiring the T11 lane: it answers three
questions that do not need the LTAE2d head yet.

1. MixMIM encoder handles (B-Dim batch, 20, 3, 128, 128) by running the four
   pyramid stages per frame (stride 4/8/16/32);
2. ImageNet-statibrated RGB stacking of PASTIS's first three bands
   (R=B04, G=B03, B=B02 at reflectance/10000 x 255) stays in a sane value range;
3. forward-only throughput at batch 8 x 20 bins and peak memory at batch 8
   (the frozen tier does ``torch.no_grad()`` forward + a detached pyramid, so
   the encoder is out of the backward graph and the memory is bounded here).

    # 4090 host, local weights root
    uv run --project workers/lim python scripts/t11_lim_smoke.py \
        --checkpoint $LIM_CHECKPOINT \
        --s2 /.../DATA_S2/S2_10000.npy /.../DATA_S2/S2_20406.npy

    # 20 GB host (CPU torch is fine on this script's --device cuda; the smoke
    # runs at the batch below; PYTHONPATH = src + workers/lim/src assumed)
    workers/prithvi/.venv/bin/python scripts/t11_lim_smoke.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]

# MixMIM pretraining statistics, 0-255 RGB domain (NORM_DECISION a808302 keeps
# LIM on its pretraining statistics; the formal decision carries into manifest
# preprocessing_id fields).
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)
# PASTIS bands are THEIA order (B02..B12); RGB for an RGB-ish backbone is
# (B04 red, B03 green, B02 blue).
S2_RGB_BANDS = (2, 1, 0)
S2_SCALING_FACTOR = 255.0 / 10000.0


def run(args) -> int:
    sys.path.insert(0, str(REPO / "workers" / "lim" / "src"))
    from lim_worker.mit import load_encoder, mit_huge

    stacks: list[np.ndarray] = []
    for path in args.s2:
        cube = np.load(path)  # (T, 10, 128, 128) int16 DN
        bins = np.round(np.linspace(0, cube.shape[0] - 1, args.bins)).astype(int)
        # Declared: bin i takes the floor(linspace) frame; matches the同事
        # protocol family "等分 20 仓, 每仓 1 帧" at T >= 20.
        rgb = cube[bins][:, list(S2_RGB_BANDS)].astype(np.float32) * S2_SCALING_FACTOR
        mean = np.array(MEAN, dtype=np.float32)[None, :, None, None]
        std = np.array(STD, dtype=np.float32)[None, :, None, None]
        # axis: (sample, bins, channels, height, width); one patch = one sample
        stacks.append(((rgb - mean) / std)[None])
    minimum = float(np.concatenate([s[0, 0] for s in stacks]).min())
    maximum = float(np.concatenate([s[0, 0] for s in stacks]).max())

    batch = torch.from_numpy(np.concatenate(stacks, axis=0))  # (n, bins, 3, 128, 128)
    while batch.shape[0] < args.batch:
        batch = torch.cat([batch, batch[: args.batch - batch.shape[0]]], dim=0)
    device = args.device
    batch = batch.to(device)

    encoder = mit_huge(in_chans=3, drop_path_rate=0.0)
    counts = {"skip": True} if args.skip_weights else load_encoder(args.checkpoint, encoder)
    encoder.to(device).eval()

    peak = 0.0
    started = time.time()
    with torch.no_grad():
        for index in range(args.batches):
            for frame in range(batch.shape[1]):
                frames = batch[:, frame]
                features = encoder(frames)
                if index == 0 and frame == 0:
                    shapes = [tuple(tensor.shape) for tensor in features]
            if device != "cpu":
                peak = max(peak, torch.cuda.max_memory_allocated(device) / 2**20)

    seconds = time.time() - started
    samples = args.batches * batch.shape[0]
    print(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint.name),
                "skip_weights": args.skip_weights,
                "encoder_keys": counts,
                "feature_stages": [
                    {"shape": list(shape), "stride": 128 // shape[-1]} for shape in shapes
                ],
                "input": {
                    "shape": list(batch.shape),
                    "min": round(minimum, 2),
                    "max": round(maximum, 2),
                },
                "encoder_frames_per_batch": batch.shape[1],
                "batches": args.batches,
                "seconds": round(seconds, 1),
                "samples_per_second": round(samples / max(seconds, 1e-9), 1),
                "peak_gpu_mb": None if device == "cpu" else round(peak, 1),
                "device": device,
            },
            indent=2,
        )
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--s2", type=Path, nargs="+", required=True)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-weights", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
