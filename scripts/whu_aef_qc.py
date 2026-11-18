#!/usr/bin/env python3
"""Overlay QC for the WHU-OPT-SAR <-> AEF alignment (T13 G0b).

One row per sheet, three panels: GF-1 RGB, labels, AEF PCA (3 leading
components). The AEF grid is the sheet grid at 2 px per cell, so a shift shows
up as rivers, roads and parcel edges cutting through the wrong PCA colours.

    workers/lim/.venv/bin/python scripts/whu_aef_qc.py NH49E004021 NH49E003024 \\
        --out /home/yzhong/rs-eval/audits/whu-aef2019-qc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

WHU = Path("/home/yzhong/rs-eval-data/whu-opt-sar")
AEF = Path("/home/yzhong/rs-eval-cache/whu-aef2019/sheets")
PALETTE = np.array(
    [
        [0, 0, 0],
        [255, 230, 120],
        [220, 40, 40],
        [250, 140, 160],
        [40, 90, 230],
        [30, 140, 40],
        [140, 140, 140],
        [160, 90, 200],
    ],
    dtype=np.uint8,
)  # background, farmland, city, village, water, forest, road, other
STEP = 8  # image px per rendered px


def stretch(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros(rgb.shape, dtype=np.float32)
    for band in range(rgb.shape[-1]):
        low, high = np.percentile(rgb[..., band], (2, 98))
        out[..., band] = np.clip((rgb[..., band] - low) / max(high - low, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def pca_rgb(codes: np.ndarray) -> np.ndarray:
    q = codes.astype(np.float32)
    x = np.sign(q) * (np.abs(q) / 127.5) ** 2
    flat = x.reshape(64, -1).T
    flat = flat - flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat[:: max(1, flat.shape[0] // 20000)], full_matrices=False)
    comp = (flat @ vt[:3].T).reshape(x.shape[1], x.shape[2], 3)
    return stretch(comp)


def row(sheet: str) -> np.ndarray:
    with rasterio.open(WHU / "optical" / f"{sheet}.tif") as src:
        rgb = src.read([1, 2, 3])[:, ::STEP, ::STEP].transpose(1, 2, 0)
    with rasterio.open(WHU / "lbl" / f"{sheet}.tif") as src:
        label = PALETTE[src.read(1)[::STEP, ::STEP] // 10]
    with rasterio.open(AEF / f"{sheet}.tif") as src:
        aef = pca_rgb(src.read()[:, :: STEP // 2, :: STEP // 2])
    h = min(rgb.shape[0], label.shape[0], aef.shape[0])
    w = min(rgb.shape[1], label.shape[1], aef.shape[1])
    gap = np.full((h, 6, 3), 255, np.uint8)
    return np.concatenate([stretch(rgb)[:h, :w], gap, label[:h, :w], gap, aef[:h, :w]], axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sheets", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sheet in args.sheets:
        panel = row(sheet)
        Image.fromarray(panel).save(args.out / f"{sheet}.png")
        rows.append(panel)
    width = max(r.shape[1] for r in rows)
    sheet_img = np.concatenate(
        [np.pad(r, ((0, 6), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    )
    Image.fromarray(sheet_img).save(args.out / "contact_sheet.png")
    print(args.out / "contact_sheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
