#!/usr/bin/env python3
"""Per-patch visual QC for the OlmoEarth v1.3 / PASTIS 10 m alignment.

Reads the extracted ``.npz`` features and renders, for every QC patch:

    S2 RGB | SPOT 1 m | OlmoEarth PCA | S2 + parcel labels

The 10 m grid is the one stored in the AEF sidecar (SPOT affine, 5+10k); the
OlmoEarth window was extracted on that grid by
``scripts/olmoearth_extract_pastis.py``. Run in the LIM worker environment:

    PYTHONPATH=src:workers/lim/src workers/lim/.venv/bin/python \
        scripts/olmoearth_align_qc.py

Writes PNGs and ``olmoearth_alignment_batch.csv`` under
``/home/yzhong/rs-eval/audits/olmoearth-20261106``.
"""

from __future__ import annotations

import csv
import glob
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw

FEATURES = Path("/home/yzhong/rs-eval/data-olmoearth/features/2019")
PASTIS = Path("/home/yzhong/rs-eval/data-aef/PASTIS-HD")
AUDIT = Path("/home/yzhong/rs-eval/audits/olmoearth-20261106")
SIZE = 160
PICKS = [
    10000,
    10107,
    10213,
    10323,
    10432,
    10540,
    20000,
    20128,
    20256,
    20387,
    20522,
    20657,
    30000,
    30146,
    30292,
    30440,
    30589,
    30737,
    40000,
    40111,
    40223,
    40337,
    40451,
    40562,
]
# A compact 20-class palette; class 19 is void.
COLORS = np.array(
    [
        [0, 0, 0],
        [0, 80, 80],
        [0, 160, 160],
        [0, 0, 160],
        [0, 160, 255],
        [0, 0, 80],
        [0, 0, 255],
        [0, 120, 190],
        [0, 80, 160],
        [0, 80, 255],
        [0, 160, 240],
        [0, 80, 0],
        [0, 160, 80],
        [0, 80, 200],
        [0, 220, 160],
        [0, 0, 220],
        [180, 80, 0],
        [180, 160, 0],
        [120, 0, 180],
        [40, 40, 40],
    ],
    dtype=np.uint8,
)


def dequantize(q: np.ndarray) -> np.ndarray:
    v = q.astype(np.float32)
    v[q == -128] = np.nan
    return (np.abs(v) / 127.5) ** 2 * np.sign(v)


def stretch(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for band in range(rgb.shape[-1]):
        values = rgb[..., band]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        low, high = np.percentile(values[finite], [2, 98])
        out[..., band] = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
    return (out * 255).astype(np.uint8)


def pca_rgb(features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    flat = features.reshape(features.shape[0], -1).T
    finite = np.isfinite(flat).all(axis=1) & valid.reshape(-1)
    out = np.zeros((features.shape[1], features.shape[2], 3), dtype=np.float32)
    if finite.sum() < 32:
        return np.zeros_like(out, dtype=np.uint8)
    x = flat[finite]
    x = x - x.mean(axis=0, keepdims=True)
    # 64 columns; compute the small-side SVD directly.
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    scores = x @ vt[:3].T
    for index in range(3):
        values = scores[:, index]
        low, high = np.percentile(values, [2, 98])
        out[..., index][finite.reshape(128, 128)] = np.clip(
            (values - low) / max(high - low, 1e-6), 0, 1
        )
    return (out * 255).astype(np.uint8)


def s2_rgb(patch: int, dates: list[int]) -> tuple[np.ndarray, np.ndarray, int]:
    stack = np.load(PASTIS / f"DATA_S2/S2_{patch}.npy", mmap_mode="r")
    index = min(range(stack.shape[0]), key=lambda i: abs(int(dates[i]) - 20190715))
    frame = np.asarray(stack[index], dtype=np.float32)
    # PASTIS band order: B02, B03, B04, ...; display B04/B03/B02.
    rgb = np.transpose(frame[[2, 1, 0]], (1, 2, 0)) / 10000.0
    ndvi = (frame[6] - frame[2]) / np.maximum(frame[6] + frame[2], 1)
    return stretch(rgb), ndvi, int(dates[index])


def spot_rgb(patch: int) -> np.ndarray:
    paths = glob.glob(str(PASTIS / f"DATA_SPOT/PASTIS_SPOT6_RVB_1M00_2019/*_{patch}.tif"))
    if not paths:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    with rasterio.open(paths[0]) as dataset:
        image = dataset.read()
    image = np.transpose(image[:3], (1, 2, 0)).astype(np.float32)
    return stretch(image)


def overlay(rgb: np.ndarray, labels: np.ndarray) -> np.ndarray:
    colors = COLORS[np.clip(labels, 0, 19)]
    mask = labels != 19
    out = rgb.astype(np.float32).copy()
    out[mask] = 0.55 * out[mask] + 0.45 * colors[mask].astype(np.float32)
    return out.clip(0, 255).astype(np.uint8)


def panel(image: np.ndarray, title: str) -> Image.Image:
    base = Image.fromarray(image).resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (SIZE, SIZE + 22), "white")
    canvas.paste(base, (0, 22))
    ImageDraw.Draw(canvas).text((4, 5), title, fill="black")
    return canvas


def corr_at_shift(pc: np.ndarray, valid: np.ndarray, ndvi: np.ndarray, dy: int, dx: int):
    height, width = valid.shape
    tgt = ndvi[max(0, dy) : height + min(0, dy), max(0, dx) : width + min(0, dx)]
    src = pc[max(0, -dy) : height - max(0, dy), max(0, -dx) : width - max(0, dx)]
    src_valid = valid[max(0, -dy) : height - max(0, dy), max(0, -dx) : width - max(0, dx)]
    both = src_valid & np.isfinite(tgt)
    if both.sum() < 32 or float(src[both].std()) < 1e-9 or float(tgt[both].std()) < 1e-9:
        return None
    result = float(np.corrcoef(src[both], tgt[both])[0, 1])
    return result if math.isfinite(result) else None


def main() -> int:
    AUDIT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    panels: list[list[Image.Image]] = []
    for patch in PICKS:
        npz = FEATURES / f"{patch}.npz"
        sidecar = FEATURES / f"{patch}.json"
        if not npz.is_file() or not sidecar.is_file():
            print(f"{patch}: missing extraction; skipped", flush=True)
            continue
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        with np.load(npz) as archive:
            features = dequantize(archive["features"])
            valid = archive["valid"]
        labels = np.load(PASTIS / f"ANNOTATIONS/TARGET_{patch}.npy")[0].astype(np.int64)
        rgb, ndvi, used_date = s2_rgb(patch, meta.get("dates_s2", []))
        spot = spot_rgb(patch)
        pca = pca_rgb(features, valid)
        overlay_rgb = overlay(rgb, labels)

        pc1 = pca[..., 0].astype(np.float32) / 255.0
        correlations = {
            (dy, dx): corr_at_shift(pc1, valid, ndvi, dy, dx)
            for dy in range(-2, 3)
            for dx in range(-2, 3)
        }
        valid_corr = {k: v for k, v in correlations.items() if v is not None}
        best = max(valid_corr, key=lambda k: abs(valid_corr[k])) if valid_corr else (None, None)
        rows.append(
            {
                "patch": patch,
                "tile": meta.get("tile"),
                "fold": meta.get("fold"),
                "date": used_date,
                "valid": round(float(valid.mean()), 3),
                "r_ndvi": round(correlations[(0, 0)], 3)
                if correlations[(0, 0)] is not None
                else None,
                "best_dx": best[0],
                "best_dy": best[1],
                "best_r": round(valid_corr[best], 3) if valid_corr else None,
            }
        )
        row = [
            panel(rgb, f"{patch} S2 RGB"),
            panel(spot, "SPOT 1 m"),
            panel(pca, "OlmoEarth PCA"),
            panel(overlay_rgb, "S2 + labels"),
        ]
        panels.append(row)
        combined = Image.new("RGB", (SIZE * 4, SIZE + 22), "white")
        for index, item in enumerate(row):
            combined.paste(item, (index * SIZE, 0))
        combined.save(AUDIT / f"olmoearth_pastis_alignment_{patch}.png")
        print(f"{patch}: {row[0].width * 4}x{row[0].height} written", flush=True)

    if panels:
        sheet = Image.new("RGB", (SIZE * 4, (SIZE + 22) * len(panels)), "white")
        for index, row in enumerate(panels):
            for column, item in enumerate(row):
                sheet.paste(item, (column * SIZE, index * (SIZE + 22)))
        sheet.save(AUDIT / "olmoearth_pastis_alignment.png")
    with (AUDIT / "olmoearth_alignment_batch.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["patch"])
        writer.writeheader()
        writer.writerows(rows)
    (AUDIT / "README.md").write_text(
        "# OlmoEarth v1.3 / PASTIS 10 m alignment QC\n\n"
        "Per patch: S2 RGB, SPOT 1 m, OlmoEarth PCA, S2 + parcel labels.\n"
        "PCA is fitted on valid OlmoEarth pixels only; invalid = black.\n"
        "`olmoearth_pastis_alignment.png` is the stacked sheet; individual files "
        "are `olmoearth_pastis_alignment_<patch>.png`.\n"
        "`olmoearth_alignment_batch.csv` reports valid share, PC1-NDVI correlation "
        "at (0,0) and the best +/-2-cell shift (diagnostic only, never applied).\n",
        encoding="utf-8",
    )
    print(AUDIT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
