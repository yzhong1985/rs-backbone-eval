#!/usr/bin/env python3
"""Visual alignment check for the T11D four-band student (P0 deliverable).

For four patches (two train, one validation, one test) the panel shows:

1. S2 false colour (B08/B04/B03) from the exact 2019 frames the student reads;
2. the same image with the 10 m labels blended in;
3. AEF 2019 -> PCA RGB;
4. OlmoEarth v1.3 2019 (128 dims) -> PCA RGB;
5. S2 false colour blended 55/45 with the OlmoEarth PCA, where a half-pixel
   offset would show as coloured fringes on field boundaries.

Runs in the Clay worker environment (matplotlib), CPU only.

Usage::

    workers/clay/.venv/bin/python scripts/pastis_rgbn_align_qc.py \\
        --out /home/yzhong/rs-eval/audits/pastis-rgbn/align_qc.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATA_ROOT = Path("/geodata/rs-eval/data")
MANIFEST_DIR = DATA_ROOT / "manifests" / "T11D"
S2_INDICES = [0, 1, 2, 6]


def rows() -> list[dict]:
    manifest = sorted(MANIFEST_DIR.glob("*.jsonl"), key=lambda path: path.stat().st_mtime)[-1]
    return [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]


def pick(rows_in: list[dict]) -> list[dict]:
    by_split = {split: [] for split in ("train", "validation", "test")}
    for row in rows_in:
        by_split[row["split"]].append(row)
    train = sorted(by_split["train"], key=lambda row: row["sample_id"])
    tiles = sorted({row["aux_paths"]["tile"] for row in train})
    chosen = [train[0]]
    for tile in tiles[1:]:
        candidate = next((row for row in train if row["aux_paths"]["tile"] == tile), None)
        if candidate is not None:
            chosen.append(candidate)
        if len(chosen) == 2:
            break
    chosen += [by_split["validation"][0], by_split["test"][0]]
    return chosen


def false_colour(pixels: np.ndarray) -> np.ndarray:
    """(T,4,128,128) raw DN -> (128,128,3) in [0,1], time-median, robust stretch."""
    cube = np.median(pixels, axis=0)  # (4,128,128)
    rgb = np.stack([cube[3], cube[2], cube[1]], axis=-1)  # NIR, red, green
    low, high = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - low) / max(high - low, 1e-6), 0, 1)


def pca_rgb(path: Path) -> np.ndarray:
    archive = np.load(path)
    features = archive["features"].astype(np.float32)
    valid = archive["valid"] if "valid" in archive.files else np.ones(features.shape[1:], bool)
    features = np.where(valid, np.sign(features) * (np.abs(features) / 127.5) ** 2, 0.0)
    flat = features.reshape(features.shape[0], -1).T
    flat = flat - flat.mean(axis=0)
    # Two leading components as R/G; the third is left at zero.
    _, _, vectors = np.linalg.svd(flat, full_matrices=False)
    projected = flat @ vectors[:3].T
    projected = projected.reshape(features.shape[1], features.shape[2], 3)
    low, high = np.percentile(projected, 2), np.percentile(projected, 98)
    return np.clip((projected - low) / np.maximum(high - low, 1e-6), 0, 1)


def label_rgb(labels: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0.20, 0.20, 0.20],
            [0.90, 0.80, 0.20],
            [0.85, 0.55, 0.15],
            [0.55, 0.75, 0.25],
            [0.25, 0.60, 0.30],
            [0.20, 0.45, 0.70],
            [0.55, 0.35, 0.70],
            [0.85, 0.35, 0.45],
            [0.60, 0.60, 0.60],
            [0.95, 0.65, 0.55],
            [0.35, 0.75, 0.75],
            [0.75, 0.45, 0.25],
            [0.45, 0.45, 0.85],
            [0.70, 0.70, 0.30],
            [0.30, 0.70, 0.50],
            [0.65, 0.30, 0.55],
            [0.90, 0.90, 0.40],
            [0.40, 0.40, 0.40],
            [0.10, 0.50, 0.90],
            [0.95, 0.95, 0.95],
        ]
    )
    return palette[np.clip(labels, 0, len(palette) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = pick(rows())
    figure, axes = plt.subplots(len(chosen), 5, figsize=(20, 4.1 * len(chosen)))
    for index, row in enumerate(chosen):
        patch = row["sample_id"]
        pixels = np.load(DATA_ROOT / row["aux_paths"]["s2_path"], mmap_mode="r")
        dates = json.loads((DATA_ROOT / row["aux_paths"]["dates_source"]).read_text())["dates_s2"]
        frames = [i for i, date in enumerate(dates) if str(date).startswith("2019")]
        if len(frames) > 20:
            picks = np.linspace(0, len(frames) - 1, 20)
            frames = sorted({frames[int(round(pick))] for pick in picks})
        cube = np.asarray(pixels[frames][:, S2_INDICES], dtype=np.float32)
        labels = np.load(DATA_ROOT / row["label_path"])[0]
        aef = pca_rgb(DATA_ROOT / "derived" / "aef-2019" / "features" / "2019" / f"{patch}.npz")
        olmo = pca_rgb(
            DATA_ROOT / "derived" / "olmoearth-2019-128" / "features" / "2019" / f"{patch}.npz"
        )
        image = false_colour(cube)
        panels = [
            (image, "S2 NIR/R/G (2019, time median)"),
            (0.5 * image + 0.5 * label_rgb(labels), "S2 + 10 m labels"),
            (aef, "AEF 2019 PCA"),
            (olmo, "OlmoEarth v1.3 2019 PCA (128)"),
            (0.55 * image + 0.45 * olmo, "S2 + OlmoEarth overlay"),
        ]
        for column, (panel, title) in enumerate(panels):
            axis = axes[index, column]
            axis.imshow(panel)
            axis.set_xticks([])
            axis.set_yticks([])
            if index == 0:
                axis.set_title(title, fontsize=10)
        axes[index, 0].set_ylabel(
            f"{patch}\n{row['split']} / {row['aux_paths']['tile']}", fontsize=9
        )
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150)
    print(f"-> {args.out} ({len(chosen)} patches)")


if __name__ == "__main__":
    main()
