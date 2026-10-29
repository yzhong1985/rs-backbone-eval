"""Tile dataset and manifest geometry for the iSAID workers (torch).

Every backbone that evaluates T9 reads the same manifest rows the same way: one
window per row, label padded with ``255``, optional flips for training. Sharing
the dataset keeps the LIM and Clay cells on identical inputs -- the only
difference between them must be the encoder.

Reads go through ``data.isaid_io`` so band counts are handled once. This module
imports torch, so only workers may import it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .isaid_io import read_label, read_rgb
from .tiling import pad_tile

IGNORE_INDEX = 255


def load_rows(manifest_path: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["split"] == split:
                    rows.append(row)
    return rows


def tile_geometry(rows: list[dict]) -> tuple[int, int]:
    """Infer ``(crop, stride)`` from the manifest so a run matches its build."""
    crop = int(rows[0]["window"][2])
    origins: dict[str, set[int]] = {}
    for row in rows:
        origins.setdefault(row["group_id"], set()).add(int(row["window"][0]))
    steps = [
        sorted(b - a for a, b in zip(sorted(axis), sorted(axis)[1:], strict=False) if b > a)
        for axis in origins.values()
        if len(axis) > 1
    ]
    strides = [step[0] for step in steps if step]
    stride = min(strides) if strides else crop // 2
    return crop, stride


class IsaidTiles(Dataset):
    """Training tiles: read the manifest window, pad the label with ignore."""

    def __init__(
        self,
        rows: list[dict],
        data_root: Path,
        crop: int,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        augment: bool = True,
    ):
        self.rows = rows
        self.data_root = data_root
        self.crop = crop
        self.mean = mean.reshape(1, 1, 3)
        self.std = std.reshape(1, 1, 3)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        window = tuple(int(value) for value in row["window"])
        image = read_rgb(self.data_root / row["image_paths"][0], window).astype(np.float32)
        label = read_label(self.data_root / row["label_path"], window)
        if image.shape[:2] != label.shape:
            raise ValueError(
                f"{row['sample_id']}: image {image.shape[:2]} != label {label.shape} "
                "(the mask and the DOTA image disagree)"
            )
        if self.augment and np.random.random() < 0.5:
            image = image[:, ::-1].copy()
            label = label[:, ::-1].copy()
        if self.augment and np.random.random() < 0.5:
            image = image[::-1].copy()
            label = label[::-1].copy()
        image = pad_tile((image - self.mean) / self.std, self.crop, fill=0.0)
        label = pad_tile(label, self.crop, fill=IGNORE_INDEX)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "label": torch.from_numpy(label.astype(np.int64)),
            "group": row["group_id"],
        }


def norm_statistics(
    rows: list[dict], data_root: Path, sample: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over a random draw of training tiles."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(rows), size=min(sample, len(rows)), replace=False)
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for index in picks:
        row = rows[int(index)]
        window = tuple(int(value) for value in row["window"])
        image = read_rgb(data_root / row["image_paths"][0], window).astype(np.float64)
        if image.size == 0:
            continue
        flat = image.reshape(-1, image.shape[-1])
        total += flat.sum(axis=0)
        total_sq += (flat**2).sum(axis=0)
        count += image.shape[0] * image.shape[1]
    if not count:
        raise ValueError("no training pixels to compute normalisation statistics")
    mean = total / count
    variance = np.maximum(total_sq / count - mean**2, 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)
