"""T13 (WHU-OPT-SAR optical) data: tiles, labels, AEF windows, augmentation.

* image: the four uint8 bands (R, G, B, NIR) of the manifest window;
* label: ``value // 10`` (0 background, 1..7 classes); 0 is the ignore index;
* AEF: the sheet's warped int8 grid (``scripts/whu_aef_warp.py``) at
  ``AEF_FACTOR`` image pixels per cell, so a 512 px tile reads an exact
  256 x 256 window. The grid lives under ``data_root/derived/<AEF_DIR>``.

Geometric augmentation (h/v flips) is applied to image, label and AEF together
from one generator, so they cannot drift apart. The encoder is run live, so
augmentation is allowed (no feature cache).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

AEF_DIR = "whu-aef2019"
AEF_FACTOR = 2
AEF_YEAR = 2019
IGNORE_INDEX = 0
NUM_CLASSES = 8
CLASS_NAMES = ("background", "farmland", "city", "village", "water", "forest", "road", "other")


def load_rows(manifest: Path, split: str) -> list[dict]:
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["split"] == split:
                rows.append(row)
    return rows


def _window(row: dict) -> Window:
    x, y, w, h = (int(v) for v in row["window"])
    return Window(x, y, w, h)


def read_image(data_root: Path, row: dict) -> np.ndarray:
    """``[4, H, W]`` uint8 (R, G, B, NIR)."""
    with rasterio.open(data_root / row["image_paths"][0]) as src:
        return src.read(window=_window(row))


def read_label(data_root: Path, row: dict) -> np.ndarray:
    """``[H, W]`` int64 class ids 0..7 (raw values are 0, 10, ..., 70)."""
    with rasterio.open(data_root / row["label_path"]) as src:
        raw = src.read(1, window=_window(row))
    label = (raw // 10).astype(np.int64)
    if label.max(initial=0) >= NUM_CLASSES or np.any(raw % 10):
        raise ValueError(f"{row['sample_id']}: label values outside {{0, 10, ..., 70}}")
    return label


def aef_path(data_root: Path, row: dict) -> Path:
    return data_root / "derived" / AEF_DIR / f"{row['group_id']}.tif"


def read_aef(data_root: Path, row: dict) -> np.ndarray:
    """``[64, H/2, W/2]`` int8 AEF codes for the tile (``-128`` = NoData)."""
    x, y, w, h = (int(v) for v in row["window"])
    if x % AEF_FACTOR or y % AEF_FACTOR or w % AEF_FACTOR or h % AEF_FACTOR:
        raise ValueError(f"{row['sample_id']}: window {row['window']} not on the AEF grid")
    window = Window(x // AEF_FACTOR, y // AEF_FACTOR, w // AEF_FACTOR, h // AEF_FACTOR)
    with rasterio.open(aef_path(data_root, row)) as src:
        if src.count != 64:
            raise ValueError(f"{src.name}: expected 64 AEF bands, got {src.count}")
        return src.read(window=window)


def band_statistics(
    rows: list[dict], data_root: Path, samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean/std over a fixed sample of *train* tiles (labelled pixels only)."""
    if not rows:
        raise ValueError("no rows to compute band statistics")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(rows), size=min(samples, len(rows)), replace=False)
    total = np.zeros(4)
    total_sq = np.zeros(4)
    count = 0
    for index in sorted(int(i) for i in picks):
        image = read_image(data_root, rows[index]).astype(np.float64)
        keep = read_label(data_root, rows[index]) != IGNORE_INDEX
        values = image[:, keep]
        total += values.sum(axis=1)
        total_sq += (values**2).sum(axis=1)
        count += values.shape[1]
    if not count:
        raise ValueError("sampled train tiles have no labelled pixels")
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


class T13Tiles(Dataset):
    """Training/eval tiles; ``augment`` draws flips from a per-item generator."""

    def __init__(
        self,
        rows: list[dict],
        data_root: Path,
        *,
        with_aef: bool,
        augment: bool,
        seed: int = 0,
    ):
        self.rows = rows
        self.data_root = data_root
        self.with_aef = with_aef
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image = read_image(self.data_root, row)
        label = read_label(self.data_root, row)
        aef = read_aef(self.data_root, row) if self.with_aef else None
        if self.augment:
            image, label, aef = flip(image, label, aef, self._flips(index))
        item = {
            "image": torch.from_numpy(np.array(image, copy=True, order="C")),
            "label": torch.from_numpy(np.array(label, copy=True, order="C")),
            "index": index,
        }
        if aef is not None:
            item["aef"] = torch.from_numpy(np.array(aef, copy=True, order="C"))
        return item

    def _flips(self, index: int) -> tuple[bool, bool]:
        gen = np.random.default_rng((self.seed, self.epoch, index))
        return bool(gen.integers(2)), bool(gen.integers(2))


def flip(
    image: np.ndarray, label: np.ndarray, aef: np.ndarray | None, flips: tuple[bool, bool]
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Apply the same horizontal/vertical flip to image, label and AEF."""
    horizontal, vertical = flips
    if horizontal:
        image, label = image[..., ::-1], label[..., ::-1]
        aef = None if aef is None else aef[..., ::-1]
    if vertical:
        image, label = image[..., ::-1, :], label[..., ::-1, :]
        aef = None if aef is None else aef[..., ::-1, :]
    return image, label, aef
