"""Chip sampling for the T8 Five-Billion-Pixels scenes, shared by the workers.

The GF-2 scenes are ~6900x7300 px, so a run cannot feed whole scenes. The
first T8 implementation read *one* window per scene: training saw the same 108
chips every epoch and the test score came from one centre chip per test scene
(~0.03% of the test pixels). This module defines the protocol instead:

* **train** - every epoch draws ``TRAIN_CHIPS_PER_SCENE`` chips per scene
  without replacement from the scene's chip grid, seeded by (run seed, epoch),
  so training covers the scenes over the epochs and stays reproducible;
* **validation** - a fixed draw of ``VALIDATION_CHIPS_PER_SCENE`` chips per
  scene (same draw for every epoch and model) keeps checkpoint selection cheap;
* **test** - the full non-overlapping chip grid of every test scene. The edge
  strip narrower than one chip (< 224 px) is not scored.

The grid, draws and reads are numpy-only; rasterio is imported when a chip is
read, so the control layer and its tests do not need it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHIP = 224
TRAIN_CHIPS_PER_SCENE = 32
VALIDATION_CHIPS_PER_SCENE = 64
VALIDATION_DRAW_SEED = 20261026
IGNORE_INDEX = -1


@dataclass(frozen=True)
class Chip:
    scene: int
    col: int
    row: int


def grid(width: int, height: int, chip: int = CHIP) -> list[tuple[int, int]]:
    """Top-left corners of the non-overlapping chips that fit in a scene."""
    return [
        (col * chip, row * chip) for row in range(height // chip) for col in range(width // chip)
    ]


def epoch_seed(seed: int, epoch: int) -> int:
    return seed * 1000 + epoch


def load_split(manifest_path: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["split"] == split:
                    rows.append(row)
    return rows


class Gf2Chips:
    """Map-style dataset over chips of one split (usable directly by a DataLoader).

    ``sizes`` (width, height per scene) can be passed in tests; otherwise the
    raster headers are read once. ``set_chips`` swaps the active chip list, which
    is how training re-draws chips each epoch.
    """

    def __init__(
        self,
        root: Path,
        rows: list[dict],
        *,
        chip: int = CHIP,
        sizes: list[tuple[int, int]] | None = None,
        bands: int = 4,
    ) -> None:
        self.root = Path(root)
        self.rows = rows
        self.chip = chip
        self.bands = bands
        self.sizes = sizes if sizes is not None else [self._size(row) for row in rows]
        self.chips: list[Chip] = []

    def _size(self, row: dict) -> tuple[int, int]:
        import rasterio

        with rasterio.open(self.root / row["image_paths"][0]) as source:
            return source.width, source.height

    def full_grid(self) -> list[Chip]:
        return [
            Chip(scene, col, row)
            for scene, (width, height) in enumerate(self.sizes)
            for col, row in grid(width, height, self.chip)
        ]

    def draw(self, per_scene: int, seed: int) -> list[Chip]:
        """``per_scene`` chips per scene without replacement (all if fewer exist)."""
        rng = np.random.default_rng(seed)
        chips: list[Chip] = []
        for scene, (width, height) in enumerate(self.sizes):
            corners = grid(width, height, self.chip)
            take = min(per_scene, len(corners))
            for index in sorted(rng.choice(len(corners), size=take, replace=False).tolist()):
                col, row = corners[index]
                chips.append(Chip(scene, col, row))
        return chips

    def set_chips(self, chips: list[Chip]) -> Gf2Chips:
        self.chips = list(chips)
        return self

    def group_of(self, chip: Chip) -> str:
        return self.rows[chip.scene]["group_id"]

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, index: int) -> dict:
        import rasterio
        from rasterio.windows import Window

        chip = self.chips[index]
        row = self.rows[chip.scene]
        window = Window(chip.col, chip.row, self.chip, self.chip)
        with rasterio.open(self.root / row["image_paths"][0]) as source:
            image = source.read(window=window).astype(np.float32)[: self.bands]
        with rasterio.open(self.root / row["label_path"]) as source:
            label = source.read(1, window=window).astype(np.int64)
        # 0 is "unlabeled" in the release; 1..24 are the land-cover classes.
        label = np.where(label <= 0, IGNORE_INDEX, label)
        return {"image": image, "label": label, "group": self.group_of(chip)}


def channel_statistics(dataset: Gf2Chips, chips: list[Chip]) -> tuple[np.ndarray, np.ndarray]:
    """Per-band mean/std over the given (training) chips."""
    total = np.zeros(dataset.bands, dtype=np.float64)
    total_sq = np.zeros(dataset.bands, dtype=np.float64)
    count = 0
    previous = dataset.chips
    dataset.set_chips(chips)
    try:
        for index in range(len(dataset)):
            image = dataset[index]["image"].reshape(dataset.bands, -1).astype(np.float64)
            total += image.sum(axis=1)
            total_sq += (image**2).sum(axis=1)
            count += image.shape[1]
    finally:
        dataset.set_chips(previous)
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean**2, 0.0))
    return mean.astype(np.float32), np.maximum(std, 1e-3).astype(np.float32)
