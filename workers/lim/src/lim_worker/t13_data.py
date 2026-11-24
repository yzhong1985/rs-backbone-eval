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
    """T13 rows of ``split``; rows of other tasks in a shared manifest are skipped.

    Must filter on ``task_id`` as well as ``split``: ``require_no_group_leakage``
    checks T13 rows only, so a foreign row read here would bypass that check.
    """
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_id") == "T13" and row.get("split") == split:
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


def donor_rows(rows: list[dict]) -> dict[str, dict]:
    """``{sample_id: row whose AEF to use}`` for the shuffled-teacher control.

    Every tile takes the AEF of the tile at the same position in the sheet that
    sits half the sheet list away (sorted by name), so the donor is always a
    different sheet, never an overlapping or adjacent tile of the same sheet,
    and the mapping is fixed for training and scoring alike. Shuffling inside a
    batch (the first version) swapped neighbouring tiles of one sheet during
    scoring, which still carries the local teacher signal (review 2026-11-18).
    """
    by_sheet: dict[str, list[dict]] = {}
    for row in rows:
        by_sheet.setdefault(row["group_id"], []).append(row)
    sheets = sorted(by_sheet)
    if len(sheets) < 2:
        raise ValueError("the shuffled-teacher control needs at least two sheets")
    for tiles in by_sheet.values():
        tiles.sort(key=lambda r: (int(r["window"][1]), int(r["window"][0])))
    shift = len(sheets) // 2
    out: dict[str, dict] = {}
    for index, sheet in enumerate(sheets):
        donor = by_sheet[sheets[(index + shift) % len(sheets)]]
        for position, row in enumerate(by_sheet[sheet]):
            out[row["sample_id"]] = donor[position % len(donor)]
    return out


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
        aef_source: dict[str, dict] | None = None,
        sar_entries: dict | None = None,
        sar_stats: tuple[float, float] | None = None,
    ):
        self.rows = rows
        self.data_root = data_root
        self.with_aef = with_aef
        self.augment = augment
        self.seed = seed
        self.aef_source = aef_source
        # SAR is optional. When on, every item carries a normalised SAR plane and a
        # float validity plane; ``sar_stats`` must come from the train split only.
        if (sar_entries is None) != (sar_stats is None):
            raise ValueError("sar_entries and sar_stats must be given together")
        self.sar_entries = sar_entries
        self.sar_stats = sar_stats
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        # Workers copy the dataset when an epoch's iterator starts, so the
        # loader must not use persistent workers (they would keep epoch 0).
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        image = read_image(self.data_root, row)
        label = read_label(self.data_root, row)
        aef = None
        if self.with_aef:
            source = self.aef_source[row["sample_id"]] if self.aef_source else row
            aef = read_aef(self.data_root, source)
        flips = self._flips(index) if self.augment else (False, False)
        if self.augment:
            image, label, aef = flip(image, label, aef, flips)
        item = {
            "image": torch.from_numpy(np.array(image, copy=True, order="C")),
            "label": torch.from_numpy(np.array(label, copy=True, order="C")),
            "index": index,
        }
        if aef is not None:
            item["aef"] = torch.from_numpy(np.array(aef, copy=True, order="C"))
        if self.sar_entries is not None:
            sar_plane = read_sar(self.data_root, row, self.sar_entries[row["group_id"]])
            mean, std = self.sar_stats
            sar = (sar_plane.astype(np.float32) - mean) / std
            valid = np.ones_like(sar, dtype=np.float32)
            sar = flip_plane(sar, flips)
            valid = flip_plane(valid, flips)
            item["sar"] = torch.from_numpy(np.array(sar, copy=True, order="C"))
            item["sar_valid"] = torch.from_numpy(np.array(valid, copy=True, order="C"))
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


def load_sar_entries(data_root: Path, index_sha256: str, rows: list[dict], *, formal: bool) -> dict:
    """Pinned SAR sheets for every group in ``rows``.

    ``formal`` turns on full-file SHA-256 checks. Smoke and diagnostic runs skip
    them (about 2 GB of reads), formal runs must not.
    """
    from .sar_index import load_sar_index

    groups = {row["group_id"] for row in rows}
    return load_sar_index(data_root, index_sha256, groups, verify_files=formal)


def flip_plane(plane: np.ndarray, flips: tuple[bool, bool]) -> np.ndarray:
    """Apply the same flips as ``flip`` to one ``[..., H, W]`` plane (SAR, validity)."""
    horizontal, vertical = flips
    if horizontal:
        plane = plane[..., ::-1]
    if vertical:
        plane = plane[..., ::-1, :]
    return plane


def read_sar(data_root: Path, row: dict, entry) -> np.ndarray:
    """``[1, H, W]`` uint8 SAR window for ``row``, read from the pinned sheet ``entry``.

    The window is the optical one: the sheets share a pixel grid (registration
    is checked by eye, see the plan §2), so the same offsets address the same
    ground. A row whose group is not the entry's sheet is refused.
    """
    if row["group_id"] != entry.group_id:
        raise ValueError(f"{row['sample_id']}: SAR entry is for {entry.group_id}")
    with rasterio.open(data_root / entry.path) as src:
        if src.count != 1:
            raise ValueError(f"{src.name}: expected one SAR band, got {src.count}")
        return src.read(window=_window(row))


def sar_statistics(
    rows: list[dict], data_root: Path, entries: dict, samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std of SAR over labelled pixels of a fixed sample of *train* tiles."""
    if not rows:
        raise ValueError("no rows to compute SAR statistics")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(rows), size=min(samples, len(rows)), replace=False)
    total = 0.0
    total_sq = 0.0
    count = 0
    for index in sorted(int(i) for i in picks):
        row = rows[index]
        sar = read_sar(data_root, row, entries[row["group_id"]]).astype(np.float64)[0]
        keep = read_label(data_root, row) != IGNORE_INDEX
        values = sar[keep]
        total += float(values.sum())
        total_sq += float((values**2).sum())
        count += int(values.size)
    if not count:
        raise ValueError("sampled train tiles have no labelled pixels")
    mean = total / count
    std = max(total_sq / count - mean**2, 1e-6) ** 0.5
    return np.float32(mean), np.float32(std)
