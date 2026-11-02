"""Training tiles must pad the label as ignore, never as background.

``read_rgb`` reads a tile window with ``boundless=True``, which filled the
out-of-raster band with zero; the mask went through the same path, so padding
became class 0 and the later ``pad_tile(..., ignore_index)`` was a no-op. On the
T9 manifest about 32% of training tiles touch the raster edge (review
2026-11-02).

Needs rasterio, which the control-plane environment does not install; run it in
a worker environment, e.g.

    workers/prithvi/.venv/bin/python -m pytest tests/test_isaid_tiles.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers" / "lim" / "src"))

pytest.importorskip("rasterio")

from rs_backbone_eval.data.isaid_io import PALETTE, read_tile  # noqa: E402

IGNORE = 255
SCALE = {"mean": np.zeros(3, dtype=np.float32), "std": np.ones(3, dtype=np.float32)}


def write_png(path, rgb: np.ndarray) -> None:
    import rasterio

    height, width = rgb.shape[:2]
    with rasterio.open(
        path, "w", driver="PNG", width=width, height=height, count=3, dtype="uint8"
    ) as destination:
        destination.write(np.transpose(rgb, (2, 0, 1)))


def foreground_colour() -> tuple[int, int, int]:
    return next(colour for colour, klass in PALETTE.items() if klass == 1)


def make_tile(tmp_path, size: int = 8, objects: int = 2):
    """Image with a distinct value per pixel, mask with an ``objects``-square blob."""
    colour = foreground_colour()
    ramp = np.arange(size * size, dtype=np.uint8).reshape(size, size, 1)
    image = np.repeat(ramp, 3, axis=2) + 10
    mask = np.zeros((size, size, 3), dtype=np.uint8)
    mask[:objects, :objects] = colour
    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    write_png(image_path, image)
    write_png(mask_path, mask)
    return image_path, mask_path


def row_for(image_path, mask_path, window, tmp_path) -> dict:
    return {
        "sample_id": "edge",
        "image_paths": [str(image_path.relative_to(tmp_path))],
        "label_path": str(mask_path.relative_to(tmp_path)),
        "window": window,
        "group_id": "0",
    }


def dataset_of(tmp_path, row, augment: bool):
    from rs_backbone_eval.data.isaid_tiles import IsaidTiles

    return IsaidTiles(
        [row], tmp_path, crop=8, mean=SCALE["mean"], std=SCALE["std"], augment=augment
    )


def test_partial_window_returns_only_the_valid_block(tmp_path) -> None:
    image_path, mask_path = make_tile(tmp_path)
    image, label = read_tile(image_path, mask_path, (4, 4, 8, 8))
    assert image.shape == (4, 4, 3)
    assert label.shape == (4, 4)
    assert int(label.min()) >= 0


def test_shared_dataset_pads_the_label_with_ignore(tmp_path) -> None:
    image_path, mask_path = make_tile(tmp_path)
    row = row_for(image_path, mask_path, [4, 4, 8, 8], tmp_path)  # only 4x4 inside
    label = dataset_of(tmp_path, row, augment=False)[0]["label"].numpy()
    assert label.shape == (8, 8)
    assert int((label == IGNORE).sum()) == 64 - 16
    assert int((label == 0).sum()) == 16  # the valid block is background
    assert int((label[4:, :] == 0).sum()) + int((label[:, 4:] == 0).sum()) == 0


def test_lim_dataset_pads_the_label_with_ignore(tmp_path) -> None:
    from lim_worker.t9 import IsaidTiles

    image_path, mask_path = make_tile(tmp_path)
    row = row_for(image_path, mask_path, [4, 4, 8, 8], tmp_path)
    dataset = IsaidTiles(
        [row], tmp_path, crop=8, mean=SCALE["mean"], std=SCALE["std"], augment=False
    )
    label = dataset[0]["label"].numpy()
    assert int((label == IGNORE).sum()) == 48
    assert int((label == 0).sum()) == 16


def test_augmented_flip_moves_padding_and_image_together(tmp_path) -> None:
    image_path, mask_path = make_tile(tmp_path)
    row = row_for(image_path, mask_path, [4, 4, 8, 8], tmp_path)
    dataset = dataset_of(tmp_path, row, augment=True)
    for seed in range(8):
        np.random.seed(seed)
        item = dataset[0]
        label = item["label"].numpy()
        image = item["image"].numpy()
        assert int((label == IGNORE).sum()) == 48
        # the padded band is zero in the image exactly where the label is ignore:
        # flipping before padding keeps the two aligned
        assert float(np.abs(image[:, label == IGNORE]).max()) == 0.0
        assert float(np.abs(image[:, label != IGNORE]).max()) > 0.0


def test_norm_statistics_ignores_the_padding(tmp_path) -> None:
    from rs_backbone_eval.data.isaid_tiles import norm_statistics

    image_path, mask_path = make_tile(tmp_path)
    row = row_for(image_path, mask_path, [4, 4, 8, 8], tmp_path)
    mean, std = norm_statistics([row], tmp_path, sample=1, seed=0)
    valid = np.arange(64, dtype=np.float64).reshape(8, 8) + 10
    expected = valid[4:, 4:].reshape(-1).mean()
    assert float(mean[0]) == pytest.approx(expected)
    # a zero-filled 8x8 read would have pulled the mean down towards 0
    assert float(mean[0]) > 20
