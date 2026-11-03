"""Declared input normalization for the RGB tracks (T9 iSAID, T10 FAIR1M).

Decision 2026-11-03: a frozen backbone sees the input distribution its
pretraining used. LIM (MiT-Huge / MixMIM) was pretrained with ImageNet RGB
statistics (``upstream/selfsup/main_pretrain_multi.py``:
``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``), so its T9/T10
runs use those constants converted to 0-255, fixed across seeds.

Workers without a declared normalization keep computing dataset statistics, but
the sampling seed is now a constant: the old ``request.seed`` made every seed of
a cell see a slightly different input distribution (T9 mean R 76.1/73.7/72.8,
T10 69.1/62.4/66.6), which contaminates a seed-variance comparison.

See ``docs/evaluation/输入归一化口径_2026-11-03.md``.
"""

from __future__ import annotations

import numpy as np

#: 0-255 equivalents of the ImageNet constants (0.485 * 255 = 123.675, ...).
IMAGENET_RGB_MEAN: tuple[float, float, float] = (123.675, 116.28, 103.53)
IMAGENET_RGB_STD: tuple[float, float, float] = (58.395, 57.12, 57.375)
IMAGENET_RGB_ID = "imagenet_rgb_0_255"

#: Dataset statistics are a fallback, not a per-seed choice.
DATASET_STATS_SEED = 0


def imagenet_rgb() -> tuple[np.ndarray, np.ndarray]:
    """ImageNet mean/std in the 0-255 space the tile readers return."""
    return (
        np.asarray(IMAGENET_RGB_MEAN, dtype=np.float32),
        np.asarray(IMAGENET_RGB_STD, dtype=np.float32),
    )


def normalization_seed() -> int:
    """Fixed seed for dataset-statistics sampling (never ``request.seed``)."""
    return DATASET_STATS_SEED
