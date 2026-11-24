"""Boundary IoU for T13 sheet predictions (Cheng et al. 2021, adapted).

For one class, the boundary of a mask is its inner edge: mask pixels with a
neighbour outside the mask. The band ``d`` is 2% of the sheet diagonal, so the
score does not depend on the resolution of the sheet. Boundary IoU is the
overlap of the two bands restricted to the mask itself:

    G_band = dilate(boundary(G), d) & G
    P_band = dilate(boundary(P), d) & P
    IoU_b  = |G_band & P_band| / |G_band | P_band|

Dilation uses a square neighbourhood (Chebyshev distance <= d), not a Euclidean
disc. That is cheap on full sheets, and it is slightly more generous than the
disc used in the paper. Pixels with label 0 are ignored, as in the mIoU.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

BAND_FRACTION = 0.02


def band_width(height: int, width: int) -> int:
    """Pixels: 2% of the sheet diagonal, at least 1."""
    return max(1, round(BAND_FRACTION * math.hypot(height, width)))


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    x = mask.float()[None, None]
    y = F.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y[0, 0] > 0


def _inner_boundary(mask: torch.Tensor) -> torch.Tensor:
    """Mask pixels with at least one 8-connected neighbour outside the mask."""
    outside = F.pad((~mask).float()[None, None], (1, 1, 1, 1), value=1.0)
    near_outside = F.max_pool2d(outside, kernel_size=3, stride=1)[0, 0]
    return mask & (near_outside > 0)


def boundary_counts(
    label: np.ndarray, prediction: np.ndarray, classes: range | list[int]
) -> dict[int, tuple[int, int]]:
    """Per class: (intersection, union) of the boundary bands, over labelled pixels."""
    gt = torch.from_numpy(np.asarray(label).astype(np.int64))
    pr = torch.from_numpy(np.asarray(prediction).astype(np.int64))
    labelled = gt != 0
    d = band_width(*gt.shape)
    counts: dict[int, tuple[int, int]] = {}
    for c in classes:
        g = (gt == c) & labelled
        p = (pr == c) & labelled
        if not bool(g.any()) and not bool(p.any()):
            continue
        g_band = _dilate(_inner_boundary(g), d) & g
        p_band = _dilate(_inner_boundary(p), d) & p
        inter = int((g_band & p_band).sum())
        union = int((g_band | p_band).sum())
        counts[c] = (inter, union)
    return counts


def add_counts(total: dict[int, list[int]], new: dict[int, tuple[int, int]]) -> None:
    """Accumulate per-sheet counts into dataset totals (in place)."""
    for c, (inter, union) in new.items():
        slot = total.setdefault(c, [0, 0])
        slot[0] += inter
        slot[1] += union


def mean_boundary_iou(total: dict[int, list[int]]) -> float:
    """Mean over classes with a non-empty union; 0.0 when no class appears."""
    scores = [inter / union for inter, union in total.values() if union > 0]
    return float(sum(scores) / len(scores)) if scores else 0.0
