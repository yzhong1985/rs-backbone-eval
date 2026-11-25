"""Boundary IoU for T13 sheet predictions (plan v3 §4, adapted from Cheng et al. 2021).

For one class, the boundary of a mask is its inner edge: mask pixels with an
8-connected labelled neighbour outside the mask. Unlabelled neighbours do not count.
The band is the 2-pixel inner strip next to that edge, measured with a square
neighbourhood (Chebyshev distance <= 2) and kept inside the mask:

    G_band = dilate(boundary(G), 2) & G & scored
    P_band = dilate(boundary(P), 2) & P & scored
    IoU_b  = |G_band & P_band| / |G_band | P_band|

``scored`` is the region where both maps are judged: labelled pixels that are more
than 2 px from any unlabelled pixel and more than 2 px from the sheet edge. Pixels
next to unlabelled areas and the sheet edge are therefore never scored, and the
sheet edge never counts as a boundary. Label 0 is unlabelled, as in the mIoU.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

BAND_PX = 2


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Pixels with any mask pixel within ``radius`` (square neighbourhood).

    A summed-area table gives each window count in constant time, so the cost
    does not grow with the radius. A max-pool with a 2r+1 kernel costs about
    (2r+1)^2 per pixel, which is far too slow on full 5555 px sheets.
    """
    if radius <= 0:
        return mask
    height, width = mask.shape
    table = F.pad(mask.to(torch.int32).cumsum(0).cumsum(1), (1, 0, 1, 0))  # (H+1, W+1)
    rows = torch.arange(height)
    cols = torch.arange(width)
    r0 = (rows - radius).clamp_min(0)
    r1 = (rows + radius + 1).clamp_max(height)
    c0 = (cols - radius).clamp_min(0)
    c1 = (cols + radius + 1).clamp_max(width)
    counts = table[r1][:, c1] - table[r0][:, c1] - table[r1][:, c0] + table[r0][:, c0]
    return counts > 0


def _inner_boundary(mask: torch.Tensor, labelled: torch.Tensor) -> torch.Tensor:
    """Mask pixels with an 8-connected neighbour that is labelled and outside the mask.

    Unlabelled neighbours do not make a boundary: the true class there is unknown.
    The padding is 0 ("not outside"), so pixels on the sheet edge are not boundary.
    """
    outside = F.pad((labelled & ~mask).float()[None, None], (1, 1, 1, 1), value=0.0)
    near_outside = F.max_pool2d(outside, kernel_size=3, stride=1)[0, 0]
    return mask & (near_outside > 0)


def _away_from_edge(height: int, width: int, radius: int) -> torch.Tensor:
    """True where a pixel is at least ``radius`` px from every sheet edge."""
    rows = torch.arange(height)[:, None]
    cols = torch.arange(width)[None, :]
    return (rows >= radius) & (rows < height - radius) & (cols >= radius) & (cols < width - radius)


def boundary_counts(
    label: np.ndarray, prediction: np.ndarray, classes: range | list[int]
) -> dict[int, tuple[int, int]]:
    """Per class: (intersection, union) of the boundary bands over the scored region."""
    gt = torch.from_numpy(np.asarray(label).astype(np.int64))
    pr = torch.from_numpy(np.asarray(prediction).astype(np.int64))
    labelled = gt != 0
    scored = labelled & ~_dilate(~labelled, BAND_PX) & _away_from_edge(*gt.shape, BAND_PX)
    counts: dict[int, tuple[int, int]] = {}
    for c in classes:
        g = gt == c
        p = (pr == c) & labelled
        if not bool(g.any()) and not bool(p.any()):
            continue
        g_band = _dilate(_inner_boundary(g, labelled), BAND_PX) & g & scored
        p_band = _dilate(_inner_boundary(p, labelled), BAND_PX) & p & scored
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
