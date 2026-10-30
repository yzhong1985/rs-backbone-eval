"""Anchors and rotated-box encode/decode for Rotated Faster R-CNN.

The detector is the *fixed* variable of the FAIR1M comparison, so every constant
that changes what a proposal means lives here: anchor sizes per level, the
``(10, 10, 5, 5, 1)`` regression weights, and the le90 angle increment. The angle
delta is expressed in units of 90°, which keeps the regression target in
``[-1, 1]`` and makes the wrap-around at ±90° explicit instead of a discontinuity
Smooth L1 would fit forever (FAIR1M plan §6.1, §7.1).

As in the reference implementation, proposals are matched with **horizontal**
IoU: FAIR1M's boxes are rotated, but the RPN and the RoI assigner match on the
axis-aligned hull of the ground truth and the RoI head regresses the rotated box.
That is what keeps torchvision's ``box_iou``/``nms``/``roi_align`` sufficient and
removes the per-worker MMCV build (plan §7.1, G0).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# Anchors in pixels for a 1024 tile with 0.5 m GSD: 32 px is ~16 m (a car is
# ~9 px), 256 px is ~128 m. Three aspect ratios, as in the reference 1x config.
ANCHOR_SIZES: tuple[int, ...] = (32, 64, 128, 256)
ANCHOR_RATIOS: tuple[float, ...] = (0.5, 1.0, 2.0)
STRIDES: tuple[int, ...] = (4, 8, 16, 32)
REG_WEIGHTS: tuple[float, float, float, float, float] = (10.0, 10.0, 5.0, 5.0, 1.0)
HALF_PI = math.pi / 2


def anchors_for_level(
    height: int,
    width: int,
    stride: int,
    sizes: tuple[int, ...] = ANCHOR_SIZES,
    ratios: tuple[float, ...] = ANCHOR_RATIOS,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """``[H*W*A, 4]`` anchors in ``x1, y1, x2, y2`` at one pyramid level."""
    shift_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride
    shift_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(shift_y, shift_x, indexing="ij")
    centres = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    boxes = []
    for size in sizes:
        for ratio in ratios:
            half_w = 0.5 * size * math.sqrt(ratio)
            half_h = 0.5 * size / math.sqrt(ratio)
            offsets = torch.tensor([-half_w, -half_h, half_w, half_h], device=device, dtype=dtype)
            centre = centres.repeat(1, 2)
            boxes.append(centre + offsets)
    # [H*W*A, 4] with anchor index varying slowest per level, as torchvision
    # expects for the matching of one image
    return torch.cat(boxes, dim=0)


def anchors_for_features(
    features: list[Tensor],
    strides: tuple[int, ...] = STRIDES,
    sizes: tuple[int, ...] = ANCHOR_SIZES,
    ratios: tuple[float, ...] = ANCHOR_RATIOS,
) -> list[Tensor]:
    """One anchor set per pyramid level, on the feature maps' device and dtype."""
    return [
        anchors_for_level(
            feature.shape[-2],
            feature.shape[-1],
            stride,
            sizes,
            ratios,
            device=feature.device,
            dtype=feature.dtype,
        )
        for feature, stride in zip(features, strides, strict=True)
    ]


def hull(boxes: Tensor) -> Tensor:
    """Axis-aligned hull ``x1, y1, x2, y2`` of ``cx, cy, w, h, theta`` boxes."""
    centres = boxes[:, :2]
    half_w = boxes[:, 2:3] / 2
    half_h = boxes[:, 3:4] / 2
    cos = torch.cos(boxes[:, 4:5]).abs()
    sin = torch.sin(boxes[:, 4:5]).abs()
    extent_x = half_w * cos + half_h * sin
    extent_y = half_w * sin + half_h * cos
    return torch.cat(
        [
            centres - torch.cat([extent_x, extent_y], dim=1),
            centres + torch.cat([extent_x, extent_y], dim=1),
        ],
        dim=1,
    )


def polygons(boxes: Tensor) -> Tensor:
    """``[N, 4, 2]`` corners of le90 boxes, counter-clockwise."""
    centres, sizes, theta = boxes[:, :2], boxes[:, 2:4], boxes[:, 4]
    cos, sin = torch.cos(theta), torch.sin(theta)
    half = sizes / 2
    corners = torch.stack(
        [
            torch.stack([-half[:, 0], -half[:, 1]], dim=1),
            torch.stack([half[:, 0], -half[:, 1]], dim=1),
            torch.stack([half[:, 0], half[:, 1]], dim=1),
            torch.stack([-half[:, 0], half[:, 1]], dim=1),
        ],
        dim=1,
    )  # [N, 4, 2]
    rotation = torch.stack(
        [torch.stack([cos, -sin], dim=1), torch.stack([sin, cos], dim=1)], dim=1
    )  # [N, 2, 2]
    return torch.einsum("nij,nkj->nki", rotation, corners) + centres[:, None, :]


def _angle_delta(target: Tensor, anchor: Tensor) -> Tensor:
    """Shortest signed difference between two angles, in units of 90°."""
    difference = torch.atan2(torch.sin(target - anchor), torch.cos(target - anchor))
    return difference / HALF_PI


def encode(anchors: Tensor, boxes: Tensor) -> Tensor:
    """Regress ``cx, cy, w, h, theta`` boxes from ``x1, y1, x2, y2`` anchors.

    Each anchor becomes a zero-angle box with the anchor's width/height, so an
    axis-aligned anchor proposes a rotated box by angle alone.
    """
    if anchors.shape[0] != boxes.shape[0]:
        raise ValueError("anchors and boxes must have the same length")
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = anchors[:, 1] + 0.5 * anchor_h
    weights = torch.tensor(REG_WEIGHTS, device=boxes.device, dtype=boxes.dtype)
    # A degenerate annotation (zero width or height, e.g. duplicate vertices) makes
    # log(w / anchor) -inf and the RPN regression loss inf. The loaders drop such
    # boxes, and this floor keeps a stray one from poisoning the loss anyway; it
    # only binds for ratios below 1e-3, far outside any real object/anchor pair.
    floor = boxes.new_tensor(1e-3)
    widths = boxes[:, 2].clamp_min(floor)
    heights = boxes[:, 3].clamp_min(floor)
    targets = torch.stack(
        [
            (boxes[:, 0] - anchor_cx) / anchor_w,
            (boxes[:, 1] - anchor_cy) / anchor_h,
            torch.log(widths / anchor_w),
            torch.log(heights / anchor_h),
            _angle_delta(boxes[:, 4], torch.zeros_like(boxes[:, 4])),
        ],
        dim=1,
    )
    return targets * weights


def decode(anchors: Tensor, deltas: Tensor) -> Tensor:
    """Inverse of :func:`encode`; returns ``cx, cy, w, h, theta`` boxes."""
    weights = torch.tensor(REG_WEIGHTS, device=deltas.device, dtype=deltas.dtype)
    raw = deltas / weights
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = anchors[:, 1] + 0.5 * anchor_h
    cx = raw[:, 0] * anchor_w + anchor_cx
    cy = raw[:, 1] * anchor_h + anchor_cy
    w = torch.exp(raw[:, 2].clamp(max=4.0)) * anchor_w
    h = torch.exp(raw[:, 3].clamp(max=4.0)) * anchor_h
    theta = torch.atan2(torch.sin(raw[:, 4] * HALF_PI), torch.cos(raw[:, 4] * HALF_PI))
    return torch.stack([cx, cy, w, h, theta], dim=1)


def clip_boxes(boxes: Tensor, width: int, height: int) -> Tensor:
    """Clamp ``cx, cy, w, h, theta`` boxes so their centre stays in the image."""
    cx = boxes[:, 0].clamp(0, width)
    cy = boxes[:, 1].clamp(0, height)
    return torch.stack([cx, cy, boxes[:, 2], boxes[:, 3], boxes[:, 4]], dim=1)
