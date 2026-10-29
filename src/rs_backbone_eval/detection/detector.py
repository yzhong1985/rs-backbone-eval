"""Rotated Faster R-CNN on a fixed FPN, for any backbone that exposes 4 levels.

Design decisions, all forced by "the detector is the fixed variable":

* horizontal proposals, horizontal IoU for matching/NMS, ordinary ``roi_align``;
  only the regression target and the final boxes are rotated (plan §7.1, G0);
* anchors, weights and the angle encoding live in ``detection.anchors``;
* the backbone is a callable returning four feature maps and is *not* owned by
  the detector, so a frozen encoder and a fine-tuned one plug into the same
  module and the trainable-parameter report stays honest.

This module imports torchvision, therefore only workers may import it; the
control plane never does (AGENTS.md layering).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import box_iou, nms, roi_align

from ..heads import default_norm
from .anchors import (
    ANCHOR_RATIOS,
    ANCHOR_SIZES,
    STRIDES,
    anchors_for_features,
    clip_boxes,
    decode,
    encode,
    hull,
    polygons,
)
from .fpn import FPN

# IoU against every ground-truth box at once is [N_anchors, N_gt]: a dense
# FAIR1M tile has ~1.3 M anchors and several hundred objects, so one matrix is
# multi-GB and `box_iou` needs a few temporaries of that size. Chunking keeps
# the same values with a working set of `IOU_CHUNK * N_gt`.
IOU_CHUNK = 32768


def _iou_rows(boxes: torch.Tensor, others: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``box_iou(boxes, others).max(dim=1)`` without materialising the matrix."""
    n = boxes.shape[0]
    values = boxes.new_empty(n)
    indices = torch.empty(n, dtype=torch.long, device=boxes.device)
    for start in range(0, n, IOU_CHUNK):
        stop = min(start + IOU_CHUNK, n)
        chunk = box_iou(boxes[start:stop], others)
        best, index = chunk.max(dim=1)
        values[start:stop] = best
        indices[start:stop] = index
    return values, indices


def _iou_columns(boxes: torch.Tensor, others: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``box_iou(boxes, others).max(dim=0)``; ties keep the first box."""
    m = others.shape[0]
    values = boxes.new_full((m,), -1.0)
    indices = torch.zeros(m, dtype=torch.long, device=boxes.device)
    for start in range(0, boxes.shape[0], IOU_CHUNK):
        stop = min(start + IOU_CHUNK, boxes.shape[0])
        chunk = box_iou(boxes[start:stop], others)
        best, index = chunk.max(dim=0)
        better = best > values
        values[better] = best[better]
        indices[better] = index[better] + start
    return values, indices


RPN_POSITIVE = 0.7
RPN_NEGATIVE = 0.3
RPN_BATCH = 256
RPN_POSITIVE_FRACTION = 0.5
ROI_POSITIVE = 0.5
ROI_BATCH = 512
ROI_POSITIVE_FRACTION = 0.25
ROI_POOL = 7
CANONICAL_SCALE = 4.0  # roi_align scale used by the reference FPN


@dataclass
class GroundTruth:
    """Rotated ground truth for one image, in that image's pixel coordinates.

    ``ignored_boxes`` holds objects that are only partly inside the tile (the
    manifest's ``ignored_instance_count``). They are neither trained as objects
    nor as background: a proposal that lands on one is dropped from sampling, so
    the model is not taught that a truncated ship is empty water (plan §5.2).
    """

    boxes: torch.Tensor  # [N, 5] cx, cy, w, h, theta (le90)
    labels: torch.Tensor  # [N] 0..num_classes-1
    ignored_boxes: torch.Tensor | None = None  # [M, 5] same convention
    # Real-pixel rectangle ``(x0, y0, x1, y1)`` of the padded tile. Anchors and
    # proposals centred in the padded band are dropped from sampling instead of
    # being counted as background: 43% of the FAIR1M tiles are less than half
    # real pixels, and those zeros would otherwise be free negatives.
    valid_region: tuple[float, float, float, float] | None = None

    def __len__(self) -> int:
        return int(self.labels.numel())

    def valid_mask(self, hulls: torch.Tensor) -> torch.Tensor:
        """Which ``x1, y1, x2, y2`` boxes have their centre on real pixels."""
        if self.valid_region is None:
            return torch.ones(hulls.shape[0], dtype=torch.bool, device=hulls.device)
        x0, y0, x1, y1 = self.valid_region
        centre_x = (hulls[:, 0] + hulls[:, 2]) / 2
        centre_y = (hulls[:, 1] + hulls[:, 3]) / 2
        return (centre_x >= x0) & (centre_x < x1) & (centre_y >= y0) & (centre_y < y1)

    def ignore_mask(self, hulls: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Which ``x1, y1, x2, y2`` hulls overlap an ignored object too much."""
        if self.ignored_boxes is None or self.ignored_boxes.numel() == 0:
            return torch.zeros(hulls.shape[0], dtype=torch.bool, device=hulls.device)
        best, _ = _iou_rows(hulls, hull(self.ignored_boxes.to(hulls.device)))
        return best >= threshold


class RPNHead(nn.Module):
    """3x3 conv + per-anchor objectness and 5-parameter box regression."""

    def __init__(self, channels: int, anchors: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            default_norm(channels),
            nn.ReLU(inplace=True),
        )
        self.objectness = nn.Conv2d(channels, anchors, kernel_size=1)
        self.regression = nn.Conv2d(channels, anchors * 5, kernel_size=1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.conv(features)
        return self.objectness(hidden), self.regression(hidden)


class RotatedFasterRCNN(nn.Module):
    """FPN + RPN + RoI head over rotated boxes.

    ``backbone`` returns ``[p2, p3, p4, p5]`` (any channel widths). The detector
    is trained with ``losses(features, targets)`` and used with
    ``detections(features, image_size)``; both share the same proposal path.
    """

    def __init__(
        self,
        backbone: Callable[[torch.Tensor], list[torch.Tensor]],
        in_channels: list[int],
        num_classes: int,
        *,
        fpn_dim: int = 256,
        anchor_sizes: tuple[int, ...] = ANCHOR_SIZES,
        anchor_ratios: tuple[float, ...] = ANCHOR_RATIOS,
        strides: tuple[int, ...] = STRIDES,
        rpn_pre_nms: int = 2000,
        rpn_post_nms: int = 1000,
        nms_threshold: float = 0.7,
        score_threshold: float = 0.05,
        detections_per_image: int = 100,
    ) -> None:
        super().__init__()
        if len(in_channels) != len(strides):
            raise ValueError("in_channels and strides must have the same length")
        self.backbone = backbone
        self.num_classes = num_classes
        self.anchor_sizes = anchor_sizes
        self.anchor_ratios = anchor_ratios
        self.num_anchors = len(anchor_sizes) * len(anchor_ratios)
        self.strides = strides
        self.fpn = FPN(in_channels, fpn_dim)
        self.rpn_head = RPNHead(fpn_dim, self.num_anchors)
        self.box_head = nn.Sequential(
            nn.Linear(fpn_dim * ROI_POOL * ROI_POOL, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
        )
        self.cls_score = nn.Linear(1024, num_classes + 1)
        self.bbox_pred = nn.Linear(1024, num_classes * 5)
        self.rpn_pre_nms = rpn_pre_nms
        self.rpn_post_nms = rpn_post_nms
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.detections_per_image = detections_per_image

    # ---------------------------------------------------------------- features
    def pyramid(self, images: torch.Tensor) -> list[torch.Tensor]:
        return self.fpn(self.backbone(images))

    # ------------------------------------------------------------------ anchors
    def anchors(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        return anchors_for_features(features, self.strides, self.anchor_sizes, self.anchor_ratios)

    # ------------------------------------------------------------------- losses
    def losses(
        self, features: list[torch.Tensor], targets: list[GroundTruth]
    ) -> dict[str, torch.Tensor]:
        if features[0].shape[0] != 1:
            raise ValueError(
                "RotatedFasterRCNN runs one image per forward pass (micro batch 1, "
                "gradient accumulation 2); batching needs per-image proposal bookkeeping"
            )
        anchors = self.anchors(features)
        objectness, regression = self._rpn_outputs(features, anchors)
        rpn_cls, rpn_reg = self._rpn_losses(anchors, objectness, regression, targets)
        proposals = self._proposals(anchors, objectness, regression, features[0].shape[-2:])
        cls_loss, reg_loss = self._roi_losses(features, proposals, targets)
        return {
            "loss_rpn_cls": rpn_cls,
            "loss_rpn_reg": rpn_reg,
            "loss_cls": cls_loss,
            "loss_reg": reg_loss,
            "loss": rpn_cls + rpn_reg + cls_loss + reg_loss,
        }

    def _rpn_outputs(
        self, features: list[torch.Tensor], anchors: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        objectness, regression = [], []
        for feature in features:
            cls, reg = self.rpn_head(feature)
            # [B, A, H, W] -> [B, H, W, A] so it lines up with anchors_for_level
            objectness.append(cls.permute(0, 2, 3, 1).reshape(cls.shape[0], -1))
            regression.append(reg.permute(0, 2, 3, 1).reshape(reg.shape[0], -1, 5))
        if len(anchors) != len(objectness):  # pragma: no cover - internal contract
            raise ValueError("anchor and RPN level counts differ")
        return objectness, regression

    def _rpn_losses(
        self,
        anchors: list[torch.Tensor],
        objectness: list[torch.Tensor],
        regression: list[torch.Tensor],
        targets: list[GroundTruth],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_anchors = torch.cat(anchors, dim=0)
        all_scores = torch.cat(objectness, dim=1)[0]
        all_deltas = torch.cat(regression, dim=1)[0]
        device = all_anchors.device
        total_cls = torch.zeros((), device=device)
        total_reg = torch.zeros((), device=device)
        count = 0
        for target in targets:
            best_index: torch.Tensor | None = None
            if len(target):
                ground_truth = hull(target.boxes.to(device))
                best_iou, best_index = _iou_rows(all_anchors, ground_truth)
                labels = torch.full_like(best_iou, -1)
                labels[best_iou >= RPN_POSITIVE] = 1
                labels[best_iou < RPN_NEGATIVE] = 0
                # every ground truth keeps at least its best anchor, as in the
                # reference sampler: otherwise a small object can vanish from the RPN
                _, per_ground_truth = _iou_columns(all_anchors, ground_truth)
                labels[per_ground_truth] = 1
            else:
                # A tile with no trainable object (11.7% of the FAIR1M manifest,
                # plus every tile whose only objects were truncated) is pure
                # background: sample it as negatives instead of skipping it, or
                # the RPN never learns what this mirror's empty ground looks
                # like (review 2026-10-29).
                labels = torch.zeros(all_anchors.shape[0], dtype=torch.long, device=device)
            # anchors sitting on a truncated object, or in the padded band, stay
            # unsampled: neither object nor background
            labels[target.ignore_mask(all_anchors)] = -1
            labels[~target.valid_mask(all_anchors)] = -1

            positive = int((labels == 1).sum())
            negative = int((labels == 0).sum())
            num_pos = min(
                positive, int(RPN_BATCH * RPN_POSITIVE_FRACTION), max(RPN_BATCH - negative, 0)
            )
            num_neg = min(negative, RPN_BATCH - num_pos)
            pos_index = torch.nonzero(labels == 1).flatten()
            neg_index = torch.nonzero(labels == 0).flatten()
            if num_pos > 0:
                pos_index = pos_index[torch.randperm(pos_index.numel(), device=device)[:num_pos]]
            if num_neg > 0:
                neg_index = neg_index[torch.randperm(neg_index.numel(), device=device)[:num_neg]]
            sampled = torch.cat([pos_index, neg_index])
            if sampled.numel() == 0:
                continue
            sample_labels = (labels[sampled] == 1).to(all_scores.dtype)
            total_cls = total_cls + F.binary_cross_entropy_with_logits(
                all_scores[sampled], sample_labels, reduction="sum"
            ) / max(sampled.numel(), 1)
            if pos_index.numel() and best_index is not None:
                matched = target.boxes.to(device)[best_index[pos_index]]
                target_deltas = encode(all_anchors[pos_index].detach(), matched)
                total_reg = total_reg + F.smooth_l1_loss(
                    all_deltas[pos_index], target_deltas, beta=1.0 / 9, reduction="sum"
                ) / max(pos_index.numel(), 1)
            count += 1
        divisor = max(count, 1)
        return total_cls / divisor, total_reg / divisor

    # ---------------------------------------------------------------- proposals
    def _proposals(
        self,
        anchors: list[torch.Tensor],
        objectness: list[torch.Tensor],
        regression: list[torch.Tensor],
        feature_size: torch.Size,
    ) -> list[torch.Tensor]:
        """Top-k, NMS and decode per level; returns ``cx, cy, w, h, theta``."""
        height, width = int(feature_size[0]) * 4, int(feature_size[1]) * 4
        proposals: list[torch.Tensor] = []
        for level_anchors, level_scores, level_deltas in zip(
            anchors, objectness, regression, strict=True
        ):
            scores = level_scores[0].sigmoid()
            boxes = decode(level_anchors, level_deltas[0])
            boxes = clip_boxes(boxes, width, height)
            keep = scores.argsort(descending=True)[: self.rpn_pre_nms]
            boxes, scores = boxes[keep], scores[keep]
            keep = nms(hull(boxes), scores, self.nms_threshold)[: self.rpn_post_nms]
            proposals.append(boxes[keep])
        return proposals

    # ----------------------------------------------------------------- RoI head
    def _roi_features(self, features: list[torch.Tensor], boxes: torch.Tensor) -> torch.Tensor:
        """``roi_align`` each box on the FPN level its area belongs to."""
        widths = boxes[:, 2].clamp(min=1e-6)
        heights = boxes[:, 3].clamp(min=1e-6)
        scale = torch.sqrt(widths * heights) / CANONICAL_SCALE
        levels = torch.floor(torch.log2(scale)).clamp(0, len(features) - 1).to(torch.long)
        pooled: list[torch.Tensor] = []
        for index, feature in enumerate(features):
            selected = torch.nonzero(levels == index).flatten()
            if selected.numel() == 0:
                continue
            # roi_align wants x1, y1, x2, y2 in *feature* coordinates; the hull of
            # the rotated box is the right container for its pixels
            rois = hull(boxes[selected]).detach()
            rois = rois / self.strides[index]
            # roi_align wants [K, 5] with the batch index first; rois are already
            # in this level's feature coordinates, so the spatial scale is 1.0
            rois = torch.cat([torch.zeros(rois.shape[0], 1, device=rois.device), rois], dim=1)
            pooled.append(
                roi_align(
                    feature,
                    rois,
                    output_size=ROI_POOL,
                    spatial_scale=1.0,
                    sampling_ratio=2,
                    aligned=True,
                )
            )
        if not pooled:
            return torch.zeros(
                0,
                features[0].shape[1],
                ROI_POOL,
                ROI_POOL,
                device=features[0].device,
                dtype=features[0].dtype,
            )
        return torch.cat(pooled, dim=0)

    def _roi_losses(
        self,
        features: list[torch.Tensor],
        proposals: list[torch.Tensor],
        targets: list[GroundTruth],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = features[0].device
        trainable = [target for target in targets if len(target)]
        if not proposals or not trainable:
            # nothing to match against: the empty tile still trains the RPN above
            zero = torch.zeros((), device=device)
            return zero, zero
        # Adding the ground truth to the proposals is what the reference head
        # does: early in training the RPN proposes almost nothing on a small
        # object, and without GT rows the RoI loss would see no positive at all.
        gt_boxes = torch.cat([target.boxes.to(device) for target in trainable])
        boxes = torch.cat([*proposals, gt_boxes.detach()], dim=0)
        overlaps = box_iou(hull(boxes), hull(gt_boxes))
        best_iou, best_index = overlaps.max(dim=1)
        labels = torch.cat([target.labels.to(device) for target in trainable])

        usable = ~torch.cat([target.ignore_mask(hull(boxes)) for target in trainable])
        usable &= torch.cat([target.valid_mask(hull(boxes)) for target in trainable])
        positive = torch.nonzero((best_iou >= ROI_POSITIVE) & usable).flatten()
        negative = torch.nonzero((best_iou < ROI_POSITIVE) & usable).flatten()
        num_pos = min(int(positive.numel()), int(ROI_BATCH * ROI_POSITIVE_FRACTION))
        num_neg = min(int(negative.numel()), ROI_BATCH - num_pos)
        if num_pos:
            positive = positive[torch.randperm(positive.numel(), device=device)[:num_pos]]
        if num_neg:
            negative = negative[torch.randperm(negative.numel(), device=device)[:num_neg]]
        sampled = torch.cat([positive, negative])
        if sampled.numel() == 0:
            zero = torch.zeros((), device=device)
            return zero, zero

        pooled = self._roi_features(features, boxes[sampled])
        hidden = self.box_head(pooled.flatten(1))
        cls_logits = self.cls_score(hidden)
        reg_deltas = self.bbox_pred(hidden).reshape(-1, self.num_classes, 5)

        positive_count = int(positive.numel())
        # `sampled` is built as [positive..., negative...], so the first
        # positive_count rows of the pooled features are exactly the positives.
        # Indexing them with the `positive` values (indices into `boxes`) was an
        # out-of-bounds bug that only appeared once ground truth was appended to
        # the proposals (review 2026-10-29).
        is_positive = torch.arange(sampled.numel(), device=device) < positive_count
        gt_labels = labels[best_index[sampled]]
        cls_target = torch.where(
            is_positive, gt_labels, torch.full_like(gt_labels, self.num_classes)
        )
        cls_loss = F.cross_entropy(cls_logits, cls_target, reduction="sum") / max(
            sampled.numel(), 1
        )
        if positive.numel() == 0:
            return cls_loss, torch.zeros((), device=device)
        with torch.no_grad():
            matched = gt_boxes[best_index[positive]]
        class_index = labels[best_index[positive]]
        # `boxes` are cx, cy, w, h, theta; `encode` regresses from the anchor's
        # x1, y1, x2, y2 corner form. Passing the rotated box straight in made the
        # "anchor width" negative whenever cx > w, which is almost always, so the
        # RoI regression targets were silently wrong (and became NaN once the
        # ground truth joined the proposals -- review 2026-10-29).
        target_deltas = encode(hull(boxes[positive]).detach(), matched)
        predicted = reg_deltas[:positive_count][
            torch.arange(positive_count, device=device), class_index
        ]
        reg_loss = F.smooth_l1_loss(predicted, target_deltas, beta=1.0 / 9, reduction="sum") / max(
            positive.numel(), 1
        )
        return cls_loss, reg_loss

    # --------------------------------------------------------------- inference
    @torch.no_grad()
    def detections(
        self, features: list[torch.Tensor], image_size: tuple[int, int]
    ) -> list[dict[str, torch.Tensor]]:
        """Per-image polygons, scores and labels after class-wise NMS."""
        height, width = image_size
        if features[0].shape[0] != 1:
            raise ValueError("detections() scores one image per call")
        anchors = self.anchors(features)
        objectness, regression = self._rpn_outputs(features, anchors)
        proposals = self._proposals(anchors, objectness, regression, features[0].shape[-2:])
        boxes = torch.cat(proposals, dim=0)
        pooled = self._roi_features(features, boxes)
        hidden = self.box_head(pooled.flatten(1))
        scores = F.softmax(self.cls_score(hidden), dim=1)[:, : self.num_classes]
        deltas = self.bbox_pred(hidden).reshape(-1, self.num_classes, 5)

        output: list[dict[str, torch.Tensor]] = []
        for class_id in range(self.num_classes):
            class_scores = scores[:, class_id]
            keep = torch.nonzero(class_scores >= self.score_threshold).flatten()
            if keep.numel() == 0:
                continue
            # `boxes` are cx, cy, w, h, theta (hulls of the proposals); decode
            # expects the anchor in x1, y1, x2, y2. Passing the rotated box here
            # produced negative widths at inference while training regressed from
            # the hull, so every predicted box was wrong (review 2026-10-29).
            decoded = decode(hull(boxes[keep]), deltas[keep, class_id])
            decoded = clip_boxes(decoded, width, height)
            keep = nms(hull(decoded), class_scores[keep], 0.5)[: self.detections_per_image]
            if keep.numel() == 0:
                continue
            output.append(
                {
                    "polygons": polygons(decoded[keep]),
                    "scores": class_scores[keep],
                    "labels": torch.full(
                        (keep.numel(),), class_id, device=decoded.device, dtype=torch.long
                    ),
                }
            )
        if not output:
            empty = torch.zeros(0, 4, 2, device=features[0].device)
            return [
                {
                    "polygons": empty,
                    "scores": torch.zeros(0, device=features[0].device),
                    "labels": torch.zeros(0, device=features[0].device, dtype=torch.long),
                }
            ]
        return [
            {
                "polygons": torch.cat([item["polygons"] for item in output], dim=0),
                "scores": torch.cat([item["scores"] for item in output], dim=0),
                "labels": torch.cat([item["labels"] for item in output], dim=0),
            }
        ]

    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def anchor_count(height: int, width: int) -> int:
    """Anchors at one level for a feature map, for memory budgeting."""
    return int(math.prod((height, width, len(ANCHOR_SIZES) * len(ANCHOR_RATIOS))))
