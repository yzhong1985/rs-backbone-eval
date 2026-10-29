"""COCO-style AP for the FAIR1M detection track (101-point interpolation).

The protocol publishes COCO AP (AP, AP50, AP75) so a FAIR1M cell can be compared
with published OBB numbers without re-deriving the metric, plus the two things a
backbone comparison actually needs: per-class AP and size-binned recall
(FAIR1M plan §9, decision 5: COCO 101, comparable).

Deliberately not a reimplementation of pycocotools: matching is greedy by score
against exact polygon IoU, ignored ground truth neither counts as a hit nor as a
false positive, and a class with no ground truth is excluded from the mean rather
than scored zero. That is the same contract, without a compiled dependency in the
evaluator.

Detections and ground truth are bucketed by (class, image) once, then reused for
every IoU threshold; rebuilding those lists inside the threshold loop is the
difference between seconds and minutes over ~1 700 scenes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import polygon_area, polygon_iou

DEFAULT_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
# COCO area bins in pixels squared. At the declared 0.5 m GSD a 32 px box is
# ~16 m across and a 96 px box ~48 m, so these separate cars from ships and
# airfields; the runs record gsd_source=declared next to them.
SMALL_AREA = 32**2
LARGE_AREA = 96**2
RECALL_LEVELS = 101


@dataclass
class Predictions:
    polygons: list[np.ndarray] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.scores)

    def append(self, polygon: np.ndarray, class_id: int, score: float) -> None:
        self.polygons.append(np.asarray(polygon, dtype=np.float64))
        self.class_ids.append(int(class_id))
        self.scores.append(float(score))


@dataclass
class Targets:
    polygons: list[np.ndarray] = field(default_factory=list)
    class_ids: list[int] = field(default_factory=list)
    # Objects cut by the tile edge: matched detections are neither TP nor FP.
    ignored: list[bool] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.class_ids)

    def append(self, polygon: np.ndarray, class_id: int, *, ignored: bool = False) -> None:
        self.polygons.append(np.asarray(polygon, dtype=np.float64))
        self.class_ids.append(int(class_id))
        self.ignored.append(bool(ignored))


def _match_class(
    detections: list[tuple[float, np.ndarray]],
    targets: list[tuple[np.ndarray, bool]],
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy score-ordered matching for one class and one IoU threshold.

    Returns ``(tp, matched_target)``; targets are matched at most once, and a
    detection that lands on an ignored object is matched to it but is not a true
    positive, so it is neither rewarded nor punished.
    """
    taken = [False] * len(targets)
    tp = np.zeros(len(detections), dtype=bool)
    matched = np.full(len(detections), -1, dtype=np.int64)
    order = sorted(range(len(detections)), key=lambda index: detections[index][0], reverse=True)
    for index in order:
        _, polygon = detections[index]
        best_index, best_iou = -1, iou_threshold
        for target_index, (target, _ignored) in enumerate(targets):
            if taken[target_index]:
                continue
            overlap = polygon_iou(polygon, target)
            if overlap >= best_iou:
                best_index, best_iou = target_index, overlap
        if best_index < 0:
            continue
        taken[best_index] = True
        matched[index] = best_index
        if not targets[best_index][1]:
            tp[index] = True
    return tp, matched


def average_precision(tp: np.ndarray, fp: np.ndarray, positives: int) -> float | None:
    """101-point interpolated AP; ``None`` when the class has no ground truth."""
    if positives <= 0:
        return None
    if tp.size == 0:
        return 0.0
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall = cumulative_tp / positives
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    levels = np.linspace(0.0, 1.0, RECALL_LEVELS)
    sampled = np.array(
        [envelope[recall >= level].max() if np.any(recall >= level) else 0.0 for level in levels]
    )
    return float(sampled.mean())


def _bucket(
    predictions: dict[str, Predictions],
    targets: dict[str, Targets],
    num_classes: int,
    score_threshold: float,
) -> tuple[dict[int, dict[str, list[tuple[float, np.ndarray]]]], dict[int, dict[str, list]]]:
    detections: dict[int, dict[str, list[tuple[float, np.ndarray]]]] = {
        class_id: {} for class_id in range(num_classes)
    }
    ground_truth: dict[int, dict[str, list]] = {class_id: {} for class_id in range(num_classes)}
    for image_id, target in targets.items():
        image_predictions = predictions.get(image_id)
        for class_id in range(num_classes):
            if image_predictions is not None and len(image_predictions):
                selected = [
                    (score, polygon)
                    for polygon, cid, score in zip(
                        image_predictions.polygons,
                        image_predictions.class_ids,
                        image_predictions.scores,
                        strict=True,
                    )
                    if cid == class_id and score >= score_threshold
                ]
                if selected:
                    selected.sort(key=lambda item: item[0], reverse=True)
                    detections[class_id][image_id] = selected
            items = [
                (polygon, ignored)
                for polygon, cid, ignored in zip(
                    target.polygons, target.class_ids, target.ignored, strict=True
                )
                if cid == class_id
            ]
            if items:
                ground_truth[class_id][image_id] = items
    return detections, ground_truth


def evaluate(
    predictions: dict[str, Predictions],
    targets: dict[str, Targets],
    num_classes: int,
    *,
    iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
    score_threshold: float = 0.0,
    small_area: float = SMALL_AREA,
    large_area: float = LARGE_AREA,
) -> dict[str, object]:
    """AP / AP50 / AP75, per-class AP50 and size-binned recall at IoU 0.5.

    ``predictions`` and ``targets`` are keyed by image id (the scene, not the
    tile) so a merged detection list is scored once per scene.
    """
    if set(predictions) - set(targets):
        missing = sorted(set(predictions) - set(targets))
        raise ValueError(f"predictions without targets: {missing[:5]}")

    detections, ground_truth = _bucket(predictions, targets, num_classes, score_threshold)
    index_50 = list(iou_thresholds).index(0.5) if 0.5 in iou_thresholds else 0
    index_75 = list(iou_thresholds).index(0.75) if 0.75 in iou_thresholds else 0
    per_class_ap: dict[int, list[float]] = {}
    recall_bins = {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}

    for class_id in range(num_classes):
        class_targets = ground_truth.get(class_id, {})
        positives = sum(
            1 for items in class_targets.values() for _, ignored in items if not ignored
        )
        if positives == 0:
            continue
        class_detections = detections.get(class_id, {})
        thresholds_ap: list[float] = []
        for threshold in iou_thresholds:
            tp_chunks: list[np.ndarray] = []
            score_chunks: list[float] = []
            matched_by_image: dict[str, np.ndarray] = {}
            for image_id, items in class_detections.items():
                tp, matched = _match_class(items, class_targets.get(image_id, []), threshold)
                tp_chunks.append(tp)
                score_chunks.extend(score for score, _ in items)
                matched_by_image[image_id] = matched
            if tp_chunks:
                tp = np.concatenate(tp_chunks)
                order = np.argsort(-np.asarray(score_chunks))
                tp = tp[order]
                fp = ~tp
            else:
                tp = np.zeros(0, dtype=bool)
                fp = np.zeros(0, dtype=bool)
            value = average_precision(tp, fp, positives)
            thresholds_ap.append(0.0 if value is None else value)
            if threshold == 0.5:
                _accumulate_recall(
                    class_targets, matched_by_image, small_area, large_area, recall_bins
                )
        per_class_ap[class_id] = thresholds_ap

    values = np.asarray(list(per_class_ap.values())) if per_class_ap else np.zeros((0, 0))
    return {
        "map": float(values.mean()) if values.size else 0.0,
        "map_50": float(values[:, index_50].mean()) if values.size else 0.0,
        "map_75": float(values[:, index_75].mean()) if values.size else 0.0,
        "classes_with_targets": len(per_class_ap),
        "per_class_ap50": {
            class_id: float(aps[index_50]) for class_id, aps in sorted(per_class_ap.items())
        },
        "recall_small": _ratio(recall_bins["small"]),
        "recall_medium": _ratio(recall_bins["medium"]),
        "recall_large": _ratio(recall_bins["large"]),
    }


def _ratio(pair: list[int]) -> float:
    return float(pair[0] / pair[1]) if pair[1] else 0.0


def _accumulate_recall(
    class_targets: dict[str, list],
    matched_by_image: dict[str, np.ndarray],
    small_area: float,
    large_area: float,
    bins: dict[str, list[int]],
) -> None:
    """Bucket matched/total ground truth by COCO area at IoU 0.5."""
    for image_id, items in class_targets.items():
        matched = matched_by_image.get(image_id, np.zeros(0, dtype=np.int64))
        hits = set(int(index) for index in matched if index >= 0)
        for index, (polygon, ignored) in enumerate(items):
            if ignored:
                continue
            area = polygon_area(polygon)
            name = "small" if area < small_area else ("large" if area > large_area else "medium")
            bins[name][1] += 1
            if index in hits:
                bins[name][0] += 1
