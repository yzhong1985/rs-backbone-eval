"""Segmentation metrics: binary and multi-class, derived from one global confusion matrix."""

from __future__ import annotations

import math

import numpy as np

from .confusion import ConfusionMatrix


def binary_metrics(cm: ConfusionMatrix, *, foreground: int = 1) -> dict[str, float]:
    """Foreground IoU (primary), F1, precision, recall, and two-class mIoU."""
    if cm.num_classes != 2:
        raise ValueError("binary metrics need a 2-class confusion matrix")
    if foreground not in (0, 1):
        raise ValueError("foreground class must be 0 or 1")
    per_class_iou = [cm.iou(index) for index in range(cm.num_classes)]
    valid_miou = [iou for iou in per_class_iou if not math.isnan(iou)]
    return {
        "foreground_iou": cm.iou(foreground),
        "foreground_f1": cm.f1(foreground),
        "precision": cm.precision(foreground),
        "recall": cm.recall(foreground),
        "mean_iou": float(np.mean(valid_miou)) if valid_miou else math.nan,
        **{f"iou/class_{index}": iou for index, iou in enumerate(per_class_iou)},
    }


def multiclass_metrics(cm: ConfusionMatrix) -> dict[str, float]:
    """Macro mIoU (primary), macro F1, per-class IoU, and frequency-weighted IoU."""
    per_class_iou = [cm.iou(index) for index in range(cm.num_classes)]
    present = cm.present_classes(ground_truth_only=False)
    miou_values = [per_class_iou[index] for index in present]
    f1_values = [cm.f1(index) for index in cm.present_classes()]
    total = cm.total
    weighted = sum(cm.support(index) * per_class_iou[index] for index in range(cm.num_classes))
    return {
        "mean_iou": float(np.mean(miou_values)) if miou_values else math.nan,
        "macro_f1": float(np.mean(f1_values)) if f1_values else math.nan,
        "frequency_weighted_iou": float(weighted / total) if total else math.nan,
        **{f"iou/class_{index}": iou for index, iou in enumerate(per_class_iou)},
    }


def per_sample_foreground_iou(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    foreground: int = 1,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Per-sample foreground IoU along axis 0, for bootstrap confidence intervals."""
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.ndim < 2 or truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must share shape [N, ...]")
    scores = np.full(truth.shape[0], math.nan, dtype=np.float64)
    for index in range(truth.shape[0]):
        sample_true, sample_pred = truth[index].ravel(), pred[index].ravel()
        valid = np.ones(sample_true.shape, dtype=bool)
        if ignore_index is not None:
            valid &= sample_true != ignore_index
        gt = sample_true[valid] == foreground
        positive = sample_pred[valid] == foreground
        tp = int((gt & positive).sum())
        union = int((gt | positive).sum())
        if union:
            scores[index] = tp / union
    return scores
