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
        # A class absent from both ground truth and prediction has no IoU
        # (0/0). Emitting NaN here poisoned both the run record and, via
        # ``0 * NaN``, the weighted average (review 2026-10-27), so undefined
        # classes are omitted; the confusion-matrix artifact keeps the counts.
        **{
            f"iou/class_{index}": iou
            for index, iou in enumerate(per_class_iou)
            if not math.isnan(iou)
        },
    }


def multiclass_metrics(cm: ConfusionMatrix) -> dict[str, float]:
    """Macro mIoU (primary), macro F1, per-class IoU, and frequency-weighted IoU."""
    per_class_iou = [cm.iou(index) for index in range(cm.num_classes)]
    present = cm.present_classes(ground_truth_only=False)
    miou_values = [per_class_iou[index] for index in present]
    f1_values = [cm.f1(index) for index in cm.present_classes()]
    total = cm.total
    # Skip undefined classes instead of adding ``0 * NaN``. For a class with no
    # support the contribution is 0 anyway, but NaN would poison the whole sum
    # and land in ``run.json`` as ``null`` (review 2026-10-27, T8 class_0).
    weighted = sum(
        cm.support(index) * iou for index, iou in enumerate(per_class_iou) if not math.isnan(iou)
    )
    # Class 0 is the background/ignore class in every multi-class task here
    # (T3 burn severity, T5 crop types, T8 land cover). Reporting its mean
    # separately stops a dominant background from masking a weak foreground:
    # T4 mIoU 0.565 vs foreground IoU 0.148, T3 mIoU 0.156 vs 0.059.
    no_background = [per_class_iou[index] for index in present if index != 0]
    return {
        "mean_iou": float(np.mean(miou_values)) if miou_values else math.nan,
        "mean_iou_no_background": float(np.mean(no_background)) if no_background else math.nan,
        "macro_f1": float(np.mean(f1_values)) if f1_values else math.nan,
        "frequency_weighted_iou": float(weighted / total) if total else math.nan,
        **{
            f"iou/class_{index}": iou
            for index, iou in enumerate(per_class_iou)
            if not math.isnan(iou)
        },
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
