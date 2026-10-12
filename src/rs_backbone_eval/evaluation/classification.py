"""Scene classification metrics: macro F1 (primary), accuracy, balanced accuracy."""

from __future__ import annotations

import math

import numpy as np

from .confusion import ConfusionMatrix


def classification_metrics(cm: ConfusionMatrix) -> dict[str, float]:
    """Macro-F1 (primary) plus accuracy, balanced accuracy, and per-class recall."""
    supported = cm.present_classes()
    f1_values = [cm.f1(index) for index in supported]
    recall_values = [cm.recall(index) for index in supported]
    return {
        "macro_f1": float(np.mean(f1_values)) if f1_values else math.nan,
        "accuracy": _safe_div(float(cm.counts.trace()), cm.total),
        "balanced_accuracy": float(np.mean(recall_values)) if recall_values else math.nan,
        **{f"recall/class_{index}": cm.recall(index) for index in supported},
    }


def _safe_div(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else math.nan
