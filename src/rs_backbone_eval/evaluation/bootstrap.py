"""Deterministic bootstrap confidence intervals for per-sample statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    point: float
    low: float
    high: float
    confidence: float
    n_resamples: int
    sample_count: int


def bootstrap_ci(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 17,
) -> BootstrapInterval:
    """Percentile bootstrap of the mean; identical inputs and seed give identical bounds."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    array = np.asarray(values, dtype=np.float64).ravel()
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("values must contain at least one finite entry")
    point = float(array.mean())
    if array.size == 1 or n_resamples <= 0:
        return BootstrapInterval(point, point, point, confidence, max(n_resamples, 0), array.size)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(n_resamples, array.size))
    means = array[draws].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return BootstrapInterval(
        point=point,
        low=float(low),
        high=float(high),
        confidence=confidence,
        n_resamples=n_resamples,
        sample_count=array.size,
    )
