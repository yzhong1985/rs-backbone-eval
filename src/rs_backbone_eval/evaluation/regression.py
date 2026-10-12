"""Pixel regression metrics: RMSE (primary), MAE, R², bias, and target-bin errors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BinStats:
    label: str
    count: int
    rmse: float
    mae: float
    bias: float


class RegressionStats:
    """Accumulate residuals and keep per-sample values for bootstrap intervals."""

    def __init__(self, bin_edges: tuple[float, ...] | None = None) -> None:
        if bin_edges is not None and len(bin_edges) < 2:
            raise ValueError("bin_edges needs at least two boundaries")
        self.bin_edges = tuple(math.inf if math.isinf(edge) else edge for edge in (bin_edges or ()))
        self._y_true: list[np.ndarray] = []
        self._y_pred: list[np.ndarray] = []

    def update(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        valid_mask: np.ndarray | None = None,
    ) -> None:
        truth = np.asarray(y_true, dtype=np.float64).ravel()
        pred = np.asarray(y_pred, dtype=np.float64).ravel()
        if truth.shape != pred.shape:
            raise ValueError(f"shape mismatch: y_true {truth.shape} != y_pred {pred.shape}")
        valid = np.ones(truth.shape, dtype=bool)
        if valid_mask is not None:
            mask = np.asarray(valid_mask).ravel().astype(bool)
            if mask.shape != truth.shape:
                raise ValueError(f"valid_mask shape {mask.shape} != targets {truth.shape}")
            valid &= mask
        truth, pred = truth[valid], pred[valid]
        if not np.isfinite(truth).all() or not np.isfinite(pred).all():
            raise ValueError("regression targets and predictions must be finite")
        if truth.size == 0:
            return
        self._y_true.append(truth)
        self._y_pred.append(pred)

    @property
    def count(self) -> int:
        return sum(array.size for array in self._y_true)

    def _concat(self, arrays: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(arrays) if arrays else np.array([], dtype=np.float64)

    @property
    def y_true(self) -> np.ndarray:
        return self._concat(self._y_true)

    @property
    def y_pred(self) -> np.ndarray:
        return self._concat(self._y_pred)

    @property
    def residuals(self) -> np.ndarray:
        return self.y_pred - self.y_true

    def metrics(self) -> dict[str, float]:
        """RMSE (primary), MAE, R², and bias over all accumulated samples."""
        truth, pred = self.y_true, self.y_pred
        if truth.size == 0:
            return {"rmse": math.nan, "mae": math.nan, "r2": math.nan, "bias": math.nan}
        errors = pred - truth
        ss_res = float((errors**2).sum())
        ss_tot = float(((truth - truth.mean()) ** 2).sum())
        return {
            "rmse": math.sqrt(ss_res / truth.size),
            "mae": float(np.abs(errors).mean()),
            "r2": _r_squared(ss_res, ss_tot),
            "bias": float(errors.mean()),
        }

    def binned_metrics(self) -> list[BinStats]:
        """RMSE / MAE / bias grouped by target value bins (e.g. AGB ranges)."""
        if not self.bin_edges:
            raise ValueError("bin_edges were not configured for this accumulator")
        truth, pred = self.y_true, self.y_pred
        stats: list[BinStats] = []
        for lower, upper in zip(self.bin_edges, self.bin_edges[1:], strict=False):
            in_bin = (truth >= lower) & (truth < upper)
            label = (
                f"{_format_edge(lower)}-{_format_edge(upper)}"
                if not math.isinf(upper)
                else f">={_format_edge(lower)}"
            )
            if in_bin.sum() == 0:
                stats.append(BinStats(label, 0, math.nan, math.nan, math.nan))
                continue
            errors = pred[in_bin] - truth[in_bin]
            stats.append(
                BinStats(
                    label=label,
                    count=int(in_bin.sum()),
                    rmse=float(math.sqrt((errors**2).mean())),
                    mae=float(np.abs(errors).mean()),
                    bias=float(errors.mean()),
                )
            )
        return stats


def _r_squared(ss_res: float, ss_tot: float) -> float:
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _format_edge(edge: float) -> str:
    return "inf" if math.isinf(edge) else f"{edge:g}"
