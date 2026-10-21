"""Training objectives shared by every backbone worker.

B1 only means something if the three backbones train under the same objective,
so the loss definitions live here rather than being re-implemented per worker.
The design doc fixes the objective per task: weighted cross-entropy for T3 and
focal loss for T4 (the TerraTorch sample configuration the design cites).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def inverse_frequency_weights(counts: list[int] | torch.Tensor) -> torch.Tensor:
    """Class weights as ``w_c ∝ 1/f_c``, normalised to mean 1 over present classes.

    A class that never occurs in the training split gets weight 0 so the loss
    cannot chase it; the mean is taken over the classes that do occur, which
    keeps the overall gradient scale comparable to an unweighted loss (and to
    the T3 runs that were trained without weights).
    """
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("counts must be a non-empty 1-D sequence")
    if (counts < 0).any():
        raise ValueError("class counts must be non-negative")
    present = counts > 0
    weights = torch.zeros_like(counts)
    if not bool(present.any()):
        return weights
    inverse = counts.sum() / (present.sum() * counts[present])
    weights[present] = inverse
    # float64 keeps the normalisation exact; the loss wants float32.
    return (weights / weights[present].mean()).to(torch.float32)


class WeightedCrossEntropy(nn.Module):
    """Class-weighted cross-entropy normalised by valid pixels.

    ``torch.nn.CrossEntropyLoss(weight=..., reduction="mean")`` divides by the
    sum of the target-class weights, so two workers given identical logits,
    targets and weights get different gradient scales — the B1 backbones would
    be optimising different objectives. This divides by the number of valid
    pixels instead and keeps ``reduction="sum"`` for the accumulation path,
    where the window gradient is scaled once by its own pixel count.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        ignore_index: int = -1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"unknown reduction: {reduction!r}")
        self.ignore_index = int(ignore_index)
        self.reduction = reduction
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        per_pixel = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        valid = target != self.ignore_index
        if not bool(valid.any()):
            raise ValueError("weighted cross-entropy got no valid pixels")
        total = per_pixel[valid].sum()
        return total / valid.sum() if self.reduction == "mean" else total


class FocalLoss(nn.Module):
    """Multi-class focal loss on logits: ``-(1-p_t)^gamma * log p_t``.

    ``gamma=2`` and no class prior, matching the TerraTorch sample config the
    design doc cites for T4. Invalid pixels (``target == ignore_index``) are
    excluded from the mean so the loss scale matches the cross-entropy it
    replaces.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        ignore_index: int = -1,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"unknown reduction: {reduction!r}")
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.reduction = reduction
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probability = F.log_softmax(logits, dim=1)
        safe = target.clone()
        safe[safe == self.ignore_index] = 0
        log_pt = log_probability.gather(1, safe.unsqueeze(1)).squeeze(1)
        loss = -(1.0 - log_pt.exp()).pow(self.gamma) * log_pt
        if self.weight is not None:
            loss = loss * self.weight[safe]
        valid = target != self.ignore_index
        if not bool(valid.any()):
            raise ValueError("focal loss got no valid pixels")
        # ``sum`` matches the accumulation path, which scales the accumulated
        # gradient by the window's valid-pixel count once per optimizer step.
        return loss[valid].sum() if self.reduction == "sum" else loss[valid].mean()
