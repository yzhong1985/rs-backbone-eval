"""Learning-rate schedules for the budget-sensitivity controls."""

from __future__ import annotations

import math


def warmup_cosine_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup from 0, then cosine decay to 0 at ``total_steps``."""
    if total_steps <= 0:
        return 0.0
    if warmup_steps > 0 and step < warmup_steps:
        return max(0.0, step / warmup_steps)
    if total_steps <= warmup_steps:
        return 0.0
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """LambdaLR implementing :func:`warmup_cosine_factor`."""
    from torch.optim.lr_scheduler import LambdaLR

    return LambdaLR(
        optimizer,
        lambda step: warmup_cosine_factor(step, total_steps, warmup_steps),
    )
