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


SCHEDULES = (None, "warmup1-cosine")


def resolve_budget(
    protocol,
    *,
    official_epochs: int,
    default_lr: float,
    epochs_override: str | None = None,
    lr_override: str | None = None,
) -> tuple[int, float, str | None]:
    """``(epochs, learning_rate, lr_schedule)`` for a run, protocol first.

    The protocol is the experimental condition: a declared ``epochs`` /
    ``learning_rate`` / ``lr_schedule`` must reach the training loop. Workers
    used to read only their environment variables, so the ``_e20`` protocols
    ran the 8-epoch default (review 2026-11-18, P1). An environment override
    is still honoured when the protocol is silent, and refused when it
    contradicts the protocol instead of silently winning.
    """
    epochs, lr = official_epochs, default_lr
    if protocol.epochs is not None:
        epochs = int(protocol.epochs)
        if epochs_override not in (None, "") and int(epochs_override) != epochs:
            raise ValueError(
                f"{protocol.protocol_id} declares epochs={epochs}; "
                f"the environment asks for {epochs_override}"
            )
    elif epochs_override not in (None, ""):
        epochs = int(epochs_override)
    if protocol.learning_rate is not None:
        lr = float(protocol.learning_rate)
        if lr_override not in (None, "") and float(lr_override) != lr:
            raise ValueError(
                f"{protocol.protocol_id} declares learning_rate={lr:g}; "
                f"the environment asks for {lr_override}"
            )
    elif lr_override not in (None, ""):
        lr = float(lr_override)
    schedule = protocol.lr_schedule
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown lr_schedule {schedule!r}")
    return epochs, lr, schedule
