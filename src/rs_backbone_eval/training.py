"""Gradient-accumulation helpers shared by the workers.

A smaller micro-batch with gradient accumulation must be numerically equivalent
to one large batch, otherwise the effective batch (and fairness) silently
changes. Naively averaging per-micro-batch *mean* losses is not equivalent when
micro-batches have different numbers of valid (non-ignored) pixels; accumulating
``reduction="sum"`` losses and dividing the gradient once by the window's valid
pixel count is.
"""

from __future__ import annotations

from collections.abc import Iterable

from torch import nn


def scale_accumulated_gradients(parameters: Iterable[nn.Parameter], valid_pixels: int) -> None:
    """Divide accumulated gradients by the window's valid pixel count.

    With ``reduction="sum"`` the accumulated gradient is the gradient of the sum
    of per-pixel losses over the whole window. Dividing once by the number of
    valid pixels gives the gradient of the full-batch *mean over valid pixels*,
    which is exactly what a single ``reduction="mean"`` batch computes. A short
    final window is scaled by its own count rather than the nominal window size.
    """
    if not valid_pixels:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad /= valid_pixels
