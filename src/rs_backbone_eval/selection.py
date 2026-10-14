"""Validation-based checkpoint selection shared by the fine-tuning (B1) workers.

Models are chosen on the project validation split only. The test split is
evaluated once, on the weights selected by validation.
"""

from __future__ import annotations

import os

import torch
from torch import nn


def validation_every() -> int:
    try:
        return max(1, int(os.environ.get("RS_EVAL_VAL_EVERY", "5")))
    except ValueError:
        return 5


class BestCheckpoint:
    def __init__(self) -> None:
        self.score = float("-inf")
        self.epoch = 0
        self.state: dict[str, torch.Tensor] | None = None

    def offer(self, score: float, epoch: int, model: nn.Module) -> bool:
        if score <= self.score:
            return False
        self.score = score
        self.epoch = epoch
        self.state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        return True

    def restore(self, model: nn.Module) -> None:
        if self.state is not None:
            model.load_state_dict(self.state)

    def save(self, path) -> None:
        """Persist the selected weights so the run can be re-evaluated later.

        Without this the selection lives only in memory and a completed run
        cannot be re-scored on a holdout split without retraining.
        """
        if self.state is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"epoch": self.epoch, "score": self.score, "state_dict": self.state},
            path,
        )
