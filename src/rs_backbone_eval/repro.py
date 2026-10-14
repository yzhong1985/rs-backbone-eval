"""Deterministic RNG seeding shared by every worker training path.

Before this module existed only the Prithvi B0 baseline called
``lightning.seed_everything``; the custom AnySat/Clay/Prithvi paths recorded the
seed but never seeded Python, NumPy or Torch, so a run could not be reproduced
from its recorded seed. ``seed_everything`` must be called before any model,
adapter, head or DataLoader is constructed.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and Torch (CPU + all CUDA devices) from one integer."""
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_init_requested() -> bool:
    """Whether the run should skip pretrained weights (the C12 control).

    Set ``RS_EVAL_INIT=random`` to train the same architecture from random
    initialisation, which is what attributes a gain to pretraining rather than
    to architecture or capacity. The default is the pretrained weights.
    """
    return os.environ.get("RS_EVAL_INIT", "pretrained").strip().lower() == "random"
