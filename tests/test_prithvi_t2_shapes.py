"""T2 (HLS burn scar) shape-contract regression tests.

The frozen training loop up-samples head logits to the *training-crop* label
grid; an earlier version passed the evaluation-size constant (512) instead of
the cropped label size (256), which crashed CrossEntropyLoss during the
frozen-feature training step. These tests keep both invariants explicit and
need torch, like the other worker tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers" / "prithvi" / "src"))

from prithvi_worker.t2 import LABEL_SIZE, PrithviT2  # noqa: E402


class _DummyEncoder(nn.Module):
    """Emits [B, tokens, 1024] at the token count the real ViT would; fp32 params."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, images):
        if images.dtype != torch.float32:
            raise ValueError("the worker feeds float32 chips")
        tokens = (images.shape[-1] // 16) ** 2 + 1  # patch grid + cls token
        return torch.randn(images.shape[0], tokens, 1024)


def _model() -> PrithviT2:
    return PrithviT2(_DummyEncoder(), freeze_backbone=True)


def test_frozen_training_logits_match_the_cropped_labels() -> None:
    model = _model()
    features = torch.randn(8, 1024, 16, 16)  # features of a 256 crop (16x16 grid)
    labels = torch.zeros(8, 256, 256, dtype=torch.int64)
    labels[:, ::64, ::64] = 1  # both classes present
    logits = model(features, size=labels.shape[-1])
    assert logits.shape == (8, 2, 256, 256)
    loss = F.cross_entropy(logits, labels, ignore_index=-1)
    assert torch.isfinite(loss)


def test_upsampling_to_the_wrong_grid_raises() -> None:
    """The pre-fix behaviour (eval-size logits against cropped labels) raises."""
    model = _model()
    features = torch.randn(8, 1024, 16, 16)
    labels = torch.zeros(8, 256, 256, dtype=torch.int64)
    with pytest.raises(RuntimeError):
        F.cross_entropy(model(features, size=LABEL_SIZE), labels, ignore_index=-1)


def test_evaluation_logits_match_the_native_labels() -> None:
    model = _model()
    features = torch.randn(8, 1024, 32, 32)  # native 512 input (32x32 grid)
    labels = torch.zeros(8, 512, 512, dtype=torch.int64)
    logits = model(features, size=LABEL_SIZE)
    assert logits.shape == (8, 2, 512, 512)
    loss = F.cross_entropy(logits, labels, ignore_index=-1)
    assert torch.isfinite(loss)
