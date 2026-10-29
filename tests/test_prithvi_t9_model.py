"""Prithvi T9: 3->6 band adapter, /16 token grid, shared UPerNet head.

Skipped where terratorch is absent; run it in the Prithvi worker environment:

    workers/prithvi/.venv/bin/python -m pytest tests/test_prithvi_t9_model.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("terratorch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers" / "prithvi" / "src"))

from prithvi_worker.t9 import ENCODER_BANDS, PATCH, PrithviSegmentation  # noqa: E402


def build(freeze: bool = True) -> PrithviSegmentation:
    from terratorch.registry import BACKBONE_REGISTRY

    encoder = BACKBONE_REGISTRY.build("prithvi_eo_v2_300_tl", pretrained=False, bands=ENCODER_BANDS)
    return PrithviSegmentation(encoder, num_classes=16, fpn_dim=32, freeze_backbone=freeze)


def test_forward_shape_at_a_small_size() -> None:
    model = build().eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 256, 256), (256, 256))
    assert logits.shape == (1, 16, 256, 256)
    assert torch.isfinite(logits).all()


def test_a_1024_tile_decodes_to_a_64x64_token_grid() -> None:
    """The /16 patch is what makes Prithvi the coarse, away-side model here."""
    model = build().eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 1024, 1024), (1024, 1024))
    assert logits.shape == (1, 16, 1024, 1024)
    assert 1024 // PATCH == 64


def test_frozen_backbone_stays_frozen_but_the_adapter_trains() -> None:
    model = build(freeze=True)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert any(p.requires_grad for p in model.adapter.parameters())
    assert any(p.requires_grad for p in model.head.parameters())
    # the graph must still reach the adapter through the frozen encoder
    loss = model(torch.randn(1, 3, 256, 256), (256, 256)).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.adapter.parameters())
    assert all(p.grad is None for p in model.encoder.parameters())
