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


def test_frozen_probe_trains_only_the_head() -> None:
    """B2U is a probe of Prithvi's representation, not a partial fine-tune."""
    model = build(freeze=True)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert not any(p.requires_grad for p in model.adapter.parameters())
    assert any(p.requires_grad for p in model.head.parameters())
    loss = model(torch.randn(1, 3, 256, 256), (256, 256)).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters())
    assert all(p.grad is None for p in model.encoder.parameters())


def test_the_fixed_mapping_puts_rgb_in_the_hls_slots() -> None:
    from prithvi_worker.t9 import ENCODER_BANDS, fixed_band_mapping

    mapping = fixed_band_mapping()
    assert mapping.shape == (len(ENCODER_BANDS), 3)
    for slot, band in enumerate(ENCODER_BANDS):
        expected = {"RED": [1.0, 0.0, 0.0], "GREEN": [0.0, 1.0, 0.0], "BLUE": [0.0, 0.0, 1.0]}
        assert mapping[slot].tolist() == expected.get(band, [0.0, 0.0, 0.0])

    # and the module really applies it: a pixel with R=1 lands only in RED
    model = build(freeze=True).eval()
    pixels = torch.zeros(1, 3, 32, 32)
    pixels[:, 0] = 1.0
    mapped = model.adapter(pixels)[0, :, 0, 0]
    red = ENCODER_BANDS.index("RED")
    assert mapped[red].item() == pytest.approx(1.0)
    assert mapped.sum().item() == pytest.approx(1.0)
