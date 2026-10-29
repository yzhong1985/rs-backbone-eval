"""UPerNet head tests (iSAID track, design 2026-10-29 §2.1).

The head is torch-only and GroupNorm by default: terratorch's ``ConvModule``
hard-codes ``nn.BatchNorm2d`` and is installed only in the prithvi worker, so
the shared head cannot reuse it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rs_backbone_eval.heads import (  # noqa: E402
    PyramidPoolingModule,
    UPerNetHead,
    _group_count,
    count_parameters,
)


def pyramid(batch: int = 2, size: int = 64, channels=(16, 32, 64, 128)) -> list:
    levels = []
    for index, width in enumerate(channels):
        side = max(size // (4 * 2**index), 1)
        levels.append(torch.randn(batch, width, side, side))
    return levels


def test_forward_returns_logits_at_the_requested_size() -> None:
    head = UPerNetHead([16, 32, 64, 128], num_classes=16, fpn_dim=32)
    logits = head(pyramid(), output_size=(64, 64))
    assert logits.shape == (2, 16, 64, 64)
    # without output_size the logits stay at the finest level's resolution
    coarse = head(pyramid())
    assert coarse.shape == (2, 16, 16, 16)


def test_single_level_pyramid_still_decodes() -> None:
    head = UPerNetHead(64, num_classes=4, fpn_dim=32)
    logits = head([torch.randn(1, 64, 12, 12)])
    assert logits.shape == (1, 4, 12, 12)


def test_level_count_must_match_the_configured_channels() -> None:
    head = UPerNetHead([16, 32], num_classes=4, fpn_dim=32)
    with pytest.raises(ValueError, match="expected 2 levels"):
        head([torch.randn(1, 16, 8, 8)])


def test_default_normalisation_is_group_norm() -> None:
    head = UPerNetHead([16, 32], num_classes=4, fpn_dim=32)
    norms = [module for module in head.modules() if isinstance(module, torch.nn.GroupNorm)]
    assert norms, "the head must use GroupNorm by default"
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in head.modules())
    # a group count that does not divide the channels would fail at forward time
    for module in norms:
        assert module.num_channels % module.num_groups == 0


def test_group_count_always_divides() -> None:
    for channels in (1, 3, 7, 32, 96, 100, 256):
        assert channels % _group_count(channels) == 0
        assert _group_count(channels) <= 32


def test_norm_layer_is_configurable() -> None:
    head = UPerNetHead(
        [16, 32], num_classes=4, fpn_dim=32, norm_layer=lambda c: torch.nn.BatchNorm2d(c)
    )
    assert any(isinstance(m, torch.nn.BatchNorm2d) for m in head.modules())
    assert not any(isinstance(m, torch.nn.GroupNorm) for m in head.modules())
    logits = head([torch.randn(2, 16, 8, 8), torch.randn(2, 32, 4, 4)])
    assert logits.shape == (2, 4, 8, 8)


def test_ppm_keeps_the_input_resolution() -> None:
    ppm = PyramidPoolingModule(32, pool_bins=(1, 2, 3, 6))
    features = torch.randn(1, 32, 9, 11)
    assert ppm(features).shape == (1, 32, 9, 11)


def test_gradients_reach_every_level_and_the_classifier() -> None:
    head = UPerNetHead([16, 32, 64, 128], num_classes=5, fpn_dim=16)
    # batch 2: a 1x1 coarsest level with batch 1 leaves GroupNorm a single value
    levels = [level.requires_grad_(True) for level in pyramid(batch=2, size=64)]
    logits = head(levels, output_size=(64, 64))
    logits.mean().backward()
    assert head.classifier.weight.grad is not None
    assert all(level.grad is not None and torch.isfinite(level.grad).all() for level in levels)


def test_parameter_count_is_reported() -> None:
    head = UPerNetHead([16, 32], num_classes=4, fpn_dim=32)
    total = count_parameters(head)
    assert total > 0
    head.classifier.requires_grad_(False)
    assert count_parameters(head, trainable_only=True) < total
