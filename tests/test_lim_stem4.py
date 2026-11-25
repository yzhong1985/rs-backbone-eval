"""CPU tests for the four-band first layer (LIM worker env).

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_stem4.py -q
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from lim_worker.mit import OverlapPatchEmbed  # noqa: E402
from lim_worker.stem4 import FourBandPatchEmbed, add_nir_stem  # noqa: E402
from torch import nn  # noqa: E402


def _base() -> OverlapPatchEmbed:
    torch.manual_seed(0)
    base = OverlapPatchEmbed(7, 4, 3, 16)
    for param in base.parameters():
        param.requires_grad_(False)
    return base


def _image(seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(2, 4, 64, 64, generator=gen)


def test_zero_nir_weights_reproduce_the_rgb_stem_exactly() -> None:
    base = _base()
    stem = FourBandPatchEmbed(base)
    x = _image()
    four, h4, w4 = stem(x)
    three, h3, w3 = base(x[:, :3])
    assert (h4, w4) == (h3, w3)
    assert torch.equal(four, three)


def test_nir_weights_change_the_output_once_nonzero() -> None:
    base = _base()
    stem = FourBandPatchEmbed(base)
    x = _image()
    before = stem(x)[0].detach().clone()
    with torch.no_grad():
        stem.nir.weight.normal_(0.0, 0.05)
    after = stem(x)[0]
    assert not torch.equal(before, after)


def test_only_the_nir_convolution_receives_gradients() -> None:
    base = _base()
    stem = FourBandPatchEmbed(base)
    tokens, _, _ = stem(_image())
    tokens.square().mean().backward()
    assert stem.nir.weight.grad is not None
    assert bool(stem.nir.weight.grad.abs().sum() > 0)
    assert all(p.grad is None for p in base.parameters())


def test_add_nir_stem_replaces_patch_embed1_in_place() -> None:
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed1 = OverlapPatchEmbed(7, 4, 3, 16)

    encoder = Tiny()
    original = encoder.patch_embed1
    stem = add_nir_stem(encoder)
    assert encoder.patch_embed1 is stem
    assert stem.base is original
    assert stem.nir.in_channels == 1 and stem.nir.out_channels == 16
