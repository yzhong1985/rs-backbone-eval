"""CPU tests for the four-band full fine-tune (whu_ft_b0_4band): zero-initialised NIR input.

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_t13_4band.py -q
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from lim_worker.t13 import Inputs, expand_patch_embed  # noqa: E402


def test_expanded_patch_embed_keeps_rgb_and_zeroes_nir() -> None:
    gen = torch.Generator().manual_seed(0)
    state = {
        "patch_embed1.proj.weight": torch.randn(8, 3, 7, 7, generator=gen),
        "patch_embed1.proj.bias": torch.randn(8, generator=gen),
        "block1.weight": torch.randn(4, 4, generator=gen),
    }
    out = expand_patch_embed(state)
    w = out["patch_embed1.proj.weight"]
    assert w.shape == (8, 4, 7, 7)
    assert torch.equal(w[:, :3], state["patch_embed1.proj.weight"])
    assert torch.count_nonzero(w[:, 3]) == 0
    assert torch.equal(out["patch_embed1.proj.bias"], state["patch_embed1.proj.bias"])
    assert torch.equal(out["block1.weight"], state["block1.weight"])
    assert set(out) == set(state)


class _Stub(torch.nn.Module):
    def forward(self, x):
        return [x]


def test_four_band_inputs_normalise_nir_with_its_own_statistics() -> None:
    mean = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    std = np.array([2.0, 4.0, 5.0, 8.0], dtype=np.float32)
    inputs = Inputs(_Stub(), mean, std, "cpu", finetune=False, in_bands=4)
    image = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4) * 3.0
    feats = inputs.features(image)[0]
    assert feats.shape == (1, 4, 4, 4)
    nir_expected = (image[:, 3:4] - 40.0) / 8.0
    assert torch.allclose(feats[:, 3:4], nir_expected, atol=1e-5)
    rgb_only = Inputs(_Stub(), mean, std, "cpu", finetune=False, in_bands=3).features(image)[0]
    assert rgb_only.shape == (1, 3, 4, 4)
    assert torch.allclose(feats[:, :3], rgb_only, atol=1e-6)
