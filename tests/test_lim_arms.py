"""CPU tests for the T13 image-only arms (plan v3.4 G0a, ported to the repo).

Run in the LIM worker env (the root env has no torch and skips this file)::

    uv run --project workers/lim --with pytest python -m pytest tests/test_lim_arms.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from lim_worker.arms import (  # noqa: E402
    AEF_NODATA,
    ArmModel,
    alignment_loss,
    decode_aef,
    distillation_loss,
    pool_teacher,
    safe_cross_entropy,
)

CHANNELS = [256, 512, 1024, 2048]  # real MiT-Huge pyramid
CLASSES = 8
IGNORE = 0


def pyramid(batch: int = 2, size: int = 64, seed: int = 0) -> list[torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return [
        torch.randn(batch, c, size // s, size // s, generator=gen)
        for c, s in zip(CHANNELS, (4, 8, 16, 32), strict=True)
    ]


def image(batch: int = 2, size: int = 64, bands: int = 4, seed: int = 1) -> torch.Tensor:
    return torch.randn(batch, bands, size, size, generator=torch.Generator().manual_seed(seed))


def labels(batch: int = 2, size: int = 64, seed: int = 2) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(1, CLASSES, (batch, size, size), generator=gen)


def model(arm: str, **plugin) -> ArmModel:
    torch.manual_seed(0)
    return ArmModel(arm, CHANNELS, CLASSES, plugin=plugin or None)


def grad_sum(module: torch.nn.Module) -> float:
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


# -- G0a-1: decoding, pooling and degenerate losses --------------------------


def test_decode_handles_signs_zero_nodata_and_rejects_floats() -> None:
    codes = torch.full((1, 64, 1, 4), 0, dtype=torch.int8)
    codes[0, :, 0, 0] = 127
    codes[0, :, 0, 1] = -127
    codes[0, 0, 0, 3] = AEF_NODATA  # one band of NoData poisons the cell
    features, valid = decode_aef(codes)
    assert features[0, 0, 0, 0] == pytest.approx((127 / 127.5) ** 2)
    assert features[0, 0, 0, 1] == pytest.approx(-((127 / 127.5) ** 2))
    assert valid[0, 0, 0].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert torch.all(features[0, :, 0, 3] == 0)
    with pytest.raises(TypeError):
        decode_aef(features)


def test_pool_and_losses_never_produce_nan() -> None:
    teacher = torch.randn(2, 64, 32, 32)
    teacher[0, 0, 0, 0] = float("nan")
    teacher[0, 1, 0, 1] = float("inf")
    valid = torch.ones(2, 1, 32, 32)
    pooled, pooled_valid = pool_teacher(teacher, valid, (8, 8))
    assert torch.isfinite(pooled).all() and pooled_valid.sum() == 2 * 64

    empty, empty_valid = pool_teacher(teacher, torch.zeros(2, 1, 32, 32), (8, 8))
    pred = torch.randn(2, 64, 8, 8, requires_grad=True)
    loss = alignment_loss(pred, empty, empty_valid)
    loss.backward()
    assert loss.item() == 0.0 and pred.grad is not None

    logits = torch.randn(2, CLASSES, 8, 8, requires_grad=True)
    all_ignored = torch.zeros(2, 8, 8, dtype=torch.long)
    assert safe_cross_entropy(logits, all_ignored, IGNORE).item() == 0.0
    assert distillation_loss(logits, torch.randn_like(logits), all_ignored, IGNORE).item() == 0.0


# -- zero init: every add-on arm starts as the B0 head ----------------------


@pytest.mark.parametrize("arm", ["R", "Bcap", "P"])
def test_addon_arms_start_identical_to_b0(arm: str) -> None:
    feats, img = pyramid(), image()
    base, other = model("B0").eval(), model(arm).eval()
    other.head.load_state_dict(base.head.state_dict())
    aef = torch.randn(2, 64, 16, 16) if arm == "R" else None
    with torch.no_grad():
        assert torch.equal(base(feats, img)["logits"], other(feats, img, aef)["logits"])


# -- G0a-3: gradient routing -------------------------------------------------


def _ce_step(m: ArmModel, opt, feats, img, lbl) -> None:
    opt.zero_grad()
    safe_cross_entropy(m(feats, img)["logits"], lbl, IGNORE).backward()


def test_deep_path_needs_two_steps_and_nir_query_three() -> None:
    feats, img, lbl = pyramid(), image(), labels()
    m = model("Bcap", shallow_levels=(), nir_drop_prob=0.0).train()
    opt = torch.optim.SGD(m.configure_trainable(train_head=False), lr=0.1)
    plugin = m.plugin

    _ce_step(m, opt, feats, img, lbl)  # step 1: only the zero-init injector output layer
    assert grad_sum(plugin.inject["2"].net[-1]) > 0
    assert grad_sum(plugin.query["2"]) == 0 and grad_sum(plugin.nir_stem) == 0
    opt.step()

    _ce_step(m, opt, feats, img, lbl)  # step 2: query head and the NIR-residual output layer
    assert grad_sum(plugin.query["2"]) > 0 and grad_sum(plugin.query_nir["2"][-1]) > 0
    assert grad_sum(plugin.nir_stem) == 0
    opt.step()

    _ce_step(m, opt, feats, img, lbl)  # step 3: through both zero-init layers to the stem
    assert grad_sum(plugin.nir_stem[0]) > 0


def test_shallow_nir_path_reaches_the_stem_on_step_two() -> None:
    feats, img, lbl = pyramid(), image(), labels()
    m = model("Bcap", deep_levels=(), nir_drop_prob=0.0).train()
    opt = torch.optim.SGD(m.configure_trainable(train_head=False), lr=0.1)
    _ce_step(m, opt, feats, img, lbl)
    assert grad_sum(m.plugin.shallow_nir["0"]) > 0 and grad_sum(m.plugin.nir_stem) == 0
    opt.step()
    _ce_step(m, opt, feats, img, lbl)
    assert grad_sum(m.plugin.nir_stem[0]) > 0


def test_alignment_loss_reaches_the_query_immediately() -> None:
    feats, img = pyramid(), image(bands=3)
    m = model("P").train()
    m.configure_trainable(train_head=False)
    out = m(feats, img)
    target, valid = pool_teacher(torch.randn(2, 64, 16, 16), torch.ones(2, 1, 16, 16), (4, 4))
    alignment_loss(out["proxy_rgb"][0], target, valid).backward()
    assert grad_sum(m.plugin.query["2"]) > 0


# -- configure_trainable and the locked head ---------------------------------


def test_plugin_training_leaves_the_head_bit_identical() -> None:
    feats, img, lbl = pyramid(), image(), labels()
    m = model("Bcap").train()
    trainable = m.configure_trainable(train_head=False)
    assert not any(p.requires_grad for p in m.head.parameters())
    assert not m.plugin.codebook.requires_grad
    before = {k: v.clone() for k, v in m.head.state_dict().items()}
    opt = torch.optim.AdamW(trainable, lr=1e-2)
    for _ in range(3):
        _ce_step(m, opt, feats, img, lbl)
        opt.step()
    assert all(torch.equal(v, before[k]) for k, v in m.head.state_dict().items())


def test_configure_trainable_is_the_only_switch() -> None:
    m = model("P")
    trainable = m.configure_trainable(train_head=False, freeze_codebook=False)
    assert m.plugin.codebook.requires_grad
    assert {id(p) for p in trainable} == {id(p) for p in m.plugin.parameters()}
    head_only = model("B0").configure_trainable(train_head=True)
    assert len(head_only) == len(list(model("B0").head.parameters()))


# -- G0a-4: codebook ---------------------------------------------------------


def test_kmeans_codebook_is_deterministic_and_needs_enough_vectors() -> None:
    vectors = torch.randn(4000, 64)
    a, b = model("P", num_prototypes=64), model("P", num_prototypes=64)
    a.plugin.init_codebook_kmeans(vectors, seed=17, region="hubei", aef_year=2019)
    b.plugin.init_codebook_kmeans(vectors, seed=17, region="hubei", aef_year=2019)
    assert torch.equal(a.plugin.codebook, b.plugin.codebook)
    assert a.plugin.codebook_metadata["init"] == "kmeans"
    assert torch.allclose(a.plugin.codebook.norm(dim=-1), torch.ones(64), atol=1e-5)
    with pytest.raises(ValueError, match="prototypes"):
        model("P", num_prototypes=512).plugin.init_codebook_kmeans(vectors[:100], seed=17)


def test_one_plugin_serves_three_and_four_bands() -> None:
    feats = pyramid()
    m = model("P", nir_drop_prob=0.0).eval()
    with torch.no_grad():
        rgb = m(feats, image(bands=3))["logits"]
        rgbn = m(feats, image(bands=4))["logits"]
        placeholder = model("P", nir_input_mode="rgb_mean_placeholder").eval()(
            feats, image(bands=4)
        )
    assert rgb.shape == rgbn.shape == (2, CLASSES, 64, 64)
    assert torch.isfinite(placeholder["logits"]).all()
    with pytest.raises(ValueError, match="3- or 4-band"):
        m(feats, image(bands=5))


# -- G0a-5/6: plugin files ----------------------------------------------------


def test_plugin_hot_swap_restores_outputs_and_checks_the_head(tmp_path: Path) -> None:
    feats, img, lbl = pyramid(), image(), labels()
    trained = model("Bcap", num_prototypes=32).train()
    opt = torch.optim.AdamW(trained.configure_trainable(train_head=False), lr=1e-2)
    for _ in range(3):
        _ce_step(trained, opt, feats, img, lbl)
        opt.step()
    trained.eval()
    with torch.no_grad():
        expected = trained(feats, img)["logits"]
    trained.save_plugin(tmp_path / "hubei.pt", backbone_id="lim-sha", region="hubei")

    fresh = model("Bcap").eval()  # default 512 prototypes: the file's config must win
    fresh.head.load_state_dict(trained.head.state_dict())
    meta = fresh.load_plugin(tmp_path / "hubei.pt", backbone_id="lim-sha")
    assert meta == {"region": "hubei"} and fresh.plugin.num_prototypes == 32
    assert not fresh.plugin.training  # hot swap keeps eval mode, NIR dropout stays off
    with torch.no_grad():
        assert torch.allclose(fresh(feats, img)["logits"], expected, atol=1e-6)
        fresh.plugin_enabled = False
        assert not torch.allclose(fresh(feats, img)["logits"], expected, atol=1e-6)

    with pytest.raises(ValueError, match="head hash"):
        fresh.load_plugin(tmp_path / "hubei.pt", backbone_id="another-checkpoint")
    other_head = model("Bcap")
    with torch.no_grad():
        other_head.head.classifier.bias.add_(1.0)  # a different B0 head
    with pytest.raises(ValueError, match="head hash"):
        other_head.load_plugin(tmp_path / "hubei.pt", backbone_id="lim-sha")


# -- G0a-8: deployment contract ----------------------------------------------


@pytest.mark.parametrize("arm", ["B0", "D", "Bcap", "P"])
def test_image_only_arms_reject_aef(arm: str) -> None:
    with pytest.raises(RuntimeError, match="image-only"):
        model(arm)(pyramid(), image(), aef=torch.zeros(2, 64, 16, 16))


def test_reference_arm_requires_aef() -> None:
    with pytest.raises(RuntimeError, match="needs the AEF"):
        model("R")(pyramid(), image())
