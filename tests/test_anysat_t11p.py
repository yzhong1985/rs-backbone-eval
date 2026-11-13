"""CPU-only contracts for the AnySat T11P (six-band) worker.

The heavy probe/distill paths need the real checkpoint and PASTIS data; these
tests pin the declarations that make the runs comparable (registry protocols,
declared block budget, teacher decoding, projector fingerprint) and exercise
the module wiring with a stubbed backbone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import anysat_worker.t11p as t11p  # noqa: E402
import torch  # noqa: E402

from rs_backbone_eval.registry import Registry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_registry_declares_the_six_anysat_t11p_protocols() -> None:
    registry = Registry(ROOT / "configs")
    protocols = {
        spec.protocol_id: spec
        for spec in registry.protocols()
        if spec.protocol_id.startswith("anysat_t11p_")
    }
    declared = set(t11p.DISTILL_PROTOCOLS) | set(t11p.PROBE_PROTOCOLS) | set(t11p.B_STAGE_PROTOCOLS)
    assert set(protocols) == declared
    for protocol_id, spec in protocols.items():
        assert spec.tasks == ["T11P"]
        assert spec.workers == ["anysat"]
        assert spec.epochs == 8
        assert spec.seeds == [17, 42, 73]
        frozen = protocol_id in (set(t11p.DISTILL_PROTOCOLS) | set(t11p.PROBE_PROTOCOLS))
        assert spec.train_backbone is (not frozen)
    diagnostic = {
        "anysat_t11p_distill_o128",
        "anysat_t11p_distill_o128_mse",
        "anysat_t11p_distill_o128_nce",
        "anysat_t11p_probe_o128_mse",
        "anysat_t11p_probe_pca128",
        "anysat_t11p_probe_o128_nce",
    }
    assert all(protocols[pid].panel == "diagnostic" for pid in diagnostic)
    assert all(
        spec.panel == "primary"
        for protocol_id, spec in protocols.items()
        if protocol_id not in diagnostic
    )


def test_b_stage_budget_is_declared_as_three_of_six_blocks() -> None:
    assert t11p.B_STAGE_BLOCKS == 3
    assert t11p.B_STAGE_BLOCKS_TOTAL == 6
    assert "anysat_t11p_ft_last3" in t11p.B_STAGE_PROTOCOLS
    assert "anysat_t11p_ft_last4" not in t11p.B_STAGE_PROTOCOLS
    assert set(t11p.B_STAGE_VARIANTS) == set(t11p.B_STAGE_PROTOCOLS)
    assert t11p.B_STAGE_VARIANTS["anysat_t11p_ft_last3_wide"] == (3, 1536, False)
    assert t11p.B_STAGE_VARIANTS["anysat_t11p_ft_last3"] == (3, 128, True)


def test_teacher_decode_matches_the_declared_quantisation() -> None:
    raw = np.array([[[127, -128], [0, 64]]], dtype=np.int8)
    valid = np.array([[True, False], [True, True]])

    decoded = t11p.decode_teacher(raw, valid)

    assert decoded[0, 0, 0] == pytest.approx((127 / 127.5) ** 2)
    assert decoded[0, 0, 1] == 0.0
    assert decoded[0, 1, 0] == 0.0
    assert decoded[0, 1, 1] == pytest.approx((64 / 127.5) ** 2)


def test_projector_fingerprint_separates_init_and_normalisation() -> None:
    first = t11p._fingerprint(
        "abc",
        "def",
        "oe128",
        17,
        init="pretrained",
        normalization="norm-a",
        epochs=8,
        budget="e8-n1455-l0-s0",
    )
    assert (
        t11p._fingerprint(
            "abc",
            "def",
            "oe128",
            17,
            init="pretrained",
            normalization="norm-a",
            epochs=8,
            budget="e8-n1455-l0-s0",
        )
        == first
    )
    assert (
        t11p._fingerprint(
            "abc",
            "def",
            "oe128",
            17,
            init="random",
            normalization="norm-a",
            epochs=8,
            budget="e8-n1455-l0-s0",
        )
        != first
    )
    assert (
        t11p._fingerprint(
            "abc",
            "def",
            "oe128",
            17,
            init="pretrained",
            normalization="norm-b",
            epochs=8,
            budget="e8-n1455-l0-s0",
        )
        != first
    )
    assert (
        t11p._fingerprint(
            "abc",
            "def",
            "oe128",
            42,
            init="pretrained",
            normalization="norm-a",
            epochs=8,
            budget="e8-n1455-l0-s0",
        )
        != first
    )
    # A one-epoch smoke adapter must never be picked up by a formal probe.
    assert (
        t11p._fingerprint(
            "abc",
            "def",
            "oe128",
            17,
            init="pretrained",
            normalization="norm-a",
            epochs=1,
            budget="e1-n4-l1-s1",
        )
        != first
    )


class _StubBackbone(torch.nn.Module):
    """Minimal AnySat-shaped module: dense output plus six fake spatial blocks."""

    def __init__(self, dim: int = 1536, blocks: int = 6) -> None:
        super().__init__()
        self.dim = dim
        self.model = torch.nn.Module()
        encoder = torch.nn.Module()
        encoder.predictor_blocks = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2) for _ in range(blocks)]
        )
        encoder.predictor_norm = torch.nn.LayerNorm(2)
        self.model.spatial_encoder = encoder

    def forward(self, payload, patch_size=None, output=None, output_modality=None):
        s2 = payload["s2"]
        batch, _, _, height, width = s2.shape
        return torch.zeros(batch, height, width, self.dim, device=s2.device)


def test_module_wiring_shapes_and_declared_unfreezing() -> None:
    backbone = _StubBackbone()
    model = t11p.AnySatT11P(
        backbone,
        t11p.SpectralAdapter(t11p.BANDS),
        projector=t11p.build_projector(t11p.B_STAGE_DIMS),
        dims=t11p.B_STAGE_DIMS,
        blocks=t11p.B_STAGE_BLOCKS,
    )
    images = torch.zeros(1, t11p.MAX_FRAMES, len(t11p.BANDS), 32, 32)
    dates = torch.zeros(1, t11p.MAX_FRAMES)
    logits, projected = model.forward_with_projection(images, dates)
    assert logits.shape == (1, t11p.NUM_CLASSES, t11p.LABEL_SIZE, t11p.LABEL_SIZE)
    assert projected.shape == (1, t11p.B_STAGE_DIMS, 4, 4)

    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    unfrozen = t11p._unfreeze_last_blocks(backbone, t11p.B_STAGE_BLOCKS)
    assert unfrozen > 0
    blocks = backbone.model.spatial_encoder.predictor_blocks
    assert [any(p.requires_grad for p in block.parameters()) for block in blocks] == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert all(p.requires_grad for p in backbone.model.spatial_encoder.predictor_norm.parameters())


def test_adapter_flags_match_the_shared_adapter_contract() -> None:
    """Only the raw probe trains the 6->10 adapter; the rest load and freeze it."""
    registry = Registry(ROOT / "configs")
    protocols = {
        spec.protocol_id: spec
        for spec in registry.protocols()
        if spec.protocol_id.startswith("anysat_t11p_")
    }
    assert protocols["anysat_t11p_probe_raw"].train_input_adapter is True
    assert protocols["anysat_t11p_probe_s128"].train_input_adapter is False
    assert protocols["anysat_t11p_probe_o128"].train_input_adapter is False
    assert protocols["anysat_t11p_distill_o128"].train_input_adapter is False
    for protocol_id in t11p.B_STAGE_PROTOCOLS:
        assert protocols[protocol_id].train_input_adapter is True


def test_epoch_order_is_seeded_and_reproducible() -> None:
    first = t11p._epoch_order(10, 17, 0)
    assert first == t11p._epoch_order(10, 17, 0)
    assert first != t11p._epoch_order(10, 42, 0)
    assert first != t11p._epoch_order(10, 17, 1)


def test_fingerprint_separates_training_budgets() -> None:
    formal = t11p._fingerprint(
        "abc",
        "def",
        "oe128",
        17,
        init="pretrained",
        normalization="norm-a",
        epochs=8,
        budget="e8-n1455-l0-s0",
    )
    limited = t11p._fingerprint(
        "abc",
        "def",
        "oe128",
        17,
        init="pretrained",
        normalization="norm-a",
        epochs=8,
        budget="e8-n4-l1-s1",
    )
    assert formal != limited


def test_norm_cache_key_separates_train_subsets() -> None:
    full = [{"sample_id": str(index)} for index in range(6)]
    subset = full[:2]
    assert t11p._norm_cache_key("a" * 64, full) == t11p._norm_cache_key("a" * 64, full)
    assert t11p._norm_cache_key("a" * 64, full) != t11p._norm_cache_key("a" * 64, subset)


def test_probe_payload_contains_head_and_projector() -> None:
    model = t11p.AnySatT11P(
        _StubBackbone(),
        t11p.SpectralAdapter(t11p.BANDS),
        projector=t11p.build_projector(128),
        dims=128,
    )
    payload = t11p._probe_payload(
        model,
        model.projector,
        dims=128,
        protocol_id="anysat_t11p_probe_s128",
        kind="supervised",
        seed=17,
        adapter_fingerprint="abc",
        adapter_source="/tmp/adapter.pt",
        normalization={"mean": "0", "std": "1"},
        checkpoint_sha256="c",
        manifest_sha256="m",
    )
    keys = payload["state_dict"]
    assert any(key.startswith("head.") for key in keys)
    assert any(key.startswith("projector.") for key in keys)
    assert payload["projector_trainable"] is True


def test_masked_mse_only_scores_observed_tokens_and_unit_identity() -> None:
    prediction = torch.zeros(1, 2, 2, 2)
    target = torch.ones(1, 2, 2, 2)
    prediction[0, :, 0, 1] = 100.0
    mask = torch.tensor([[[True, False], [False, False]]])
    assert t11p._masked_mse(prediction, target, mask).item() == pytest.approx(1.0)

    generator = torch.Generator().manual_seed(0)
    first = torch.nn.functional.normalize(torch.randn(1, 8, 4, 4, generator=generator), dim=1)
    second = torch.nn.functional.normalize(torch.randn(1, 8, 4, 4, generator=generator), dim=1)
    mask = torch.ones(1, 4, 4, dtype=torch.bool)
    mse = t11p._masked_mse(first, second, mask)
    cosine = t11p._cosine(first, second, mask)
    assert mse.item() == pytest.approx((2.0 - 2.0 * cosine.item()) / 8, rel=1e-5)


def test_anysat_t11p_mse_and_pca_protocols_are_registered() -> None:
    assert t11p.DISTILL_LOSS == {
        "anysat_t11p_distill_o128_mse": "mse",
        "anysat_t11p_distill_o128_nce": "infonce",
    }
    assert t11p.DISTILL_PROTOCOLS["anysat_t11p_distill_o128_mse"][0] == "oe128mse"
    assert t11p.PROBE_PROTOCOLS["anysat_t11p_probe_o128_mse"] == (
        "distilled",
        "oe128mse",
    )
    assert t11p.PROBE_PROTOCOLS["anysat_t11p_probe_pca128"] == ("pca128", 128)
    assert t11p.PROBE_PROTOCOLS["anysat_t11p_probe_o128_nce"] == (
        "distilled",
        "oe128nce",
    )
    assert t11p._loss_improved(0.31, 0.30, "cosine")
    assert t11p._loss_improved(0.01, 0.02, "mse")
    # InfoNCE is a loss: selecting the best epoch means minimising it, so a
    # run must not sit at best_epoch=0 with NaN tags (2026-11-13).
    assert t11p._loss_improved(4.1, float("inf"), "infonce")
    assert t11p._loss_improved(4.1, 4.2, "infonce")
    assert not t11p._loss_improved(4.3, 4.2, "infonce")
    assert not t11p._loss_improved(4.1, 4.2, "cosine")
