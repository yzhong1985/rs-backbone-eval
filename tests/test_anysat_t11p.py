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
    assert protocols["anysat_t11p_distill_o128"].panel == "diagnostic"
    assert all(
        spec.panel == "primary"
        for protocol_id, spec in protocols.items()
        if protocol_id != "anysat_t11p_distill_o128"
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
        "abc", "def", "oe128", 17, init="pretrained", normalization="norm-a", epochs=8
    )
    assert (
        t11p._fingerprint(
            "abc", "def", "oe128", 17, init="pretrained", normalization="norm-a", epochs=8
        )
        == first
    )
    assert (
        t11p._fingerprint(
            "abc", "def", "oe128", 17, init="random", normalization="norm-a", epochs=8
        )
        != first
    )
    assert (
        t11p._fingerprint(
            "abc", "def", "oe128", 17, init="pretrained", normalization="norm-b", epochs=8
        )
        != first
    )
    assert (
        t11p._fingerprint(
            "abc", "def", "oe128", 42, init="pretrained", normalization="norm-a", epochs=8
        )
        != first
    )
    # A one-epoch smoke adapter must never be picked up by a formal probe.
    assert (
        t11p._fingerprint(
            "abc", "def", "oe128", 17, init="pretrained", normalization="norm-a", epochs=1
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
