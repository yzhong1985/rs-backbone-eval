"""Tests for the shared objectives and the training-budget identity axis."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

from rs_backbone_eval.contracts import RunRecord  # noqa: E402
from rs_backbone_eval.losses import (  # noqa: E402
    FocalLoss,
    WeightedCrossEntropy,
    inverse_frequency_weights,
)
from rs_backbone_eval.matrix import build_matrix  # noqa: E402
from tests.test_scorecard import payload  # noqa: E402


def test_inverse_frequency_weights_are_normalised_over_present_classes() -> None:
    weights = inverse_frequency_weights([100, 10, 0])
    # 1/f over the two present classes (1/100 and 1/10), rescaled so their mean
    # is 1; the absent class gets weight 0.
    assert float(weights[0]) == pytest.approx(2 / 11)
    assert float(weights[1]) == pytest.approx(20 / 11)
    assert float(weights[2]) == 0.0
    assert float(weights[[0, 1]].mean()) == pytest.approx(1.0)


def test_inverse_frequency_weights_are_zero_when_no_class_occurs() -> None:
    assert torch.equal(inverse_frequency_weights([0, 0]), torch.zeros(2))


def test_focal_loss_with_gamma_zero_is_cross_entropy_on_valid_pixels() -> None:
    logits = torch.tensor(
        [
            [[[2.0]], [[0.0]]],  # pixel (0,0): class 0 strongly favoured
            [[[0.0]], [[3.0]]],  # pixel (0,1) and (1,0) handled below
        ]
    )
    # [N=2, C=2, H=1, W=1] with targets favouring the arg-max class.
    target = torch.tensor([[[0]], [[1]]])
    focal = FocalLoss(gamma=0.0, ignore_index=-1)(logits, target)
    expected = F.cross_entropy(logits, target)
    assert float(focal) == pytest.approx(float(expected), rel=1e-6)


def test_focal_loss_ignores_invalid_pixels() -> None:
    logits = torch.tensor([[[[5.0]], [[-5.0]]], [[[2.0]], [[-2.0]]]])
    target = torch.tensor([[[0]], [[-1]]])
    loss = FocalLoss(gamma=2.0, ignore_index=-1)(logits, target)
    # Only the first pixel counts, and its probability is already ~1, so the
    # result is small but strictly positive.
    assert 0.0 < float(loss) < 1e-3


def test_focal_loss_raises_when_every_pixel_is_ignored() -> None:
    with pytest.raises(ValueError, match="no valid pixels"):
        FocalLoss(ignore_index=-1)(torch.zeros(1, 2, 1, 1), torch.full((1, 1, 1), -1))


def test_training_budget_is_a_variant_axis_not_an_average() -> None:
    """A 150-epoch run must never be averaged with a 50-epoch one."""
    short = [
        RunRecord.model_validate(
            payload(
                run_id=f"short-s{seed}",
                model_id="m",
                seed=seed,
                value=0.5,
                tags={"epochs_budget": "50"},
            )
        )
        for seed in (17, 42, 73)
    ]
    long = [
        RunRecord.model_validate(
            payload(
                run_id=f"long-s{seed}",
                model_id="m",
                seed=seed,
                value=0.7,
                tags={"epochs_budget": "150"},
            )
        )
        for seed in (17, 42, 73)
    ]
    matrix = build_matrix(short + long, {})
    assert len(matrix["cells"]) == 1
    cell = matrix["cells"][0]
    # Both budgets are recorded as competing variants; the headline is one of
    # them at n=3, never a six-seed blend of the two.
    assert cell["variants"] == 2
    assert cell["n"] == 3
    assert set(cell["runs"]) <= {f"short-s{seed}" for seed in (17, 42, 73)} | {
        f"long-s{seed}" for seed in (17, 42, 73)
    }
    picked = {"short" if run.startswith("short") else "long" for run in cell["runs"]}
    assert len(picked) == 1


def test_weighted_cross_entropy_normalises_by_pixels_not_weight_sum() -> None:
    """Review P1: identical inputs and weights must give identical gradients."""
    logits = torch.tensor(
        [
            [[[2.0, -1.0]], [[0.5, 0.5]], [[-2.0, 3.0]]],
            [[[0.0, 1.0]], [[1.0, 0.0]], [[0.0, 0.0]]],
        ]
    )
    target = torch.tensor([[[0, 2]], [[1, 0]]])
    weight = torch.tensor([0.115, 1.0, 2.874])

    ours = WeightedCrossEntropy(weight=weight, reduction="mean")(logits, target)
    per_pixel = F.cross_entropy(logits, target, weight=weight, reduction="none")
    expected = per_pixel.sum() / target.numel()
    assert float(ours) == pytest.approx(float(expected), rel=1e-6)

    # torch's weighted mean divides by the sum of the target-class weights,
    # which is exactly the normalisation this class replaces.
    torch_mean = F.cross_entropy(logits, target, weight=weight)
    assert float(torch_mean) != pytest.approx(float(ours), rel=1e-3)

    summed = WeightedCrossEntropy(weight=weight, reduction="sum")(logits, target)
    assert float(summed) / target.numel() == pytest.approx(float(ours), rel=1e-6)


def test_weighted_cross_entropy_drops_ignored_pixels() -> None:
    logits = torch.tensor([[[[5.0]], [[-5.0]]], [[[0.0]], [[0.0]]]])
    target = torch.tensor([[[0]], [[-1]]])
    weight = torch.tensor([0.5, 2.0])
    loss = WeightedCrossEntropy(weight=weight, ignore_index=-1)(logits, target)
    single = F.cross_entropy(
        logits[:1], target[:1], weight=weight, ignore_index=-1, reduction="none"
    )
    assert float(loss) == pytest.approx(float(single.sum()), rel=1e-6)


def test_every_t3_worker_uses_the_shared_weighted_objective() -> None:
    """Review P1: three workers, one normalisation."""
    root = Path(__file__).resolve().parents[1]
    for worker in ("anysat", "clay", "prithvi"):
        source = (root / "workers" / worker / "src" / f"{worker}_worker" / "t3.py").read_text(
            encoding="utf-8"
        )
        assert "WeightedCrossEntropy" in source
        assert "CrossEntropyLoss(ignore_index=IGNORE_INDEX, weight=" not in source
