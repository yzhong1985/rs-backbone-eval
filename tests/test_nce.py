"""CPU-only tests for the shared symmetric InfoNCE helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rs_backbone_eval.nce import (  # noqa: E402
    info_nce,
    patches_per_step,
    retrieval_top1,
    sample_token_pairs,
)


def test_patch_rule_keeps_regression_losses_at_two() -> None:
    assert patches_per_step("infonce") == 8
    assert patches_per_step("cosine") == 2
    assert patches_per_step("mse") == 2


def test_info_nce_prefers_aligned_pairs_and_is_symmetric() -> None:
    generator = torch.Generator().manual_seed(0)
    teacher = torch.nn.functional.normalize(torch.randn(32, 16, generator=generator), dim=1)
    aligned = info_nce(teacher, teacher)
    permuted = info_nce(teacher, teacher[torch.randperm(32, generator=generator)])
    assert aligned < permuted
    assert torch.allclose(aligned, info_nce(teacher, teacher))
    assert torch.allclose(info_nce(teacher, teacher * 2.0), aligned)


def test_sampling_never_picks_masked_tokens_and_keeps_the_global_rng() -> None:
    predictions = torch.zeros(1, 4, 2, 2)
    target = torch.ones(1, 4, 2, 2)
    mask = torch.tensor([[[False, True], [False, False]]])
    state = torch.random.get_rng_state().clone()
    student, teacher = sample_token_pairs(
        predictions, target, mask, 1024, torch.Generator().manual_seed(1)
    )
    assert student.shape == (1, 4)
    assert torch.equal(student, predictions[:, :, 0, 1])
    assert torch.equal(teacher, target[:, :, 0, 1])
    assert torch.equal(torch.random.get_rng_state(), state)


def test_retrieval_top1_counts_matched_pairs() -> None:
    teacher = torch.eye(4)
    correct, total = retrieval_top1(teacher, teacher)
    assert (correct, total) == (4, 4)
    swapped, _ = retrieval_top1(teacher, teacher[[1, 0, 3, 2]])
    assert swapped == 0
