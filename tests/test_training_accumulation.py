"""Tests for gradient accumulation (workers' effective-batch equivalence)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rs_backbone_eval.training import scale_accumulated_gradients  # noqa: E402


def _full_batch_grad(x, y, *, micro_batch: int, accum: int):
    """Gradient from ``micro_batch`` x ``accum`` with the shared scaling."""
    model = torch.nn.Linear(x.shape[-1], 3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
    params = list(model.parameters())
    model.zero_grad(set_to_none=True)
    valid_total = 0
    for start in range(0, x.shape[0], micro_batch):
        xb, yb = x[start : start + micro_batch], y[start : start + micro_batch]
        logits = model(xb).permute(0, 2, 1)  # [B, C, N]
        loss = criterion(logits, yb)
        loss.backward()
        valid_total += int((yb != -1).sum())
    scale_accumulated_gradients(params, valid_total)
    return [p.grad.detach().clone() for p in params]


def _reference_grad(x, y):
    model = torch.nn.Linear(x.shape[-1], 3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
    params = list(model.parameters())
    model.zero_grad(set_to_none=True)
    logits = model(x).permute(0, 2, 1)
    criterion(logits, y).backward()
    return [p.grad.detach().clone() for p in params]


def test_accumulation_equals_full_batch_with_uneven_valid_pixels() -> None:
    torch.manual_seed(0)
    # two samples with different numbers of ignored pixels
    x = torch.randn(2, 5, 4)
    y = torch.tensor([[0, 1, 1, -1, -1], [0, 0, -1, -1, -1]])
    reference = _reference_grad(x, y)
    accumulated = _full_batch_grad(x, y, micro_batch=1, accum=2)
    for got, want in zip(accumulated, reference, strict=True):
        assert torch.allclose(got, want, atol=1e-6), (got, want)


def test_short_final_window_is_scaled_by_its_own_count() -> None:
    torch.manual_seed(1)
    # three samples, micro-batch 1: the last window is a single sample
    x = torch.randn(3, 4, 4)
    y = torch.tensor([[0, 1, -1, -1], [1, 1, 0, -1], [2, 2, 2, 2]])
    reference = _reference_grad(x, y)
    accumulated = _full_batch_grad(x, y, micro_batch=1, accum=2)
    for got, want in zip(accumulated, reference, strict=True):
        assert torch.allclose(got, want, atol=1e-6), (got, want)


def test_scaling_is_a_noop_without_valid_pixels() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3))
    parameter.grad = torch.ones(3)
    scale_accumulated_gradients([parameter], 0)
    assert torch.allclose(parameter.grad, torch.ones(3))
