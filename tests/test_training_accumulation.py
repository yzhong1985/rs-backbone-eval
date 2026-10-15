"""Tests for gradient accumulation (workers' effective-batch equivalence).

These tests run the accumulation exactly the way the workers do: micro-batches
with ``reduction="sum"``, a step every ``accum`` micro-batches, and one scaling
by the window's valid pixels before each step. They must run under an
interpreter that has torch (for example ``workers/prithvi/.venv``); in the root
environment the whole module is skipped.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rs_backbone_eval.training import scale_accumulated_gradients  # noqa: E402


def _build_model():
    """Same weights for every caller: reseed before creating."""
    torch.manual_seed(0)
    return torch.nn.Linear(4, 3)


def _windowed_run(x, y, *, micro_batch: int, accum: int):
    """The workers' accumulation loop; returns (per-window grads, updates, loss sum)."""
    model = _build_model()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
    params = list(model.parameters())
    windows: list[list] = []
    updates = 0
    loss_sum = 0.0
    pending, window_valid = 0, 0
    model.zero_grad(set_to_none=True)
    for start in range(0, x.shape[0], micro_batch):
        xb, yb = x[start : start + micro_batch], y[start : start + micro_batch]
        loss = criterion(model(xb).permute(0, 2, 1), yb)  # [B, C, N]
        loss.backward()
        loss_sum += float(loss.detach())
        window_valid += int((yb != -1).sum())
        pending += 1
        if pending == accum:
            scale_accumulated_gradients(params, window_valid)
            windows.append([p.grad.clone() for p in params])
            model.zero_grad(set_to_none=True)
            pending, window_valid = 0, 0
            updates += 1
    if pending:  # a short final window is scaled by its own count
        scale_accumulated_gradients(params, window_valid)
        windows.append([p.grad.clone() for p in params])
        model.zero_grad(set_to_none=True)
        updates += 1
    return windows, updates, loss_sum


def _full_batch_grad(x, y):
    """Gradient of one full batch with the default (mean over valid) loss."""
    model = _build_model()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
    model.zero_grad(set_to_none=True)
    criterion(model(x).permute(0, 2, 1), y).backward()
    return [p.grad.clone() for p in model.parameters()]


def test_accumulation_matches_a_full_batch_per_window() -> None:
    # Two samples with different numbers of valid pixels: one full window.
    x = torch.randn(2, 5, 4)
    y = torch.tensor([[0, 1, 1, -1, -1], [0, 0, -1, -1, -1]])
    windows, updates, loss_sum = _windowed_run(x, y, micro_batch=1, accum=2)
    assert updates == 1
    assert len(windows) == 1
    for got, want in zip(windows[0], _full_batch_grad(x, y), strict=True):
        assert torch.allclose(got, want, atol=1e-6), (got, want)
    # the logged loss is the sum over valid pixels, not the mean times steps
    mean = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")(
        _build_model()(x).permute(0, 2, 1), y
    )
    valid = int((y != -1).sum())
    assert abs(loss_sum - float(mean.detach()) * valid) < 1e-3


def test_short_final_window_is_scaled_by_its_own_count() -> None:
    # Three samples with accum=2: a window of two, then a short window of one.
    x = torch.randn(3, 4, 4)
    y = torch.tensor([[0, 1, -1, -1], [1, 1, 0, -1], [2, 2, 2, 2]])
    windows, updates, _ = _windowed_run(x, y, micro_batch=1, accum=2)
    assert updates == 2
    assert len(windows) == 2
    for got, want in zip(windows[0], _full_batch_grad(x[:2], y[:2]), strict=True):
        assert torch.allclose(got, want, atol=1e-6), (got, want)
    for got, want in zip(windows[1], _full_batch_grad(x[2:], y[2:]), strict=True):
        assert torch.allclose(got, want, atol=1e-6), (got, want)


def test_accumulation_updates_parameters_like_a_full_batch() -> None:
    """End-to-end: real optimizer.step() over a 2+1 split must match a full batch."""
    x = torch.randn(3, 4, 4)
    y = torch.tensor([[0, 1, -1, -1], [1, 1, 0, -1], [2, 2, 2, 2]])

    mean = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")
    reference = _build_model()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=1.0)
    updates = 0
    for xb, yb in ((x[:2], y[:2]), (x[2:], y[2:])):
        reference_optimizer.zero_grad(set_to_none=True)
        mean(reference(xb).permute(0, 2, 1), yb).backward()
        reference_optimizer.step()
        updates += 1
    assert updates == 2

    summed = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")
    model = _build_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    params = list(model.parameters())
    pending, valid = 0, 0
    optimizer.zero_grad(set_to_none=True)
    scaled = 0
    for start in range(0, x.shape[0], 1):
        summed(model(x[start : start + 1]).permute(0, 2, 1), y[start : start + 1]).backward()
        pending += 1
        valid += int((y[start : start + 1] != -1).sum())
        if pending == 2:
            scale_accumulated_gradients(params, valid)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending, valid = 0, 0
            scaled += 1
    if pending:  # the short final window
        scale_accumulated_gradients(params, valid)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scaled += 1
    assert scaled == updates

    for got, want in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.allclose(got, want, atol=1e-5), (got, want)


def test_scaling_is_a_noop_without_valid_pixels() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3))
    parameter.grad = torch.ones(3)
    scale_accumulated_gradients([parameter], 0)
    assert torch.allclose(parameter.grad, torch.ones(3))
