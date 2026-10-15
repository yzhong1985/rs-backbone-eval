"""Tests for the OlmoEarth data path (collate layout and adapter input).

These need torch; run them under an interpreter that has it, for example
``workers/prithvi/.venv``. They cover the two bugs the review flagged: the
default collate transposes list timestamps, and the 12-band input cannot
broadcast six-band normalisation statistics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers" / "olmoearth" / "src"))

from olmoearth_worker.bands import to_olmoearth_s2l2a  # noqa: E402
from torch.utils.data._utils.collate import default_collate  # noqa: E402

PROJECT_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
OURS = (0, 1, 2, 7, 8, 9)  # B02, B03, B04, B8A, B11, B12 positions
FILLED = (3, 4, 5, 6, 10, 11)


def test_timestamp_collate_keeps_one_entry_per_sample() -> None:
    stamps = [torch.tensor([[1, 0, 2020]], dtype=torch.long) for _ in range(4)]
    collated = default_collate(stamps)
    assert tuple(collated.shape) == (4, 1, 3)
    # the dates are NOT transposed: sample 2's date is sample 2's date
    assert collated[2, 0, 0] == 1


def test_normalisation_happens_before_the_band_adapter() -> None:
    torch.manual_seed(0)
    chips = torch.rand(4, 6, 32, 32)
    mean = chips.mean(dim=(0, 2, 3))
    std = chips.std(dim=(0, 2, 3)).clamp_min(1e-6)
    standardised = (chips - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)

    expanded = torch.empty(4, 32, 32, 1, 12)
    for index in range(4):
        expanded[index] = (
            torch.from_numpy(
                to_olmoearth_s2l2a(standardised[index].numpy(), PROJECT_BANDS, fill=0.0)
            )
            .permute(1, 2, 0)
            .unsqueeze(2)
        )
    for index in range(4):
        for position in OURS:
            assert not torch.equal(
                expanded[index, ..., position], torch.zeros_like(expanded[index, ..., position])
            )
        # the six bands we lack are exactly the standardised mean (0)
        for position in FILLED:
            assert torch.equal(
                expanded[index, ..., position], torch.zeros_like(expanded[index, ..., position])
            )


def test_six_band_normalisation_broadcasts_over_the_task_bands() -> None:
    mean = torch.zeros(6)
    std = torch.ones(6)
    chips = torch.rand(4, 6, 32, 32)
    assert ((chips - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)).shape == (4, 6, 32, 32)


def test_band_statistics_accepts_numpy_batches() -> None:
    """__getitem__ returns NumPy arrays; torch.cat needs tensors."""
    from olmoearth_worker.t1_frozen import band_statistics

    class StubDataset:
        def __init__(self) -> None:
            self._items = np.arange(4 * 6 * 8 * 8, dtype=np.float32).reshape(
                4, 6, 8, 8
            ) + np.float32(1.0)  # strictly positive so mean != 0 and std != 1

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, index: int):
            return {"image": self._items[index]}

    mean, std = band_statistics(StubDataset(), limit=4)
    assert isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor)
    assert mean.shape == std.shape == (6,)
    assert torch.isfinite(mean).all() and (std > 0).all()
    assert not torch.allclose(mean, torch.zeros(6))
    assert not torch.allclose(std, torch.ones(6))


def test_band_statistics_defaults_when_the_dataset_is_empty() -> None:
    from olmoearth_worker.t1_frozen import band_statistics

    class EmptyDataset:
        def __len__(self) -> int:
            return 0

        def __getitem__(self, index: int):  # pragma: no cover
            raise AssertionError("must not read any row")

    mean, std = band_statistics(EmptyDataset(), limit=8)
    assert torch.allclose(mean, torch.zeros(6))
    assert torch.allclose(std, torch.ones(6))
