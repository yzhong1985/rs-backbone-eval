"""Frozen-feature cache unit tests for the AnySat frozen tracks.

Covers the audit's cache-identity rule (checkpoint/manifest/bands/patch/
dates/preprocessing/torch + worker sources participate), the fp16 save→load
round trip, and populate()'s equivalence property: the cached dense map must
match the direct adapter+backbone path within the fp16 rounding step.
Run in a torch environment (e.g. workers/prithvi/.venv/bin/python -m pytest).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers" / "anysat" / "src"))

from anysat_worker.frozen_cache import CachedSampleDataset, FrozenFeatureCache, populate  # noqa: E402


def identity(**overrides) -> dict[str, object]:
    base: dict[str, object] = {
        "checkpoint_sha256": "abc123",
        "manifest_sha256": "man456",
        "bands": "BLUE,GREEN,RED,NIR_BROAD,SWIR_1,SWIR_2",
        "patch_size_m": 160,
        "time_steps": 3,
        "preprocessing_id": "official_multicrop_stats_native_time_series",
        "torch_version": torch.__version__,
    }
    base.update(overrides)
    return base


def worker_sources(tmp_path: Path) -> list[Path]:
    first = tmp_path / "t5.py"
    first.write_text("version A\n", encoding="utf-8")
    second = tmp_path / "spectral_adapter.py"
    second.write_text("adapter A\n", encoding="utf-8")
    return [first, second]


def make_cache(tmp_path: Path, *, split: str = "train", **overrides) -> FrozenFeatureCache:
    return FrozenFeatureCache(
        cache_root=tmp_path / "cache",
        identity_parts=identity(**overrides),
        worker_sources=worker_sources(tmp_path),
        split=split,
    )


def test_identity_rotation_invalidates_the_cache(tmp_path: Path) -> None:
    first = make_cache(tmp_path)
    first.save(0, torch.zeros(2, 1536, 8, 8), torch.zeros(8, 8, dtype=torch.int64))
    assert first.has(0)

    second = make_cache(tmp_path, checkpoint_sha256="DIFFERENT-weights")
    assert not second.has(0)
    keydirs = list((tmp_path / "cache" / "anysat_t5_features").glob("*"))
    assert len(keydirs) == 2  # old content untouched, new key on its own row


def test_identity_marker_stores_every_required_field(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    recorded = json.loads(cache.marker.read_text(encoding="utf-8"))
    for field in (
        "checkpoint_sha256",
        "manifest_sha256",
        "bands",
        "patch_size_m",
        "time_steps",
        "preprocessing_id",
        "torch_version",
        "sources",
        "split",
    ):
        assert field in recorded


def test_fp16_save_load_round_trip(tmp_path: Path) -> None:
    cache = make_cache(tmp_path)
    dense = torch.randn(2, 8, 16, 16)
    labels = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    cache.save(7, dense, labels)
    block = cache.load(7)
    assert block["dense"].dtype == torch.float32
    assert block["label"].dtype == torch.int64
    assert torch.allclose(block["dense"], dense, atol=1e-3)
    assert torch.equal(block["label"], labels)


def test_state_hash_distinguishes_actual_weights(tmp_path: Path) -> None:
    from anysat_worker.frozen_cache import state_identity_hash

    class _Module(nn.Module):
        def __init__(self, seed: int) -> None:
            super().__init__()
            self.linear = nn.Conv2d(2, 2, kernel_size=1)
            torch.manual_seed(seed)

    model_a = _Module(seed=1)
    model_b = _Module(seed=2)
    hash_a = state_identity_hash(model_a)
    assert hash_a == state_identity_hash(model_a)
    assert hash_a != state_identity_hash(model_b)


def test_full_cache_hit_skips_extraction(tmp_path: Path) -> None:
    class _Stub:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int):
            return {
                "images": torch.randn(1, 2, 4, 4),
                "label": torch.zeros(4, 4, dtype=torch.int64),
                "dates": torch.tensor([[5.0]]),
            }

    cache = make_cache(tmp_path)
    for row in range(2):
        cache.save(row, torch.zeros(2, 4, 4), torch.zeros(4, 4, dtype=torch.int64))
    assert cache.coverage(2) == 2

    class _RefusesExtraction(nn.Module):
        def fused_features(self, images, dates):
            raise AssertionError("extraction must not run on a full hit")

    populate(
        cache,
        _Stub(),
        _RefusesExtraction(),
        torch.zeros(1, 1, 2, 1, 1),
        torch.ones(1, 1, 2, 1, 1),
        "r",
        lambda *a, **k: None,
        "cpu",
    )


def test_populate_max_rows_caps_and_preserves_train_state(tmp_path: Path) -> None:
    class _Stub:
        def __len__(self) -> int:
            return 6

        def __getitem__(self, index: int):
            return {
                "images": torch.ones(1, 2, 4, 4),
                "label": torch.zeros(4, 4, dtype=torch.int64),
                "dates": torch.tensor([[7.0]]),
            }

    class _StandIn(nn.Module):
        def fused_features(self, images, dates):
            return torch.randn(images.shape[0], 2, 4, 4)

    cache = make_cache(tmp_path)
    model = _StandIn()
    model.train()
    was_training = model.training
    populate(
        cache,
        _Stub(),
        model,
        torch.zeros(1, 1, 2, 1, 1),
        torch.ones(1, 1, 2, 1, 1),
        "r",
        lambda *a, **k: None,
        "cpu",
        max_rows=2,
    )
    assert cache.coverage(6) == 2  # only the first two rows materialised
    assert model.training == was_training  # eval() semantics restored


def test_cached_dense_matches_the_direct_path(tmp_path: Path) -> None:
    """populate() must record exactly what the frozen forward would have produced."""

    class _FrozenStandIn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Conv2d(6, 8, kernel_size=1)

        def fused_features(self, images, dates):
            assert images.dtype == torch.float32
            return self.linear(images[:, 0])

    dataset_rows = 6

    class _StubDataset:
        def __len__(self) -> int:
            return dataset_rows

        def __getitem__(self, index: int):
            torch.manual_seed(index)
            return {
                "images": torch.randn(1, 6, 8, 8),
                "label": torch.full((8, 8), index % 2, dtype=torch.int64),
                "dates": torch.tensor([[10.0]]),
            }

    means = torch.zeros(1, 1, 6, 1, 1)
    stds = torch.ones(1, 1, 6, 1, 1)
    cache = make_cache(tmp_path)
    model = _FrozenStandIn()
    emitted: list[int] = []

    def emit(run_id, event, message, **kwargs):
        assert kwargs["progress"] <= 1.0
        emitted.append(1)

    populate(cache, _StubDataset(), model, means, stds, "run-test", emit, "cpu")
    assert cache.coverage(dataset_rows) == dataset_rows and sum(emitted) > 0

    # determinism: the stub dataset returns the same images per row, so the
    # cached row must equal a fresh direct pass within the fp16 rounding step
    for index in (0, dataset_rows - 1):
        item = _StubDataset()[index]
        # populate feeds the loader batch [B, T, C, H, W]-shaped inputs,
        # so the direct pass mirrors that layout with B=1
        images = (item["images"][None] - means) / stds
        direct = model.fused_features(images, item["dates"][None])[0]
        cached = cache.load(index)["dense"]
        torch.testing.assert_close(cached, direct, rtol=2e-2, atol=2e-2)

    loader = torch.utils.data.DataLoader(CachedSampleDataset(cache, dataset_rows), batch_size=3)
    for batch in loader:
        logits = nn.Conv2d(8, 2, kernel_size=3, padding=1)(batch["dense"])
        loss = torch.nn.functional.cross_entropy(logits.float(), batch["label"])
        assert torch.isfinite(loss)
