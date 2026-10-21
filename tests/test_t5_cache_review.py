"""Regression tests for the review's cache findings (P2 1-3).

- a resume must not shrink the training subset through a short feature cache;
- a run that already spent its epoch budget must not rebuild the cache;
- the cached and online paths must iterate the same samples in the same order.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


class _TinyMulticrop(torch.utils.data.Dataset):
    """Four training rows, two validation rows; no rasters on disk."""

    def __init__(self, manifest, root, split, *args, **kwargs):
        self.rows = [{"group_id": f"group-{i}"} for i in range(4 if split == "train" else 2)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        image = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4) / 100 + index
        label = np.full((4, 4), index % 2, dtype=np.int64)
        return {
            "images": np.stack([image] * 3),
            "label": label,
            "dates": np.asarray([1, 10, 20]),
        }


class _TinyAnySatT5(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = torch.nn.Identity()
        self.adapter = torch.nn.Conv2d(6, 4, 1)
        self.adapter.synthesized_bands = []
        self.head = torch.nn.Conv2d(4, 13, 1)

    def forward(self, images, dates):
        return self.head(self.adapter(images.mean(dim=1)))

    def fused_features(self, images, dates):
        return self.adapter(images.mean(dim=1))


def _writer(tmp_path: Path, run_id: str):
    from rs_backbone_eval.contracts import WorkerRequest
    from rs_backbone_eval.data.manifest import write_manifest
    from rs_backbone_eval.registry import ProtocolSpec, TaskSpec, load_yaml_spec
    from rs_backbone_eval.worker_runtime import RunWriter

    request = WorkerRequest(
        action="train",
        run_id=run_id,
        task_config_path=str(ROOT / "configs/tasks/T5_us_multicrop.yaml"),
        model_config_path=str(ROOT / "configs/models/anysat.yaml"),
        protocol_config_path=str(ROOT / "configs/protocols/frozen_adapter_two_stage.yaml"),
        manifest_path=str(tmp_path / "manifest.jsonl"),
        data_root=str(tmp_path),
        weights_root=str(tmp_path),
        cache_root=str(tmp_path / "cache"),
        runs_root=str(tmp_path / "runs"),
        seed=42,
        resume_from=None,
    )
    write_manifest(Path(request.manifest_path), [])
    writer = RunWriter(
        Path(request.runs_root) / request.run_id,
        run_id=request.run_id,
        task_spec=load_yaml_spec(Path(request.task_config_path), TaskSpec),
        protocol_spec=load_yaml_spec(Path(request.protocol_config_path), ProtocolSpec),
        model_id="anysat_base",
        seed=42,
        manifest_sha256="c" * 64,
    )
    writer.update(model_checkpoint_sha256="d" * 64)
    return request, writer


def _harness(
    tmp_path: Path,
    monkeypatch,
    *,
    epochs: int,
    limit: int,
    max_rows: int | None,
    cache: bool = True,
):
    module = importlib.import_module("anysat_worker.t5")
    from rs_backbone_eval.repro import seed_everything

    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("ANYSAT_WORKER_EPOCHS", str(epochs))
    monkeypatch.setenv("ANYSAT_WORKER_BATCH_SIZE", "1")
    monkeypatch.setenv("ANYSAT_WORKER_GRAD_ACCUM", "1")
    monkeypatch.setenv("ANYSAT_WORKER_NUM_WORKERS", "0")
    monkeypatch.setenv("ANYSAT_WORKER_LIMIT_BATCHES", str(limit))
    monkeypatch.setenv("ANYSAT_T5_STAGE", "two")
    monkeypatch.setenv("ANYSAT_STAGE_B_START", "1")
    monkeypatch.setenv("ANYSAT_FROZEN_CACHE", "1" if cache else "0")
    monkeypatch.setenv("RS_EVAL_VAL_EVERY", "1")
    monkeypatch.setenv("RS_EVAL_CHECKPOINT_ROOT", str(tmp_path / "drive"))
    monkeypatch.setenv("RS_EVAL_CHECKPOINT_EVERY", "1")
    if max_rows is None:
        monkeypatch.delenv("ANYSAT_WORKER_CACHE_MAX_ROWS", raising=False)
    else:
        monkeypatch.setenv("ANYSAT_WORKER_CACHE_MAX_ROWS", str(max_rows))
    monkeypatch.setattr(module, "build_backbone", lambda _: torch.nn.Identity())
    monkeypatch.setattr(module, "MulticropDataset", _TinyMulticrop)
    monkeypatch.setattr(module, "AnySatT5", lambda *a, **k: _TinyAnySatT5())
    monkeypatch.setattr(
        module,
        "multicrop_statistics",
        lambda *a: (np.zeros(6, dtype=np.float32), np.ones(6, dtype=np.float32)),
    )
    real_loader = torch.utils.data.DataLoader

    def loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(module, "DataLoader", loader)
    seed_everything(42)
    return module


def _run(module, request, writer, events):
    return module.train_t5(
        request,
        writer,
        lambda *args, **kwargs: events.append((args[1], args[2])),
        checkpoint=Path(request.weights_root) / "weights.pt",
        task_bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
        freeze_backbone=True,
        metric_name="mean_iou",
    )


def test_short_feature_cache_fails_the_run(tmp_path, monkeypatch) -> None:
    """Review P2: 4 training rows must not silently become 2 on resume."""
    from anysat_worker import frozen_cache

    # stage_b_start=1, so the boundary (and the cache) is reached at epoch 2
    module = _harness(tmp_path, monkeypatch, epochs=2, limit=0, max_rows=None)
    request, writer = _writer(tmp_path, "short-cache")
    writes = []

    def short_populate(cache, dataset, *args, **kwargs):
        for index in range(2):  # only two of the four rows make it to disk
            cache.save(
                index,
                torch.zeros(4, 14, 14),
                torch.zeros(4, 4, dtype=torch.long),
            )
        return 2

    monkeypatch.setattr(frozen_cache, "populate", short_populate)
    with pytest.raises(RuntimeError, match="must not change the training subset"):
        _run(module, request, writer, writes)
    assert writer.record.tags.get("feature_cache_rows") is None


def test_completed_run_resume_does_not_rebuild_the_cache(tmp_path, monkeypatch) -> None:
    """Review P2: re-running a finished run for evaluation must not rebuild ~428GB."""
    from anysat_worker import frozen_cache

    # two epochs so the boundary and the cache are exercised before finishing
    module = _harness(tmp_path, monkeypatch, epochs=2, limit=1, max_rows=None)
    request, writer = _writer(tmp_path, "finished-run")
    assert _run(module, request, writer, []) == 0
    assert writer.record.status == "completed"
    # the first run must really have gone through the cache, otherwise the
    # guard below could pass for the wrong reason
    assert writer.record.tags.get("feature_cache_rows") == "4"

    def explode(*args, **kwargs):
        raise AssertionError("populate must not run when the epoch budget is already spent")

    monkeypatch.setattr(frozen_cache, "populate", explode)
    events: list[tuple[str, str]] = []
    request2, writer2 = _writer(tmp_path, "finished-run-resume")
    request2 = request2.model_copy(
        update={"resume_from": str(tmp_path / "drive" / "finished-run" / "latest.pt")}
    )
    assert _run(module, request2, writer2, events) == 0
    assert "feature_cache_rows" not in writer2.record.tags
    assert any("cache not built" in message for _, message in events)


def test_cached_and_online_loaders_iterate_the_same_order() -> None:
    """Review P2: equal row counts per epoch do not prove equal sample sequences."""
    from anysat_worker.t5 import seed_loader_for_epoch

    class Indicies(torch.utils.data.Dataset):
        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

        def __getitem__(self, index):
            return index

    def order(loader):
        return [int(value) for batch in loader for value in batch]

    online = torch.utils.data.DataLoader(
        Indicies(9), batch_size=2, shuffle=True, generator=torch.Generator()
    )
    cached = torch.utils.data.DataLoader(
        Indicies(9), batch_size=2, shuffle=True, generator=torch.Generator()
    )
    for epoch in range(3):
        seed_loader_for_epoch(online, 42, epoch)
        seed_loader_for_epoch(cached, 42, epoch)
        assert order(online) == order(cached)
        # Model-side randomness (dropout etc.) must not shift the next epoch.
        torch.rand(100)
        np.random.rand(100)

    # Different epochs get different permutations, otherwise the shuffle is dead.
    seed_loader_for_epoch(online, 42, 0)
    first = order(online)
    seed_loader_for_epoch(online, 42, 1)
    assert order(online) != first


def test_resume_with_a_different_cache_cap_is_rejected(tmp_path, monkeypatch) -> None:
    """Review P2: cap 4 -> 2 changes the training rows, so resume must fail."""
    module = _harness(tmp_path, monkeypatch, epochs=2, limit=1, max_rows=4)
    request, writer = _writer(tmp_path, "capped-run")
    assert _run(module, request, writer, []) == 0
    assert writer.record.tags.get("feature_cache_rows") == "4"

    module = _harness(tmp_path, monkeypatch, epochs=2, limit=1, max_rows=2)
    request2, writer2 = _writer(tmp_path, "capped-run-resume")
    request2 = request2.model_copy(
        update={"resume_from": str(tmp_path / "drive" / "capped-run" / "latest.pt")}
    )
    with pytest.raises(ValueError, match="resume configuration differs"):
        _run(module, request2, writer2, [])


def test_cached_stage_b_matches_an_online_stage_b(tmp_path, monkeypatch) -> None:
    """Review P2: the cache may change speed, not the training trajectory."""
    module = _harness(tmp_path, monkeypatch, epochs=2, limit=2, max_rows=None)
    request, writer = _writer(tmp_path, "with-cache")
    assert _run(module, request, writer, []) == 0
    assert writer.record.tags.get("feature_cache_rows") == "4"

    module = _harness(tmp_path, monkeypatch, epochs=2, limit=2, max_rows=None, cache=False)
    request2, writer2 = _writer(tmp_path, "without-cache")
    assert _run(module, request2, writer2, []) == 0
    assert "feature_cache_rows" not in writer2.record.tags

    cached_losses = [c.value for c in writer.record.curves if c.name == "loss"]
    online_losses = [c.value for c in writer2.record.curves if c.name == "loss"]
    assert len(cached_losses) == len(online_losses) == 2
    for cached, online in zip(cached_losses, online_losses, strict=True):
        # fp16 storage is one rounding step away from the online bf16 feature
        assert cached == pytest.approx(online, rel=2e-3, abs=1e-4)

    cached_weights = torch.load(tmp_path / "drive" / "with-cache" / "latest.pt", weights_only=True)[
        "model"
    ]
    online_weights = torch.load(
        tmp_path / "drive" / "without-cache" / "latest.pt", weights_only=True
    )["model"]
    assert cached_weights.keys() == online_weights.keys()
    for key, value in cached_weights.items():
        torch.testing.assert_close(online_weights[key], value, atol=2e-4, rtol=2e-3)


def test_smoke_cap_binds_the_online_fallback(tmp_path, monkeypatch) -> None:
    """Review P2 (third round): cap=2 must mean 2 rows even without the cache."""
    module = _harness(tmp_path, monkeypatch, epochs=1, limit=3, max_rows=2, cache=False)
    request, writer = _writer(tmp_path, "capped-online")
    assert _run(module, request, writer, []) == 0
    # four training rows, cap two, micro-batch one: two updates, not three
    assert writer.record.training.optimizer_steps == 2

    module = _harness(tmp_path, monkeypatch, epochs=1, limit=3, max_rows=None, cache=False)
    request, writer = _writer(tmp_path, "uncapped-online")
    assert _run(module, request, writer, []) == 0
    assert writer.record.training.optimizer_steps == 3


def test_smoke_cap_survives_a_failed_cache_extraction(tmp_path, monkeypatch) -> None:
    """The fallback loader after an extraction failure must keep the cap."""
    from anysat_worker import frozen_cache

    module = _harness(tmp_path, monkeypatch, epochs=2, limit=3, max_rows=2)
    request, writer = _writer(tmp_path, "capped-fallback")

    def exploding_populate(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(frozen_cache, "populate", exploding_populate)
    assert _run(module, request, writer, []) == 0
    # both epochs run on the capped online loader: 2 rows x 2 epochs
    assert writer.record.training.optimizer_steps == 4
    assert "feature_cache_rows" not in writer.record.tags
