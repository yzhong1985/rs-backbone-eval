"""CPU end-to-end checks for the T13 fusion trainer through ``train_fusion``.

A small conv pyramid stands in for MiT-Huge and the FT-B0 checkpoint is
replaced by a fixed dependency path. Everything else (tiles, group-normalised
CE, validation scoring, export guards, ledger calls) runs as in production.

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_t13_fusion_train.py -q
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
import torch  # noqa: E402
from lim_worker import t13, t13_fusion_train  # noqa: E402
from torch import nn  # noqa: E402

W, H, CROP = 96, 64, 64

PLUGIN = "fusion_plugin.pt"
# The trainer records the dependency's parent directory name as the baseline run,
# so the stand-in checkpoint lives in a directory named like the pinned baseline.
BASELINE_RUN = "lim_mit_huge-T13-whu_ft_b0-train-17"


class _Pyramid(nn.Module):
    """Four levels at strides 4, 8, 16, 32 with the LIM channel widths."""

    def __init__(self) -> None:
        super().__init__()
        chans = t13.CHANNELS
        self.stem = nn.Conv2d(3, chans[0], 4, stride=4)
        self.stages = nn.ModuleList(
            nn.Conv2d(chans[i], chans[i + 1], 2, stride=2) for i in range(3)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outs = [x]
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs


def _write(path: Path, array: np.ndarray, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[-1],
        height=array.shape[-2],
        count=array.shape[0],
        dtype=dtype,
    ) as dst:
        dst.write(array.astype(dtype))


def fake_sheet(root: Path, name: str, seed: int) -> list[dict]:
    """Two overlapping 64-px tiles over a 64x96 sheet, as in test_lim_t13."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (4, H, W))
    label = rng.integers(0, 8, (1, H, W)) * 10
    label[0, :4, :4] = 0
    _write(root / "raw/whu/optical" / f"{name}.tif", image, "uint8")
    _write(root / "raw/whu/lbl" / f"{name}.tif", label, "uint8")
    return [
        {
            "sample_id": f"{name}_x{x}",
            "group_id": name,
            "split": "test",
            "image_paths": [f"raw/whu/optical/{name}.tif"],
            "label_path": f"raw/whu/lbl/{name}.tif",
            "window": [x, 0, CROP, CROP],
        }
        for x in (0, W - CROP)
    ]


class _Writer:
    """Records the ledger calls the trainer makes."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.attached: Path | None = None
        self.completed: str | None = None
        self.tags: dict = {}
        self.metrics: list = []

    def attach_checkpoint(self, path) -> None:
        self.attached = Path(path)

    def add_curve(self, *args) -> None:
        pass

    def add_artifact(self, *args) -> None:
        pass

    def update(self, **kwargs) -> None:
        self.tags.update(kwargs.get("tags") or {})

    def add_metrics(self, records) -> None:
        self.metrics += list(records)

    def add_confusion(self, *args) -> None:
        pass

    def update_training(self, **kwargs) -> None:
        pass

    def complete(self, summary: str) -> None:
        self.completed = summary


def _dataset(root: Path) -> Path:
    """Four train sheets (8 tiles, one accumulation group) and two validation sheets."""
    rows = []
    for i in range(4):
        rows += fake_sheet(root, f"TRAIN{i}", seed=i)
    for row in rows:
        row["split"] = "train"
    val = []
    for i in range(2):
        val += fake_sheet(root, f"VAL{i}", seed=10 + i)
    for row in val:
        row["split"] = "validation"
    test = fake_sheet(root, "TEST0", seed=20)
    for row in test:
        row["split"] = "test"
    manifest = root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows + val + test:
            handle.write(json.dumps({**row, "task_id": "T13"}) + "\n")
    return manifest


def _run(
    root: Path,
    monkeypatch,
    tmp_name: str,
    *,
    micro: int = 2,
    limit: int = 0,
    lr: float = 1e-3,
    schedule: str | None = None,
    epochs: int = 2,
    protocol_id: str = "whu_ft_nir",
    train_head: bool = False,
) -> _Writer:
    manifest = root / "manifest.jsonl"
    run_dir = root / "runs" / tmp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "runs" / BASELINE_RUN / "best_full_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"stand-in")
    monkeypatch.setenv("LIM_WORKER_MICRO_BATCH", str(micro))
    monkeypatch.setenv("LIM_WORKER_LIMIT_BATCHES", str(limit))
    monkeypatch.setenv("LIM_WORKER_EVAL_IMAGES", "0")
    monkeypatch.setenv("LIM_WORKER_NUM_WORKERS", "0")
    # CPU fp32 path: on CUDA the FT-B0 bf16 autocast adds ~1e-3 weight noise per step.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(t13_fusion_train, "mit_huge", lambda in_chans=3, **kw: _Pyramid())
    monkeypatch.setattr(t13_fusion_train, "find_full_dependency", lambda *a, **kw: checkpoint)
    monkeypatch.setattr(t13_fusion_train, "load_full_checkpoint", lambda *a, **kw: {})
    request = SimpleNamespace(
        run_id=tmp_name,
        seed=17,
        data_root=str(root),
        runs_root=str(root / "runs"),
        manifest_path=str(manifest),
    )
    protocol = SimpleNamespace(
        protocol_id=protocol_id,
        epochs=epochs,
        learning_rate=lr,
        lr_schedule=schedule,
        train_head=train_head,
    )
    writer = _Writer(run_dir)
    code = t13_fusion_train.train_fusion(
        request,
        writer,
        lambda *a, **kw: None,
        protocol=protocol,
        task_spec=None,
        model_id="lim_mit_huge",
    )
    assert code == 0
    return writer


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    _dataset(tmp_path)
    return tmp_path


def test_ledger_top_level_checkpoint_is_the_ft_b0_baseline(dataset, monkeypatch) -> None:
    writer = _run(dataset, monkeypatch, "a")
    assert writer.attached == dataset / "runs" / BASELINE_RUN / "best_full_checkpoint.pt"
    assert (writer.run_dir / PLUGIN).is_file()
    assert "test not evaluated" in writer.completed


def test_one_batch_smoke_takes_no_group_and_saves_no_plugin(dataset, monkeypatch) -> None:
    with pytest.raises(ValueError, match="no complete accumulation group"):
        _run(dataset, monkeypatch, "limit1", micro=2, limit=1)
    assert not (dataset / "runs" / "limit1" / PLUGIN).exists()


def test_zero_effective_update_is_refused_and_not_exported(dataset, monkeypatch) -> None:
    # A learning rate this small leaves every float32 weight unchanged.
    with pytest.raises(RuntimeError, match="no trained update"):
        _run(dataset, monkeypatch, "tiny-lr", lr=1e-30, schedule=None)
    assert not (dataset / "runs" / "tiny-lr" / PLUGIN).exists()


@pytest.mark.parametrize("micro", [4, 8])
def test_micro_batch_split_does_not_change_the_trained_plugin(dataset, monkeypatch, micro) -> None:
    base = _run(dataset, monkeypatch, "micro2")
    other = _run(dataset, monkeypatch, f"micro{micro}", micro=micro)
    a = torch.load(base.run_dir / PLUGIN)["state_dict"]
    b = torch.load(other.run_dir / PLUGIN)["state_dict"]
    assert a.keys() == b.keys()
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-5), key


def test_nonfinite_logits_stop_the_run_and_save_nothing(dataset, monkeypatch) -> None:
    real = t13_fusion_train.fused_logits

    def poisoned(*args, **kwargs):
        return real(*args, **kwargs) * float("nan")

    monkeypatch.setattr(t13_fusion_train, "fused_logits", poisoned)
    with pytest.raises(RuntimeError, match="not finite"):
        _run(dataset, monkeypatch, "nan")
    assert not (dataset / "runs" / "nan" / PLUGIN).exists()


def test_require_finite_rejects_nan_and_inf() -> None:
    t13_fusion_train._require_finite("ok", torch.tensor([0.0, 1.0]))
    with pytest.raises(RuntimeError, match="not finite"):
        t13_fusion_train._require_finite("bad", torch.tensor([0.0, float("inf")]))


def test_zero_gradient_run_is_refused_even_when_weight_decay_moves_other_weights(
    dataset, monkeypatch
) -> None:
    real = t13_fusion_train.fused_logits
    # Multiplying by zero keeps the graph but gives every branch weight a zero gradient.
    monkeypatch.setattr(t13_fusion_train, "fused_logits", lambda *a, **kw: real(*a, **kw) * 0.0)
    with pytest.raises(RuntimeError, match="no trained update"):
        _run(dataset, monkeypatch, "zero-grad")
    assert not (dataset / "runs" / "zero-grad" / PLUGIN).exists()


def test_overflowing_cross_entropy_stops_the_run(dataset, monkeypatch) -> None:
    real = t13_fusion_train.fused_logits
    # Finite but huge logits: the fp32 sum of the per-pixel losses overflows to inf.
    monkeypatch.setattr(t13_fusion_train, "fused_logits", lambda *a, **kw: real(*a, **kw) * 1e36)
    with pytest.raises(RuntimeError, match="not finite"):
        _run(dataset, monkeypatch, "ce-overflow")
    assert not (dataset / "runs" / "ce-overflow" / PLUGIN).exists()


def _source_record(
    run_dir: Path,
    *,
    artifact: str = "fusion_plugin.pt",
    status: str = "completed",
    epochs: int = 2,
    tags: dict | None = None,
) -> None:
    """Write the source run record that the locked evaluate reads (test stand-in)."""
    import hashlib
    import json

    if not (run_dir / artifact).exists():
        (run_dir / artifact).write_bytes(b"plugin-bytes")
    sha = hashlib.sha256((run_dir / artifact).read_bytes()).hexdigest()
    record = {
        "status": status,
        "training": {"epochs_completed": epochs},
        "artifacts": [{"kind": "checkpoint", "path": artifact}],
        "tags": {"plugin_sha256": sha, **(tags or {})},
    }
    (run_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")


def test_locked_evaluate_scores_the_test_split_once(dataset, monkeypatch) -> None:
    from lim_worker import t13_fusion_eval

    _run(dataset, monkeypatch, "trained")
    _source_record(dataset / "runs" / "trained")
    checkpoint = dataset / "runs" / BASELINE_RUN / "best_full_checkpoint.pt"
    monkeypatch.setattr(t13_fusion_eval, "mit_huge", lambda in_chans=3, **kw: _Pyramid())
    monkeypatch.setattr(t13_fusion_eval, "find_full_dependency", lambda *a, **kw: checkpoint)
    monkeypatch.setattr(t13_fusion_eval, "load_full_checkpoint", lambda *a, **kw: {})
    monkeypatch.setenv("LIM_WORKER_PLUGIN_RUN", "trained")
    monkeypatch.setenv("LIM_WORKER_EVAL_IMAGES", "0")
    monkeypatch.setenv("LIM_WORKER_NUM_WORKERS", "0")
    request = SimpleNamespace(
        run_id="evaluated",
        seed=17,
        data_root=str(dataset),
        runs_root=str(dataset / "runs"),
        manifest_path=str(dataset / "manifest.jsonl"),
    )
    writer = _Writer(dataset / "runs" / "evaluated")
    code = t13_fusion_eval.evaluate_fusion(
        request,
        writer,
        lambda *a, **kw: None,
        protocol=SimpleNamespace(protocol_id="whu_ft_nir", epochs=2),
        task_spec=None,
        model_id="lim_mit_huge",
    )
    assert code == 0
    assert writer.tags["test_evaluated"] == "yes"
    assert writer.tags["test_sheets"] == "1"
    names = {record.name for record in writer.metrics}
    assert {"mean_iou", "boundary_iou"} <= names
    assert "test mean_iou" in writer.completed


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"status": "failed"}, "not completed"),
        ({"tags": {"superseded": "yes"}}, "superseded"),
        ({"epochs": 1}, "budget"),
        ({"artifact": "other_plugin.pt"}, "registered artifact"),
    ],
)
def test_source_run_must_be_completed_full_budget_and_registered(tmp_path, kwargs, message) -> None:
    from lim_worker import t13_fusion_eval

    run_dir = tmp_path / "trained"
    run_dir.mkdir()
    _source_record(run_dir, **kwargs)
    plugin = {"smoke": False}
    with pytest.raises(RuntimeError, match=message):
        t13_fusion_eval.check_source_run(
            tmp_path, "trained", plugin_file="fusion_plugin.pt", epochs=2, plugin=plugin
        )


@pytest.mark.parametrize("plugin", [{"smoke": True}, {}])
def test_smoke_or_unflagged_plugin_is_refused(tmp_path, plugin) -> None:
    from lim_worker import t13_fusion_eval

    run_dir = tmp_path / "trained"
    run_dir.mkdir()
    _source_record(run_dir)
    with pytest.raises(RuntimeError, match="smoke"):
        t13_fusion_eval.check_source_run(
            tmp_path, "trained", plugin_file="fusion_plugin.pt", epochs=2, plugin=plugin
        )


def test_matching_source_run_returns_the_plugin_sha256(tmp_path) -> None:
    import hashlib

    from lim_worker import t13_fusion_eval

    run_dir = tmp_path / "trained"
    run_dir.mkdir()
    _source_record(run_dir)
    sha = t13_fusion_eval.check_source_run(
        tmp_path, "trained", plugin_file="fusion_plugin.pt", epochs=2, plugin={"smoke": False}
    )
    assert sha == hashlib.sha256(b"plugin-bytes").hexdigest()


def test_head_training_exports_the_head_and_keeps_branch_keys_clean(dataset, monkeypatch) -> None:
    import torch

    writer = _run(
        dataset,
        monkeypatch,
        "head-trained",
        protocol_id="whu_ft_nir_head",
        train_head=True,
    )
    plugin = torch.load(writer.run_dir / "fusion_plugin.pt")
    assert plugin["config"]["head"] is True
    assert plugin["head_state"], "the trained UPerNet head must be exported"
    assert not any(k.startswith("head.") for k in plugin["state_dict"])


def test_replaced_plugin_file_is_refused(tmp_path) -> None:
    from lim_worker import t13_fusion_eval

    run_dir = tmp_path / "trained"
    run_dir.mkdir()
    _source_record(run_dir)
    (run_dir / "fusion_plugin.pt").write_bytes(b"swapped-bytes")
    with pytest.raises(RuntimeError, match="differs from the SHA-256"):
        t13_fusion_eval.check_source_run(
            tmp_path, "trained", plugin_file="fusion_plugin.pt", epochs=2, plugin={"smoke": False}
        )


def test_head_movement_alone_is_not_an_injector_update() -> None:
    import torch
    from lim_worker.t13_fusion_train import _residual_update

    initial = {"head.classifier.weight": torch.zeros(3), "nir.proj.weight": torch.zeros(3)}
    state = {"head.classifier.weight": torch.ones(3), "nir.proj.weight": torch.zeros(3)}
    assert _residual_update(initial, state) == 0.0
