"""Resume real worker loops with a tiny CPU backbone and AdamW state."""

from __future__ import annotations

import importlib
import random
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rs_backbone_eval.checkpoint import TrainingCheckpoint  # noqa: E402
from rs_backbone_eval.contracts import WorkerRequest  # noqa: E402
from rs_backbone_eval.registry import ProtocolSpec, TaskSpec, load_yaml_spec  # noqa: E402
from rs_backbone_eval.repro import seed_everything  # noqa: E402
from rs_backbone_eval.worker_runtime import RunWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers/anysat/src"))


class Disconnected(Exception):
    pass


class TinyAdapter(torch.nn.Module):
    def __init__(self, backbone, bands, *args, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.adapter = torch.nn.Conv2d(6, 4, 1)
        self.adapter.synthesized_bands = []
        classes = args[0] if args else 5
        self.head = torch.nn.Sequential(torch.nn.Dropout2d(0.2), torch.nn.Conv2d(4, classes, 1))

    def forward(self, images, dates):
        return self.head(self.adapter(images.mean(dim=1)))


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, root, split, *args, **kwargs):
        self.rows = [
            {"group_id": f"group-{index}"} for index in range(5 if split == "train" else 2)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # Deterministic inputs, different ignored-pixel counts across samples.
        image = np.arange(6 * 4 * 4, dtype=np.float32).reshape(6, 4, 4) / 100 + index
        label = np.full((4, 4), index % 2, dtype=np.int64)
        label.ravel()[:index] = -1
        return {
            "image": image,
            "images": np.stack([image] * 3),
            "label": label,
            "date": 42,
            "dates": np.asarray([1, 10, 20]),
        }


def make_request(tmp_path, task, run_id, resume_from=None):
    task_files = {
        "T1": "T1_flood",
        "T3": "T3_burn_intensity",
        "T4": "T4_landslide",
        "T5": "T5_us_multicrop",
    }
    request = WorkerRequest(
        action="train",
        run_id=run_id,
        task_config_path=str(ROOT / f"configs/tasks/{task_files[task]}.yaml"),
        model_config_path=str(ROOT / "configs/models/anysat.yaml"),
        protocol_config_path=str(ROOT / "configs/protocols/full_finetune.yaml"),
        manifest_path=str(tmp_path / "manifest.jsonl"),
        data_root=str(tmp_path),
        weights_root=str(tmp_path),
        cache_root=str(tmp_path / "cache"),
        runs_root=str(tmp_path / "runs"),
        seed=42,
        resume_from=str(resume_from) if resume_from else None,
    )
    writer = RunWriter(
        Path(request.runs_root) / run_id,
        run_id=run_id,
        task_spec=load_yaml_spec(Path(request.task_config_path), TaskSpec),
        protocol_spec=load_yaml_spec(Path(request.protocol_config_path), ProtocolSpec),
        model_id="anysat_base",
        seed=42,
        manifest_sha256="a" * 64,
    )
    writer.update(model_checkpoint_sha256="b" * 64)
    return request, writer


def assert_tensors_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tensors_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert_tensors_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("task", ["T1", "T3", "T4", "T5"])
def test_worker_resume_matches_uninterrupted_training(task, tmp_path, monkeypatch):
    torch.set_num_threads(1)
    module_name = "t1_adapter" if task == "T1" else task.lower()
    module = importlib.import_module(f"anysat_worker.{module_name}")
    monkeypatch.setenv("ANYSAT_WORKER_EPOCHS", "3")
    monkeypatch.setenv("ANYSAT_WORKER_BATCH_SIZE", "1")
    monkeypatch.setenv("ANYSAT_WORKER_GRAD_ACCUM", "2")
    monkeypatch.setenv("ANYSAT_WORKER_NUM_WORKERS", "0")
    monkeypatch.setenv("RS_EVAL_VAL_EVERY", "1")
    monkeypatch.setenv("RS_EVAL_CHECKPOINT_ROOT", str(tmp_path / "drive"))
    monkeypatch.setenv("RS_EVAL_CHECKPOINT_EVERY", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module, "build_backbone", lambda _: torch.nn.Identity())
    dataset_name = {
        "T1": "T1SixBandDataset",
        "T3": "BurnIntensityDataset",
        "T4": "LandslideDataset",
        "T5": "MulticropDataset",
    }[task]
    model_name = {"T1": "AdapterProbe", "T3": "AnySatT3", "T4": "AdapterProbe", "T5": "AnySatT5"}[
        task
    ]
    monkeypatch.setattr(module, dataset_name, TinyDataset)

    def tiny_model(backbone, bands, *args, **kwargs):
        classes = {"T1": 2, "T3": 5, "T4": 2, "T5": 13}[task]
        return TinyAdapter(backbone, bands, classes)

    monkeypatch.setattr(module, model_name, tiny_model)
    statistics_name = {
        "T1": "train_normalization",
        "T3": "burn_intensity_statistics",
        "T4": "landslide_statistics",
        "T5": "multicrop_statistics",
    }[task]
    stats_size = 18 if task == "T3" else 6
    monkeypatch.setattr(
        module,
        statistics_name,
        lambda *args: (
            np.zeros(stats_size, dtype=np.float32),
            np.ones(stats_size, dtype=np.float32),
        ),
    )
    real_loader = torch.utils.data.DataLoader

    def loader(*args, **kwargs):
        kwargs["num_workers"] = 0
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(module, "DataLoader", loader)
    train = getattr(module, "train_adapter" if task == "T1" else f"train_{task.lower()}")

    def run(run_id, resume=None):
        seed_everything(42)
        request, writer = make_request(tmp_path, task, run_id, resume)
        kwargs = {
            "checkpoint": tmp_path / "weights.pt",
            "task_bands": list(writer.record.input.bands),
            "freeze_backbone": False,
        }
        if task != "T1":
            kwargs["metric_name"] = "mean_iou"
        assert train(request, writer, lambda *args, **kwargs: None, **kwargs) == 0
        return writer

    full = run("full-run")
    reference = torch.load(tmp_path / "drive/full-run/latest.pt", weights_only=True)
    original_save = TrainingCheckpoint.save

    def disconnect(self, epoch, steps, **kwargs):
        original_save(self, epoch, steps, **kwargs)
        if epoch == 1:
            raise Disconnected

    monkeypatch.setattr(TrainingCheckpoint, "save", disconnect)
    with pytest.raises(Disconnected):
        run("interrupted-run")
    source = tmp_path / "drive/interrupted-run/latest.pt"
    assert source.is_file()
    monkeypatch.setattr(TrainingCheckpoint, "save", original_save)
    # Consume every RNG stream before resuming to prove they are restored.
    random.random(), np.random.rand(), torch.rand(3)
    resumed = run("resumed-run", source)
    recovered = torch.load(tmp_path / "drive/resumed-run/latest.pt", weights_only=True)
    for key in ("model", "optimizer", "rng", "best"):
        assert_tensors_equal(reference[key], recovered[key])
    assert recovered["next_epoch"] == 3
    assert full.record.training.optimizer_steps == resumed.record.training.optimizer_steps
    assert full.record.curves == resumed.record.curves
    assert resumed.record.tags["resume_parent_run_id"] == "interrupted-run"
    assert resumed.record.tags["resume_epoch"] == "1"
    assert resumed.record.training.epochs_completed == 3
    assert len([c for c in resumed.record.curves if c.name == "loss"]) == 3
    # A disconnect during final evaluation must not retrain the finished epochs.
    evaluated = run("reevaluated-run", tmp_path / "drive/resumed-run/latest.pt")
    assert evaluated.record.curves == full.record.curves
    assert evaluated.record.training.optimizer_steps == full.record.training.optimizer_steps


def test_resume_rejects_changed_settings_and_preserves_durable_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("RS_EVAL_CHECKPOINT_ROOT", str(tmp_path / "drive"))
    request, writer = make_request(tmp_path, "T5", "initial-run")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = TrainingCheckpoint(
        request, writer, model, optimizer, settings={"epochs": 3, "batch": 2}
    )
    checkpoint.save(1, 4)
    original = checkpoint.durable_path.read_bytes()
    with pytest.raises(ValueError, match="new run ID"):
        TrainingCheckpoint(request, writer, model, optimizer, settings={"epochs": 3, "batch": 2})
    request, writer = make_request(tmp_path, "T5", "resume-run", checkpoint.durable_path)
    with pytest.raises(ValueError, match="settings"):
        TrainingCheckpoint(request, writer, model, optimizer, settings={"epochs": 3, "batch": 4})
    assert checkpoint.durable_path.read_bytes() == original


def test_failed_drive_copy_keeps_last_complete_checkpoint(tmp_path, monkeypatch):
    import rs_backbone_eval.checkpoint as module

    monkeypatch.setenv("RS_EVAL_CHECKPOINT_ROOT", str(tmp_path / "drive"))
    request, writer = make_request(tmp_path, "T5", "copy-failure-run")
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = TrainingCheckpoint(request, writer, model, optimizer, settings={"epochs": 3})
    checkpoint.save(1, 2)
    original = checkpoint.durable_path.read_bytes()

    def interrupted_copy(source, destination):
        Path(destination).write_bytes(b"partial upload")
        raise OSError("Drive disconnected")

    monkeypatch.setattr(module.shutil, "copyfile", interrupted_copy)
    with pytest.raises(OSError, match="Drive disconnected"):
        checkpoint.save(2, 4)
    assert checkpoint.durable_path.read_bytes() == original
    assert torch.load(checkpoint.durable_path, weights_only=True)["next_epoch"] == 1
