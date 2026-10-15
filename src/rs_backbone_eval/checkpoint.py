"""Epoch-boundary training recovery for the AnySat Colab workers.

Only worker environments import this module (torch is not a controller dependency).
Checkpoints contain tensors and primitive containers and load with weights_only=True.
"""

from __future__ import annotations

import hashlib
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .contracts import RunRecord, WorkerRequest
from .selection import BestCheckpoint
from .worker_runtime import RunWriter, emit_event


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(item) for item in value)
    return value


def _rng_state() -> dict:
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict) -> None:
    cuda = state["cuda"]
    if len(cuda) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("resume requires the same number of CUDA devices")
    random.setstate(state["python"])
    kind, values, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((kind, np.asarray(values, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


class TrainingCheckpoint:
    """Save after validation and before the next epoch's DataLoader iterator.

    Current datasets have deterministic __getitem__ methods, and each epoch
    creates fresh DataLoader workers. Restoring the main-process RNG therefore
    also restores shuffle order and worker seeds. Mid-epoch progress is replayed.
    """

    def __init__(
        self,
        request: WorkerRequest,
        writer: RunWriter,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        settings: dict,
        best: BestCheckpoint | None = None,
    ) -> None:
        self.writer, self.model, self.optimizer, self.best = writer, model, optimizer, best
        self.start_epoch = 0
        self.optimizer_steps = 0
        self.previous_wall = 0.0
        self.started = time.monotonic()
        self.every = int(os.environ.get("RS_EVAL_CHECKPOINT_EVERY", "1"))
        if self.every < 1:
            raise ValueError("RS_EVAL_CHECKPOINT_EVERY must be at least 1")
        self.local_path = writer.run_dir / "latest.pt"
        external = os.environ.get("RS_EVAL_CHECKPOINT_ROOT")
        self.durable_path = (
            Path(external).expanduser().resolve() / request.run_id / "latest.pt"
            if external
            else self.local_path
        )
        if self.durable_path.exists():
            raise ValueError("checkpoint destination already exists; use a new run ID")
        record = writer.record
        self.identity = {
            "task": record.task_id,
            "model": record.model_id,
            "protocol": record.protocol_id,
            "seed": record.seed,
            "manifest": record.dataset.manifest_sha256,
            "pretrained_checkpoint": record.model_checkpoint_sha256,
            "git_commit": record.git_commit,
            "upstream_commit": record.tags.get("anysat_commit"),
            "torch_version": str(torch.__version__),
            "device_type": next(model.parameters()).device.type,
            "init": record.tags.get("init", "pretrained"),
            "label_fraction": record.tags.get("label_fraction", "1.0"),
            "configs": {
                key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for key, path in {
                    "task": request.task_config_path,
                    "model": request.model_config_path,
                    "protocol": request.protocol_config_path,
                }.items()
            },
            "settings": settings,
        }
        if request.resume_from:
            self._load(Path(request.resume_from))
        writer.update(tags={"training_checkpoint": str(self.durable_path)})

    @property
    def elapsed(self) -> float:
        return self.previous_wall + time.monotonic() - self.started

    def _load(self, path: Path) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("not a supported training checkpoint (use latest.pt)")
        differences = [
            key for key, value in self.identity.items() if payload["identity"].get(key) != value
        ]
        if differences:
            raise ValueError(f"resume configuration differs: {', '.join(differences)}")
        saved = RunRecord.model_validate(payload["run_record"])
        self.start_epoch = int(payload["next_epoch"])
        self.optimizer_steps = saved.training.optimizer_steps
        self.previous_wall = saved.training.wall_time_seconds or 0.0
        if not 0 <= self.start_epoch <= self.identity["settings"]["epochs"]:
            raise ValueError("resume epoch is outside the requested training budget")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if self.best is not None:
            self.best.score, self.best.epoch, self.best.state = payload["best"]
        _restore_rng(payload["rng"])
        # A resumed attempt gets a new run ID; restore curves/counters, retaining
        # its own code version and recording its parent rather than overwriting it.
        self.writer.update(
            curves=saved.curves,
            training=saved.training,
            tags={
                "resume_from": str(path.resolve()),
                "resume_parent_run_id": saved.run_id,
                "resume_epoch": str(self.start_epoch),
            },
        )

    def save(self, next_epoch: int, optimizer_steps: int, *, final: bool = False) -> None:
        if not final and next_epoch % self.every:
            return
        peak = self.writer.record.training.peak_gpu_memory_mb or 0.0
        if next(self.model.parameters()).is_cuda:
            peak = max(peak, torch.cuda.max_memory_allocated() / 1024**2)
        self.writer.update_training(
            epochs_completed=next_epoch,
            optimizer_steps=optimizer_steps,
            wall_time_seconds=self.elapsed,
            peak_gpu_memory_mb=peak,
        )
        payload = {
            "format_version": 1,
            "identity": self.identity,
            "next_epoch": next_epoch,
            "model": _cpu(self.model.state_dict()),
            "optimizer": _cpu(self.optimizer.state_dict()),
            "best": (
                (self.best.score, self.best.epoch, _cpu(self.best.state))
                if self.best is not None
                else None
            ),
            "rng": _rng_state(),
            "run_record": self.writer.record.model_dump(mode="json"),
        }
        temporary = self.local_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(self.local_path)
        if self.durable_path != self.local_path:
            self.durable_path.parent.mkdir(parents=True, exist_ok=True)
            durable_tmp = self.durable_path.with_suffix(".pt.tmp")
            shutil.copyfile(self.local_path, durable_tmp)
            durable_tmp.replace(self.durable_path)
        if not any(a.path == "latest.pt" for a in self.writer.record.artifacts):
            self.writer.add_artifact("checkpoint", "latest.pt", "epoch-boundary training state")
        emit_event(
            self.writer.record.run_id,
            "train.checkpoint",
            f"saved epoch {next_epoch}: {self.durable_path}",
        )
