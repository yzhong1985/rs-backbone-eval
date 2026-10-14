"""T3 (burn scar intensity) training for the AnySat worker.

AnySat's public ``s2`` entrance natively consumes a time series
(``[B,T,10,H,W]`` plus day-of-year dates) and fuses it internally, so the three
HLS dates (pre/during/post, each a separate six-band raster) are stacked into
one series and passed through the explicit spectral adapter (zero padding is
forbidden by the design). The shared spatial head sits on the fused dense map.

Protocol note: unlike Clay and Prithvi, which encode each date separately and
let the shared ``TemporalSegmentationHead`` fuse them, AnySat uses its native
temporal fusion. That difference is recorded in the run tags and must be stated
in any comparison table. Both tracks select on ``validation`` and report
``test`` once.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
from rs_backbone_eval.evaluation.segmentation import multiclass_metrics
from rs_backbone_eval.heads import LiteSpatialHead, count_parameters
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.temporal_io import (
    burn_intensity_statistics,
    read_burn_intensity_label,
    read_hls_dates,
)
from rs_backbone_eval.worker_runtime import RunWriter

from .s2_probe import add_upstream_to_path
from .spectral_adapter import SpectralAdapter

BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR_1", "SWIR_2"]
NUM_CLASSES = 5
IGNORE_INDEX = -1
TIME_STEPS = 3
CHIP_SIZE = 224
PATCH_SIZE_M = 160
OFFICIAL_EPOCHS = 50
SELECT_SPLIT = "validation"
EVAL_SPLIT = "test"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class BurnIntensityDataset(Dataset):
    def __init__(self, manifest_path: Path, data_root: Path, split: str):
        self.data_root = data_root
        self.rows: list[dict] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["split"] == split:
                    self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        images = read_hls_dates([self.data_root / path for path in row["image_paths"]])
        label = read_burn_intensity_label(self.data_root / row["label_path"])
        return {
            "images": images,
            "label": label,
            "dates": np.arange(len(row["image_paths"]), dtype=np.int64),
        }


def build_backbone(checkpoint: Path):
    add_upstream_to_path()
    from hubconf import AnySat  # type: ignore[import-not-found]  # upstream checkout

    model = AnySat(model_size="base", flash_attn=False)
    if not random_init_requested():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.model.load_state_dict(state)
    return model


class AnySatT3(nn.Module):
    """Spectral adapter + AnySat's native time-series entrance + spatial head."""

    def __init__(self, backbone, task_bands: list[str], freeze_backbone: bool):
        super().__init__()
        self.backbone = backbone
        self.adapter = SpectralAdapter(task_bands)
        # A trainable adapter sits upstream of the backbone, so the graph
        # must stay alive through the frozen backbone or it gets no gradient.
        self.adapter_trainable = any(
            parameter.requires_grad for parameter in self.adapter.parameters()
        )
        self.head = LiteSpatialHead(2 * 768, NUM_CLASSES)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor, dates: torch.Tensor) -> torch.Tensor:
        adapted = self.adapter(images)  # [B,T,10,H,W]
        context = (
            torch.no_grad()
            if self.freeze_backbone and not self.adapter_trainable
            else torch.enable_grad()
        )
        with context:
            dense = self.backbone(
                {"s2": adapted, "s2_dates": dates},
                patch_size=PATCH_SIZE_M,
                output="dense",
                output_modality="s2",
            )
        dense = dense.permute(0, 3, 1, 2)
        return self.head(dense)


def _evaluate(model, dataset, means, stds, device) -> dict[str, float]:
    matrix = ConfusionMatrix(num_classes=NUM_CLASSES)
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2):
            images = (batch["images"].to(device) - means) / stds
            dates = batch["dates"].to(device).float()
            logits = model(images, dates)
            matrix.update(
                batch["label"].numpy(),
                logits.argmax(dim=1).cpu().numpy(),
                ignore_index=IGNORE_INDEX,
            )
    return multiclass_metrics(matrix)


def train_t3(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    checkpoint: Path,
    task_bands: list[str],
    freeze_backbone: bool,
    metric_name: str,
) -> int:
    run_id = request.run_id
    data_root = Path(request.data_root)
    device = os.environ.get("ANYSAT_WORKER_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    epochs = _int_env("ANYSAT_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("ANYSAT_WORKER_BATCH_SIZE", 2)
    lr = _float_env("ANYSAT_WORKER_LR", 1e-3)
    workers = _int_env("ANYSAT_WORKER_NUM_WORKERS", 4)
    limit = _int_env("ANYSAT_WORKER_LIMIT_BATCHES", 0) or None

    mean, std = burn_intensity_statistics(BANDS)
    means = torch.tensor(mean, device=device).view(1, TIME_STEPS, len(BANDS), 1, 1)
    stds = torch.tensor(std, device=device).view(1, TIME_STEPS, len(BANDS), 1, 1)

    backbone = build_backbone(checkpoint).to(device)
    model = AnySatT3(backbone, task_bands, freeze_backbone=freeze_backbone).to(device)
    adapter_params = count_parameters(model.adapter)
    head_params = count_parameters(model.head)

    manifest = Path(request.manifest_path)
    train_ds = BurnIntensityDataset(manifest, data_root, "train")
    select_ds = BurnIntensityDataset(manifest, data_root, SELECT_SPLIT)
    eval_ds = BurnIntensityDataset(manifest, data_root, EVAL_SPLIT)
    eval_limit = _int_env("ANYSAT_WORKER_EVAL_LIMIT", 0)
    if eval_limit:
        select_ds.rows = select_ds.rows[:eval_limit]
        eval_ds.rows = eval_ds.rows[:eval_limit]
    emit(
        run_id,
        "train.prepare",
        f"AnySat T3 native time series (freeze={freeze_backbone}): {len(train_ds)} train / "
        f"{len(select_ds)} {SELECT_SPLIT} / {len(eval_ds)} {EVAL_SPLIT}, "
        f"adapter {adapter_params} + head {head_params}",
    )
    writer.update_training(
        head_id="lite_spatial_head+spectral_adapter(native_time_series)",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="official_burn_intensity_stats_native_time_series")
    writer.update(
        tags={
            "adapter_parameters": str(adapter_params),
            "head_parameters": str(head_params),
            "backbone_frozen": str(freeze_backbone),
            "time_steps": str(TIME_STEPS),
            "date_layout": "one HLS raster per date (pre, during, post)",
            "label_mapping": "burn severity 0..4 kept (0 is a valid class)",
            "temporal_fusion": "native (model-internal); spatial head on the fused map",
            "head_note": "differs from Clay/Prithvi (per-date encoding + temporal head)",
            "select_split": SELECT_SPLIT,
            "eval_split": EVAL_SPLIT,
            "learning_rate": f"{lr:g}",
            "device": device,
        }
    )

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    best = BestCheckpoint()
    val_every = validation_every()

    started = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in loader:
            images = (batch["images"].to(device) - means) / stds
            labels = batch["label"].to(device)
            dates = batch["dates"].to(device).float()
            if int((labels != IGNORE_INDEX).sum()) == 0:
                continue
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(images, dates)
            loss = criterion(logits.float(), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
            if limit and steps >= limit:
                break
        mean_loss = total_loss / max(steps, 1)
        writer.add_curve("loss", "train", epoch, mean_loss)
        emit(
            run_id,
            "train.epoch",
            f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}",
            progress=(epoch + 1) / epochs,
        )
        if (epoch + 1) % val_every == 0 or epoch + 1 == epochs:
            validation = _evaluate(model, select_ds, means, stds, device)
            writer.add_curve("mean_iou", SELECT_SPLIT, epoch, validation["mean_iou"])
            best.offer(validation["mean_iou"], epoch + 1, model)
    wall = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

    best.restore(model)
    writer.update(
        tags={
            "selection": "best_validation_mean_iou",
            "selected_epoch": str(best.epoch),
            "validation_mean_iou": f"{best.score:.4f}",
        }
    )
    emit(run_id, "train.eval", f"evaluating on {len(eval_ds)} {EVAL_SPLIT} chips")
    metrics = _evaluate(model, eval_ds, means, stds, device)
    writer.add_metrics(
        to_metric_records(metrics, primary_name=metric_name, direction="maximize", split=EVAL_SPLIT)
    )
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=wall, peak_gpu_memory_mb=peak_mb
    )
    summary = (
        f"{metric_name}={metrics['mean_iou']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"({epochs} epochs, adapter {adapter_params} + head {head_params}, "
        f"{'frozen' if freeze_backbone else 'fine-tuned'}, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]" if limit else " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
