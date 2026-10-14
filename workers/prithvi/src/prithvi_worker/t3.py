"""T3 (burn scar intensity) training for the Prithvi-EO-2.0 worker.

Unlike T5, the three dates arrive as three separate six-band HLS rasters
(``pre`` / ``during`` / ``post``) instead of one stacked file. The encoder runs
per date and the shared ``TemporalSegmentationHead`` fuses the three
``[B,T,1024,14,14]`` feature stacks. Labels are the burn severity classes
0..4 (0 = "no burn" is a *valid* class, so there is no reduce-zero remap) and
the official per-date mean/std from TerraTorch are applied.

The backbone is frozen for B2 and fine-tuned for B1, driven by the protocol.
For the fine-tuning track the checkpoint is selected on the ``validation``
split and the reported metric is measured once on the untouched ``test`` split.
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
from rs_backbone_eval.heads import count_parameters
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.temporal_io import (
    burn_intensity_statistics,
    read_burn_intensity_label,
    read_hls_dates,
)
from rs_backbone_eval.worker_runtime import RunWriter

from .t5 import PrithviT5, build_encoder

BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR_1", "SWIR_2"]
NUM_CLASSES = 5
IGNORE_INDEX = -1
TIME_STEPS = 3
CHIP_SIZE = 224
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
    """Reads one HLS raster per date as ``[T,C,H,W]`` and the severity label."""

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


def _head_metrics(model, features, dates, labels, device) -> dict[str, float]:
    """Evaluate the temporal head on cached ``[N,T,1024,h,w]`` features."""
    matrix = ConfusionMatrix(num_classes=NUM_CLASSES)
    for start in range(0, features.shape[0], 8):
        batch = features[start : start + 8].to(device)
        mask = torch.ones(batch.shape[:2], dtype=torch.bool, device=device)
        logits = model.head(
            batch, dates[start : start + 8].to(device), mask, output_size=(CHIP_SIZE, CHIP_SIZE)
        )
        matrix.update(
            labels[start : start + 8].numpy(),
            logits.argmax(dim=1).cpu().numpy(),
            ignore_index=IGNORE_INDEX,
        )
    return multiclass_metrics(matrix)


def _extract(model, dataset, means, stds, device, batch_size: int = 8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    features, dates, labels = [], [], []
    for batch in loader:
        images = (batch["images"].to(device) - means) / stds
        features.append(model.encode_dates(images).cpu())
        dates.append(batch["dates"])
        labels.append(batch["label"])
    if not features:
        return torch.empty(0), torch.empty(0), torch.empty(0)
    return torch.cat(features), torch.cat(dates), torch.cat(labels)


def _eval_finetune(model, dataset, means, stds, device) -> dict[str, float]:
    matrix = ConfusionMatrix(num_classes=NUM_CLASSES)
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2):
            images = (batch["images"].to(device) - means) / stds
            features = model.encode_dates(images)
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
            logits = model(features, batch["dates"].to(device), mask)
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
    freeze_backbone: bool,
    metric_name: str,
) -> int:
    run_id = request.run_id
    data_root = Path(request.data_root)
    device = os.environ.get("PRITHVI_WORKER_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    epochs = _int_env("PRITHVI_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("PRITHVI_WORKER_BATCH_SIZE", 4)
    lr = _float_env("PRITHVI_WORKER_LR", 1e-3)
    workers = _int_env("PRITHVI_WORKER_NUM_WORKERS", 4)
    limit = _int_env("PRITHVI_WORKER_LIMIT_BATCHES", 0) or None

    manifest = Path(request.manifest_path)
    train_ds = BurnIntensityDataset(manifest, data_root, "train")
    select_ds = BurnIntensityDataset(manifest, data_root, SELECT_SPLIT)
    eval_ds = BurnIntensityDataset(manifest, data_root, EVAL_SPLIT)
    eval_limit = _int_env("PRITHVI_WORKER_EVAL_LIMIT", 0)
    if eval_limit:
        select_ds.rows = select_ds.rows[:eval_limit]
        eval_ds.rows = eval_ds.rows[:eval_limit]
    mean, std = burn_intensity_statistics(BANDS)
    means = torch.tensor(mean, device=device).view(1, TIME_STEPS, len(BANDS), 1, 1)
    stds = torch.tensor(std, device=device).view(1, TIME_STEPS, len(BANDS), 1, 1)

    encoder = build_encoder(checkpoint).to(device)
    model = PrithviT5(encoder, freeze_backbone=freeze_backbone, num_classes=NUM_CLASSES).to(device)
    writer.update_training(
        head_id="temporal_segmentation_head",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="official_burn_intensity_per_date_stats")
    writer.update(
        tags={
            "head_parameters": str(count_parameters(model.head)),
            "backbone_frozen": str(freeze_backbone),
            "time_steps": str(TIME_STEPS),
            "date_layout": "one HLS raster per date (pre, during, post)",
            "label_mapping": "burn severity 0..4 kept (0 is a valid class)",
            "select_split": SELECT_SPLIT,
            "eval_split": EVAL_SPLIT,
            "learning_rate": f"{lr:g}",
        }
    )
    emit(
        run_id,
        "train.prepare",
        f"Prithvi T3 (freeze={freeze_backbone}): {len(train_ds)} train / "
        f"{len(select_ds)} {SELECT_SPLIT} / {len(eval_ds)} {EVAL_SPLIT}, "
        f"head {count_parameters(model.head)} params",
    )

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    best = BestCheckpoint()
    val_every = validation_every()
    started = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if freeze_backbone:
        emit(run_id, "train.cache", "extracting per-date frozen features")
        train_features, train_dates, train_labels = _extract(model, train_ds, means, stds, device)
        select_features, select_dates, select_labels = _extract(
            model, select_ds, means, stds, device
        )
        grid = train_features.shape[-1]
        labels_grid = (
            torch.nn.functional.interpolate(
                train_labels.unsqueeze(1).float(), size=(grid, grid), mode="nearest"
            )
            .squeeze(1)
            .long()
        )
        for epoch in range(epochs):
            permutation = torch.randperm(train_features.shape[0])
            total_loss, steps = 0.0, 0
            model.head.train()
            for start in range(0, permutation.numel(), 8):
                index = permutation[start : start + 8]
                features = train_features[index].to(device)
                dates = train_dates[index].to(device)
                labels = labels_grid[index].to(device)
                mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
                if int((labels != IGNORE_INDEX).sum()) == 0:
                    continue
                logits = model.head(features, dates, mask)
                loss = criterion(logits, labels)
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
                model.head.eval()
                validation = _head_metrics(
                    model, select_features, select_dates, select_labels, device
                )
                writer.add_curve("mean_iou", SELECT_SPLIT, epoch, validation["mean_iou"])
                best.offer(validation["mean_iou"], epoch + 1, model.head)
        best.restore(model.head)
        eval_features, eval_dates, eval_labels = _extract(model, eval_ds, means, stds, device)
        metrics = _head_metrics(model, eval_features, eval_dates, eval_labels, device)
    else:
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
        for epoch in range(epochs):
            model.train()
            total_loss, steps = 0.0, 0
            for batch in loader:
                images = (batch["images"].to(device) - means) / stds
                labels = batch["label"].to(device)
                if int((labels != IGNORE_INDEX).sum()) == 0:
                    continue
                features = model.encode_dates(images)
                mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
                logits = model(features, batch["dates"].to(device), mask)
                loss = criterion(logits, labels)
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
                validation = _eval_finetune(model, select_ds, means, stds, device)
                writer.add_curve("mean_iou", SELECT_SPLIT, epoch, validation["mean_iou"])
                best.offer(validation["mean_iou"], epoch + 1, model)
        best.restore(model)
        metrics = _eval_finetune(model, eval_ds, means, stds, device)

    wall = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
    writer.update(
        tags={
            "selection": "best_validation_mean_iou",
            "selected_epoch": str(best.epoch),
            "validation_mean_iou": f"{best.score:.4f}",
        }
    )
    writer.add_metrics(
        to_metric_records(metrics, primary_name=metric_name, direction="maximize", split=EVAL_SPLIT)
    )
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=wall, peak_gpu_memory_mb=peak_mb
    )
    summary = (
        f"{metric_name}={metrics['mean_iou']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"({epochs} epochs, head {count_parameters(model.head)} params, "
        f"{'frozen' if freeze_backbone else 'fine-tuned'}, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]" if limit else " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
