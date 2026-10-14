"""T5 (US multi-temporal crop) training for the Prithvi-EO-2.0 worker.

Each chip is a single 18-band raster holding three dates (date-major, per the
official loader). The encoder runs per date (224 px chips), producing a
``[B,T,1024,14,14]`` feature stack that the shared
``TemporalSegmentationHead`` fuses along time. Labels follow the official
``reduce_zero_label`` recipe (0 -> ignore, classes 1..13 -> 0..12).

The backbone is frozen for B2 (features cached per date) and fine-tuned for
B1, driven by the protocol; evaluation runs on ``official_validation``.
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
from rs_backbone_eval.evaluation.groups import GroupedSegmentation
from rs_backbone_eval.evaluation.segmentation import multiclass_metrics
from rs_backbone_eval.heads import TemporalSegmentationHead, count_parameters
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.splits import evaluation_split
from rs_backbone_eval.temporal_io import (
    multicrop_statistics,
    read_multicrop_label,
    read_stacked_dates,
)
from rs_backbone_eval.timeutil import day_of_year
from rs_backbone_eval.worker_runtime import RunWriter

BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
NUM_CLASSES = 13
IGNORE_INDEX = -1
TIME_STEPS = 3
CHIP_SIZE = 224
TRAIN_CROP = 224
OFFICIAL_EPOCHS = 125
EVAL_SPLIT = evaluation_split("official_validation")


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


class MulticropDataset(Dataset):
    """Reads the stacked raster as [T,C,H,W] and the reduced label."""

    def __init__(
        self, manifest_path: Path, data_root: Path, split: str, *, crop: int | None = None
    ):
        self.data_root = data_root
        self.crop = crop
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
        images = read_stacked_dates(self.data_root / row["image_paths"][0], channels=len(BANDS))
        label = read_multicrop_label(self.data_root / row["label_path"])
        if self.crop is not None and self.crop < images.shape[-1]:
            offset = (images.shape[-1] - self.crop) // 2
            images = images[:, :, offset : offset + self.crop, offset : offset + self.crop]
            label = label[offset : offset + self.crop, offset : offset + self.crop]
        doy = [day_of_year(stamp) for stamp in row["timestamps"]]
        return {"images": images, "label": label, "dates": np.asarray(doy, dtype=np.int64)}


def build_encoder(checkpoint: Path):
    from terratorch.registry import BACKBONE_REGISTRY

    encoder = BACKBONE_REGISTRY.build(
        "prithvi_eo_v2_300_tl",
        pretrained=not random_init_requested(),
        ckpt_path=None if random_init_requested() else str(checkpoint),
        bands=BANDS,
    )
    encoder.eval()
    return encoder


class PrithviT5(nn.Module):
    def __init__(self, encoder, freeze_backbone: bool, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.encoder = encoder
        self.freeze_backbone = freeze_backbone
        self.head = TemporalSegmentationHead(1024, num_classes)
        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def encode_dates(self, images: torch.Tensor) -> torch.Tensor:
        """[B,T,C,H,W] -> [B,T,1024,h,w] per-date encoder features."""
        batch, time = images.shape[0], images.shape[1]
        flat = images.reshape(batch * time, *images.shape[2:])
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            layers = self.encoder(flat)
        tokens = layers[-1][:, 1:, :] if isinstance(layers, (list, tuple)) else layers[:, 1:, :]
        _, length, dim = tokens.shape
        grid = int(length**0.5)
        features = tokens.transpose(1, 2).reshape(batch * time, dim, grid, grid)
        return features.reshape(batch, time, dim, grid, grid)

    def forward(
        self, features: torch.Tensor, dates: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        return self.head(features, dates, mask, output_size=(CHIP_SIZE, CHIP_SIZE))


def train_t5(
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
    train_ds = MulticropDataset(manifest, data_root, "train", crop=TRAIN_CROP)
    eval_ds = MulticropDataset(manifest, data_root, EVAL_SPLIT)
    eval_limit = _int_env("PRITHVI_WORKER_EVAL_LIMIT", 0)
    if eval_limit:
        eval_ds.rows = eval_ds.rows[:eval_limit]
    mean, std = multicrop_statistics(BANDS)
    means = torch.tensor(mean, device=device).view(1, 1, -1, 1, 1)
    stds = torch.tensor(std, device=device).view(1, 1, -1, 1, 1)

    encoder = build_encoder(checkpoint).to(device)
    model = PrithviT5(encoder, freeze_backbone=freeze_backbone).to(device)
    writer.update_training(
        head_id="temporal_segmentation_head",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="official_multicrop_stats_reduce_zero_label")
    writer.update(
        tags={
            "head_parameters": str(count_parameters(model.head)),
            "backbone_frozen": str(freeze_backbone),
            "time_steps": str(TIME_STEPS),
            "band_layout": "date-major (T x C)",
            "label_mapping": "reduce_zero_label (0->ignore)",
            "eval_split": EVAL_SPLIT,
            "learning_rate": f"{lr:g}",
        }
    )
    emit(
        run_id,
        "train.prepare",
        f"Prithvi T5 (freeze={freeze_backbone}): {len(train_ds)} train / {len(eval_ds)} "
        f"{EVAL_SPLIT}, {TIME_STEPS} dates, head {count_parameters(model.head)} params",
    )

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    started = time.time()
    phases: dict[str, float] = {}
    torch.cuda.reset_peak_memory_stats()
    optimizer_steps = 0

    if freeze_backbone:
        emit(run_id, "train.cache", "extracting per-date frozen features")
        extract_started = time.time()
        train_features, train_dates, train_labels = _extract(model, train_ds, means, stds, device)
        eval_features, eval_dates, eval_labels = _extract(model, eval_ds, means, stds, device)
        phases["extract"] = time.time() - extract_started
        # Train the head against the FULL label grid (upsample the logits) instead
        # of a downsampled label: 14x14 labels keep only 1/256 of the pixels and
        # drop small plots and boundaries that interpolation cannot recover.
        train_started = time.time()
        for epoch in range(epochs):
            permutation = torch.randperm(train_features.shape[0])
            total_loss, steps = 0.0, 0
            for start in range(0, permutation.numel(), 8):
                index = permutation[start : start + 8]
                features = train_features[index].to(device)
                dates = train_dates[index].to(device)
                labels = train_labels[index].to(device)
                mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
                if int((labels != IGNORE_INDEX).sum()) == 0:
                    continue
                logits = model.head(features, dates, mask, output_size=(CHIP_SIZE, CHIP_SIZE))
                loss = criterion(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                steps += 1
                if limit and steps >= limit:
                    break
            optimizer_steps += steps
            mean_loss = total_loss / max(steps, 1)
            writer.add_curve("loss", "train", epoch, mean_loss)
            emit(
                run_id,
                "train.epoch",
                f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}",
                progress=(epoch + 1) / epochs,
            )
        phases["train"] = time.time() - train_started
        eval_features, eval_dates = eval_features.to(device), eval_dates.to(device)
    else:
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
        train_started = time.time()
        for epoch in range(epochs):
            model.train()
            total_loss, steps = 0.0, 0
            for batch in loader:
                images = (batch["images"].to(device) - means) / stds
                labels = batch["label"].to(device)
                dates = batch["dates"].to(device)
                if int((labels != IGNORE_INDEX).sum()) == 0:
                    continue
                features = model.encode_dates(images)
                mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
                logits = model(features, dates, mask)
                loss = criterion(logits, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                steps += 1
                if limit and steps >= limit:
                    break
            optimizer_steps += steps
            mean_loss = total_loss / max(steps, 1)
            writer.add_curve("loss", "train", epoch, mean_loss)
            emit(
                run_id,
                "train.epoch",
                f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}",
                progress=(epoch + 1) / epochs,
            )
        phases["train"] = time.time() - train_started
        eval_features, eval_dates = None, None

    wall = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

    emit(run_id, "train.eval", f"evaluating on {len(eval_ds)} {EVAL_SPLIT} chips")
    eval_started = time.time()
    collector = GroupedSegmentation(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    model.eval()
    with torch.inference_mode():
        if freeze_backbone:
            for start in range(0, eval_features.shape[0], 8):
                features = eval_features[start : start + 8]
                dates = eval_dates[start : start + 8]
                mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
                logits = model.head(features, dates, mask, output_size=(CHIP_SIZE, CHIP_SIZE))
                collector.update(
                    [row["group_id"] for row in eval_ds.rows[start : start + 8]],
                    eval_labels[start : start + 8].numpy(),
                    logits.argmax(dim=1).cpu().numpy(),
                )
        else:
            loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)
            for index, batch in enumerate(loader):
                images = (batch["images"].to(device) - means) / stds
                dates = batch["dates"].to(device)
                features = model.encode_dates(images)
                mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
                logits = model(features, dates, mask)
                collector.update(
                    [eval_ds.rows[index]["group_id"]],
                    batch["label"].numpy(),
                    logits.argmax(dim=1).cpu().numpy(),
                )
    metrics = multiclass_metrics(collector.global_matrix)
    phases["eval"] = time.time() - eval_started
    writer.add_metrics(
        to_metric_records(metrics, primary_name=metric_name, direction="maximize", split=EVAL_SPLIT)
    )
    writer.add_confusion(collector.group_confusion(), NUM_CLASSES)
    writer.update_training(
        epochs_completed=epochs,
        wall_time_seconds=wall,
        peak_gpu_memory_mb=peak_mb,
        optimizer_steps=optimizer_steps,
        phases=phases,
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


def _extract(model: PrithviT5, dataset: MulticropDataset, means, stds, device):
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    features, dates, labels = [], [], []
    for batch in loader:
        images = (batch["images"].to(device) - means) / stds
        features.append(model.encode_dates(images).cpu())
        dates.append(batch["dates"])
        labels.append(batch["label"])
    return torch.cat(features), torch.cat(dates), torch.cat(labels)
