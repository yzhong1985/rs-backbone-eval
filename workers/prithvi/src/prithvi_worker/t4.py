"""T4 (Landslide4Sense) training for the Prithvi-EO-2.0 worker.

The six official bands (BLUE, GREEN, RED, NIR_BROAD, SWIR_1, SWIR_2) are
subset from the 14-band HDF5 patch and resized 128 -> 224 (the official config)
before the encoder; the shared ``LiteSpatialHead`` decodes the 14x14 token grid.
The HuggingFace copy's scale does not match TerraTorch's published statistics,
so train-split mean/std are used (identical for every backbone).

The backbone is frozen for B2 and fine-tuned for B1; both tracks select the
head/checkpoint on ``validation`` and report ``test`` once. Labels are binary
(0 = non-landslide, 1 = landslide) with no ignore index.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
from rs_backbone_eval.evaluation.segmentation import binary_metrics
from rs_backbone_eval.heads import LiteSpatialHead, count_parameters
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.temporal_io import landslide_statistics, read_landslide_sample
from rs_backbone_eval.worker_runtime import RunWriter

BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]
ENCODER_INPUT = 224
LABEL_SIZE = 224
OFFICIAL_EPOCHS = 50
SELECT_SPLIT = "validation"
EVAL_SPLIT = "test"
IGNORE_INDEX = -1


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


class LandslideDataset(Dataset):
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
        image, label = read_landslide_sample(
            self.data_root / row["image_paths"][0],
            self.data_root / row["label_path"],
            BANDS,
            size=ENCODER_INPUT,
        )
        return {"image": image, "label": label}


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


class PrithviT4(nn.Module):
    def __init__(self, encoder, freeze_backbone: bool):
        super().__init__()
        self.encoder = encoder
        self.freeze_backbone = freeze_backbone
        self.head = LiteSpatialHead(1024, 2)
        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def features(self, images: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            layers = self.encoder(images)
        tokens = layers[-1][:, 1:, :] if isinstance(layers, (list, tuple)) else layers[:, 1:, :]
        batch, length, dim = tokens.shape
        grid = int(length**0.5)
        return tokens.transpose(1, 2).reshape(batch, dim, grid, grid)

    def forward(self, features: torch.Tensor, size: int) -> torch.Tensor:
        logits = self.head(features)
        return torch.nn.functional.interpolate(
            logits, size=(size, size), mode="bilinear", align_corners=False
        )


def _extract(model, dataset, means, stds, device):
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    features, labels = [], []
    for batch in loader:
        images = (batch["image"].to(device) - means) / stds
        features.append(model.features(images).cpu())
        labels.append(batch["label"])
    if not features:
        return torch.empty(0), torch.empty(0)
    return torch.cat(features), torch.cat(labels)


def _metrics_from_features(model, features, labels, device) -> dict[str, float]:
    matrix = ConfusionMatrix(num_classes=2)
    for start in range(0, features.shape[0], 8):
        logits = model(features[start : start + 8].to(device), size=LABEL_SIZE)
        matrix.update(
            labels[start : start + 8].numpy(),
            logits.argmax(dim=1).cpu().numpy(),
            ignore_index=IGNORE_INDEX,
        )
    return binary_metrics(matrix, foreground=1)


def _eval_live(model, dataset, means, stds, device) -> dict[str, float]:
    matrix = ConfusionMatrix(num_classes=2)
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2):
            images = (batch["image"].to(device) - means) / stds
            logits = model(model.features(images), size=LABEL_SIZE)
            matrix.update(
                batch["label"].numpy(),
                logits.argmax(dim=1).cpu().numpy(),
                ignore_index=IGNORE_INDEX,
            )
    return binary_metrics(matrix, foreground=1)


def train_t4(
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
    lr = _float_env("PRITHVI_WORKER_LR", 1e-4)
    workers = _int_env("PRITHVI_WORKER_NUM_WORKERS", 4)
    limit = _int_env("PRITHVI_WORKER_LIMIT_BATCHES", 0) or None

    mean, std = landslide_statistics(BANDS)
    means = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    stds = torch.tensor(std, device=device).view(1, -1, 1, 1)

    manifest = Path(request.manifest_path)
    train_ds = LandslideDataset(manifest, data_root, "train")
    select_ds = LandslideDataset(manifest, data_root, SELECT_SPLIT)
    eval_ds = LandslideDataset(manifest, data_root, EVAL_SPLIT)
    eval_limit = _int_env("PRITHVI_WORKER_EVAL_LIMIT", 0)
    if eval_limit:
        select_ds.rows = select_ds.rows[:eval_limit]
        eval_ds.rows = eval_ds.rows[:eval_limit]
    encoder = build_encoder(checkpoint).to(device)
    model = PrithviT4(encoder, freeze_backbone=freeze_backbone).to(device)
    writer.update_training(
        head_id="lite_spatial_head",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="landslide_train_split_stats_resize224")
    writer.update(
        tags={
            "head_parameters": str(count_parameters(model.head)),
            "backbone_frozen": str(freeze_backbone),
            "encoder_input_px": str(ENCODER_INPUT),
            "feature_grid": "14x14",
            "bands_source": "subset of the 14-band Landslide4Sense stack",
            "normalization": "train-split statistics (shipped scale differs from TerraTorch)",
            "select_split": SELECT_SPLIT,
            "eval_split": EVAL_SPLIT,
            "learning_rate": f"{lr:g}",
        }
    )
    emit(
        run_id,
        "train.prepare",
        f"Prithvi T4 (freeze={freeze_backbone}): {len(train_ds)} train / {len(select_ds)} "
        f"{SELECT_SPLIT} / {len(eval_ds)} {EVAL_SPLIT}",
    )

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05
    )
    best = BestCheckpoint()
    val_every = validation_every()
    started = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if freeze_backbone:
        emit(run_id, "train.cache", "extracting frozen features")
        train_features, train_labels = _extract(model, train_ds, means, stds, device)
        select_features, select_labels = _extract(model, select_ds, means, stds, device)
        grid = train_features.shape[-1]
        train_grid = (
            torch.nn.functional.interpolate(
                train_labels.unsqueeze(1).float(), size=(grid, grid), mode="nearest"
            )
            .squeeze(1)
            .long()
            .to(device)
        )
        train_features = train_features.to(device)
        select_features = select_features.to(device)
        for epoch in range(epochs):
            model.train()
            permutation = torch.randperm(train_features.shape[0], device=device)
            total_loss, steps = 0.0, 0
            for start in range(0, permutation.numel(), 8):
                index = permutation[start : start + 8]
                logits = model.head(train_features[index])
                loss = criterion(logits, train_grid[index])
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
                model.eval()
                matrix = ConfusionMatrix(num_classes=2)
                with torch.inference_mode():
                    for start in range(0, select_features.shape[0], 8):
                        logits = model(select_features[start : start + 8], size=LABEL_SIZE)
                        matrix.update(
                            select_labels[start : start + 8].numpy(),
                            logits.argmax(dim=1).cpu().numpy(),
                            ignore_index=IGNORE_INDEX,
                        )
                validation = binary_metrics(matrix, foreground=1)
                writer.add_curve("mean_iou", SELECT_SPLIT, epoch, validation["mean_iou"])
                best.offer(validation["mean_iou"], epoch + 1, model.head)
        best.restore(model.head)
        metrics = _metrics_from_features(
            model, *_extract(model, eval_ds, means, stds, device), device
        )
    else:
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
        for epoch in range(epochs):
            model.train()
            total_loss, steps = 0.0, 0
            for batch in loader:
                images = (batch["image"].to(device) - means) / stds
                labels = batch["label"].to(device)
                logits = model(model.features(images), size=images.shape[-1])
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
                validation = _eval_live(model, select_ds, means, stds, device)
                writer.add_curve("mean_iou", SELECT_SPLIT, epoch, validation["mean_iou"])
                best.offer(validation["mean_iou"], epoch + 1, model)
        best.restore(model)
        metrics = _eval_live(model, eval_ds, means, stds, device)

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
        f"{metric_name}={metrics[metric_name]:.4f} "
        f"foreground_iou={metrics['foreground_iou']:.4f} "
        f"({epochs} epochs, head {count_parameters(model.head)} params, "
        f"{'frozen' if freeze_backbone else 'fine-tuned'}, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]" if limit else " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
