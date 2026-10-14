"""T3 (burn scar intensity) training for the Clay v1.5 worker.

The three dates arrive as three separate six-band HLS rasters (pre/during/post).
Clay encodes each date separately (224 px chips -> 28x28 token grids) and the
shared ``TemporalSegmentationHead`` fuses them. Normalization uses the official
burn-intensity per-date statistics with Clay's HLS wavelength metadata; labels
are the severity classes 0..4 (0 = "no burn" is a *valid* class).

The backbone is frozen for B2 (per-date features cached to a float16 memmap)
and fine-tuned for B1. Both tracks select the head/checkpoint on ``validation``
and report ``test`` once.
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

from rs_backbone_eval.bands import HLS_WAVELENGTHS_UM
from rs_backbone_eval.cache import cache_key, model_identity
from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.groups import GroupedSegmentation
from rs_backbone_eval.evaluation.segmentation import multiclass_metrics
from rs_backbone_eval.heads import TemporalSegmentationHead, count_parameters
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.temporal_io import (
    burn_intensity_statistics,
    read_burn_intensity_label,
    read_hls_dates,
)
from rs_backbone_eval.worker_runtime import RunWriter

from .clay_probe import add_upstream_to_path, prepare_encoder
from .t1 import CLAY_LARGE

BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR_1", "SWIR_2"]
HLS_CLAY_BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]
NUM_CLASSES = 5
IGNORE_INDEX = -1
TIME_STEPS = 3
CHIP_SIZE = 224
OFFICIAL_EPOCHS = 50
SELECT_SPLIT = "validation"
EVAL_SPLIT = "test"
SENSOR = "hls"


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


class ClayT3(nn.Module):
    """Per-date Clay encoder + shared temporal head."""

    def __init__(self, encoder, gsd: float, sensor: str, freeze_backbone: bool):
        super().__init__()
        self.encoder = encoder
        self.gsd = gsd
        self.sensor = sensor
        self.freeze_backbone = freeze_backbone
        self.head = TemporalSegmentationHead(CLAY_LARGE["dim"], NUM_CLASSES)
        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def encode_dates(self, images: torch.Tensor, waves: torch.Tensor) -> torch.Tensor:
        batch, time, _, height, width = images.shape
        flat = images.reshape(batch * time, *images.shape[2:])
        datacube = {
            "pixels": flat,
            "time": torch.zeros(batch * time, 4, device=images.device),
            "latlon": torch.zeros(batch * time, 4, device=images.device),
            "gsd": torch.tensor(self.gsd, device=images.device),
            "waves": waves,
            "platform": self.sensor,
        }
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            tokens = self.encoder(datacube)
        patch = CLAY_LARGE["patch_size"]
        grid_h, grid_w = height // patch, width // patch
        features = tokens.transpose(1, 2).reshape(batch * time, CLAY_LARGE["dim"], grid_h, grid_w)
        return features.reshape(batch, time, CLAY_LARGE["dim"], grid_h, grid_w)

    def forward(self, features, dates, mask) -> torch.Tensor:
        return self.head(features, dates, mask, output_size=(CHIP_SIZE, CHIP_SIZE))


def _read_labels(dataset, start: int, stop: int) -> torch.Tensor:
    """Full-resolution labels; the loss runs against these, not a coarse grid."""
    labels = [
        read_burn_intensity_label(dataset.data_root / row["label_path"])
        for row in dataset.rows[start:stop]
    ]
    return torch.from_numpy(np.stack(labels))


def _feature_cache(
    model,
    dataset,
    waves,
    means,
    stds,
    device,
    cache_dir,
    manifest_path,
    split,
    emit,
    run_id,
    identity,
):
    """Extract per-date frozen features to a float16 memmap; return (path, shape).

    ``identity`` ties the cache to the weights (checkpoint + init/seed); a
    partially written or wrong-shape file is rejected and re-extracted, and the
    file is written to a temporary path and moved into place.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key(manifest_path, split, len(dataset), identity)}.npy"
    grid = CHIP_SIZE // CLAY_LARGE["patch_size"]
    shape = (len(dataset), TIME_STEPS, CLAY_LARGE["dim"], grid, grid)
    if path.is_file():
        try:
            existing = np.load(path, mmap_mode="r")
            if existing.shape == shape and existing.dtype == np.float16:
                emit(run_id, "train.cache", f"{split}: reusing feature cache {path.name}")
                return path, shape
        except (OSError, ValueError):
            pass
        emit(run_id, "train.cache", f"{split}: cache {path.name} incomplete; re-extracting")
    emit(run_id, "train.cache", f"{split}: extracting {len(dataset)} chips to {path.name}")
    temporary = path.with_suffix(".partial.npy")
    memmap = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
    cursor = 0
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4):
            images = (batch["images"].to(device) - means) / stds
            features = model.encode_dates(images, waves)
            count = features.shape[0]
            memmap[cursor : cursor + count] = features.to(torch.float16).cpu().numpy()
            cursor += count
    memmap.flush()
    del memmap
    if cursor != shape[0]:
        raise RuntimeError(f"{split}: extracted {cursor} of {shape[0]} chips")
    temporary.replace(path)
    return path, shape


def _sequence_dates(rows) -> torch.Tensor:
    """T3's temporal protocol is the observation order, not a calendar date."""
    return torch.tensor(
        [[float(index) for index in range(len(row["image_paths"]))] for row in rows],
        dtype=torch.float32,
    )


def _head_metrics(model, dataset, features, labels, device) -> tuple[dict, dict]:
    collector = GroupedSegmentation(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    for start in range(0, features.shape[0], 4):
        batch = torch.from_numpy(np.asarray(features[start : start + 4], dtype=np.float32)).to(
            device
        )
        rows = dataset.rows[start : start + 4]
        dates = _sequence_dates(rows).to(device)
        mask = torch.ones(batch.shape[:2], dtype=torch.bool, device=device)
        logits = model.head(batch, dates, mask, output_size=(CHIP_SIZE, CHIP_SIZE))
        collector.update(
            [row["group_id"] for row in rows],
            labels[start : start + 4].numpy(),
            logits.argmax(dim=1).cpu().numpy(),
        )
    return multiclass_metrics(collector.global_matrix), collector.group_confusion()


def _eval_live(model, dataset, waves, means, stds, device) -> tuple[dict, dict]:
    collector = GroupedSegmentation(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    model.eval()
    with torch.inference_mode():
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        for index, batch in enumerate(loader):
            images = (batch["images"].to(device) - means) / stds
            features = model.encode_dates(images, waves)
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
            logits = model(features, batch["dates"].to(device), mask)
            collector.update(
                [dataset.rows[index]["group_id"]],
                batch["label"].numpy(),
                logits.argmax(dim=1).cpu().numpy(),
            )
    return multiclass_metrics(collector.global_matrix), collector.group_confusion()


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
    device = os.environ.get("CLAY_WORKER_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    epochs = _int_env("CLAY_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("CLAY_WORKER_BATCH_SIZE", 4)
    lr = _float_env("CLAY_WORKER_LR", 1e-3)
    workers = _int_env("CLAY_WORKER_NUM_WORKERS", 4)
    limit = _int_env("CLAY_WORKER_LIMIT_BATCHES", 0) or None

    add_upstream_to_path()
    from claymodel.finetune.segment.factory import SegmentEncoder  # upstream checkout

    encoder = SegmentEncoder(
        **CLAY_LARGE, ckpt_path=None if random_init_requested() else str(checkpoint)
    ).to(device)
    prepare_encoder(encoder, freeze_backbone)
    model = ClayT3(encoder, gsd=30.0, sensor=SENSOR, freeze_backbone=freeze_backbone).to(device)
    head_params = count_parameters(model.head)
    waves = torch.tensor(
        [HLS_WAVELENGTHS_UM[band] for band in BANDS], dtype=torch.float32, device=device
    )
    mean, std = burn_intensity_statistics(BANDS)
    means = torch.tensor(mean, device=device).view(1, TIME_STEPS, len(BANDS), 1, 1)
    stds = torch.tensor(std, device=device).view(1, TIME_STEPS, len(BANDS), 1, 1)

    manifest = Path(request.manifest_path)
    train_ds = BurnIntensityDataset(manifest, data_root, "train")
    select_ds = BurnIntensityDataset(manifest, data_root, SELECT_SPLIT)
    eval_ds = BurnIntensityDataset(manifest, data_root, EVAL_SPLIT)
    eval_limit = _int_env("CLAY_WORKER_EVAL_LIMIT", 0)
    if eval_limit:
        select_ds.rows = select_ds.rows[:eval_limit]
        eval_ds.rows = eval_ds.rows[:eval_limit]
    emit(
        run_id,
        "train.prepare",
        f"Clay T3 (freeze={freeze_backbone}): {len(train_ds)} train / {len(select_ds)} "
        f"{SELECT_SPLIT} / {len(eval_ds)} {EVAL_SPLIT}, head {head_params} params",
    )
    writer.update_training(
        head_id="temporal_segmentation_head",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="official_burn_intensity_per_date_stats")
    writer.update(
        tags={
            "head_parameters": str(head_params),
            "backbone_frozen": str(freeze_backbone),
            "time_steps": str(TIME_STEPS),
            "date_layout": "one HLS raster per date (pre, during, post)",
            "label_mapping": "burn severity 0..4 kept (0 is a valid class)",
            "derived_metadata_bands": ",".join(HLS_CLAY_BANDS),
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
    best = BestCheckpoint()
    val_every = validation_every()
    started = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    if freeze_backbone:
        cache_dir = Path(request.cache_root) / "clay_t3_features"
        identity = model_identity(
            checkpoint, seed=request.seed, random_init=random_init_requested()
        )
        train_path, train_shape = _feature_cache(
            model,
            train_ds,
            waves,
            means,
            stds,
            device,
            cache_dir,
            manifest,
            "train",
            emit,
            run_id,
            identity,
        )
        select_path, _ = _feature_cache(
            model,
            select_ds,
            waves,
            means,
            stds,
            device,
            cache_dir,
            manifest,
            SELECT_SPLIT,
            emit,
            run_id,
            identity,
        )
        train_features = np.load(train_path, mmap_mode="r")
        select_features = np.load(select_path, mmap_mode="r")
        cache_bytes = Path(train_path).stat().st_size + Path(select_path).stat().st_size
        train_labels = _read_labels(train_ds, 0, train_shape[0])
        select_labels = _read_labels(select_ds, 0, len(select_ds))
        for epoch in range(epochs):
            model.train()
            total_loss, steps = 0.0, 0
            order = torch.randperm(train_shape[0])
            for start in range(0, order.numel(), 8):
                index = order[start : start + 8].numpy()
                features = torch.from_numpy(np.asarray(train_features[index], dtype=np.float32)).to(
                    device
                )
                labels = train_labels[index].to(device)
                dates = _sequence_dates([train_ds.rows[int(j)] for j in index]).to(device)
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
                validation, _ = _head_metrics(
                    model, select_ds, select_features, select_labels, device
                )
                writer.add_curve("mean_iou", SELECT_SPLIT, epoch, validation["mean_iou"])
                best.offer(validation["mean_iou"], epoch + 1, model.head)
        best.restore(model.head)
        best.save(writer.run_dir / "best_checkpoint.pt")
        writer.add_artifact("checkpoint", "best_checkpoint.pt", "validation-selected weights")
        metrics, group_json = _eval_live(model, eval_ds, waves, means, stds, device)
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
                features = model.encode_dates(images, waves)
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
                validation, _ = _eval_live(model, select_ds, waves, means, stds, device)
                writer.add_curve("mean_iou", SELECT_SPLIT, epoch, validation["mean_iou"])
                best.offer(validation["mean_iou"], epoch + 1, model)
        best.restore(model)
        best.save(writer.run_dir / "best_checkpoint.pt")
        writer.add_artifact("checkpoint", "best_checkpoint.pt", "validation-selected weights")
        metrics, group_json = _eval_live(model, eval_ds, waves, means, stds, device)

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
    writer.add_confusion(group_json, NUM_CLASSES)
    writer.update_training(
        epochs_completed=epochs,
        wall_time_seconds=wall,
        peak_gpu_memory_mb=peak_mb,
        cache_bytes=cache_bytes,
    )
    summary = (
        f"{metric_name}={metrics['mean_iou']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"({epochs} epochs, head {head_params} params, "
        f"{'frozen' if freeze_backbone else 'fine-tuned'}, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]" if limit else " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
