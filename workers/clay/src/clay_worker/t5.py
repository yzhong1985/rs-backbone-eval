"""T5 (US multi-temporal crop) training for the Clay v1.5 worker.

The chip is one 18-band raster holding three dates (date-major, per the
official loader). Clay encodes each date separately (224 px chips -> 28x28
token grids) and the shared ``TemporalSegmentationHead`` fuses the dates along
time. Normalization uses the official multicrop per-band statistics (matching
the Prithvi protocol) with Clay's HLS wavelength metadata; labels follow
``reduce_zero_label`` (0 -> ignore, classes 1..13 -> 0..12).
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
from rs_backbone_eval.cache import cache_key
from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
from rs_backbone_eval.evaluation.segmentation import multiclass_metrics
from rs_backbone_eval.heads import TemporalSegmentationHead, count_parameters
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.temporal_io import (
    multicrop_statistics,
    read_multicrop_label,
    read_stacked_dates,
)
from rs_backbone_eval.timeutil import day_of_year
from rs_backbone_eval.worker_runtime import RunWriter

from .clay_probe import add_upstream_to_path, prepare_encoder
from .t1 import CLAY_LARGE

BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
HLS_CLAY_BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]
NUM_CLASSES = 13
IGNORE_INDEX = -1
TIME_STEPS = 3
CHIP_SIZE = 224
OFFICIAL_EPOCHS = 125
EVAL_SPLIT = "official_validation"
SENSOR = "hls-multicrop"


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
        images = read_stacked_dates(self.data_root / row["image_paths"][0], channels=len(BANDS))
        label = read_multicrop_label(self.data_root / row["label_path"])
        doy = [day_of_year(stamp) for stamp in row["timestamps"]]
        return {"images": images, "label": label, "dates": np.asarray(doy, dtype=np.int64)}


class ClayT5(nn.Module):
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
        """[B,T,C,H,W] -> [B,T,1024,h,w] per-date token grids."""
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
    model = ClayT5(encoder, gsd=30.0, sensor=SENSOR, freeze_backbone=freeze_backbone).to(device)
    head_params = count_parameters(model.head)
    waves = torch.tensor(
        [HLS_WAVELENGTHS_UM[band] for band in BANDS], dtype=torch.float32, device=device
    )
    mean, std = multicrop_statistics(BANDS)
    means = torch.tensor(mean, device=device).view(1, 1, -1, 1, 1)
    stds = torch.tensor(std, device=device).view(1, 1, -1, 1, 1)

    manifest = Path(request.manifest_path)
    train_ds = MulticropDataset(manifest, data_root, "train")
    eval_ds = MulticropDataset(manifest, data_root, EVAL_SPLIT)
    eval_limit = _int_env("CLAY_WORKER_EVAL_LIMIT", 0)
    if eval_limit:
        eval_ds.rows = eval_ds.rows[:eval_limit]
    emit(
        run_id,
        "train.prepare",
        f"Clay T5 (freeze={freeze_backbone}): {len(train_ds)} train / {len(eval_ds)} "
        f"{EVAL_SPLIT}, {TIME_STEPS} dates, head {head_params} params",
    )
    writer.update_training(
        head_id="temporal_segmentation_head",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="official_multicrop_stats_reduce_zero_label")
    writer.update(
        tags={
            "head_parameters": str(head_params),
            "backbone_frozen": str(freeze_backbone),
            "time_steps": str(TIME_STEPS),
            "band_layout": "date-major (T x C)",
            "label_mapping": "reduce_zero_label (0->ignore)",
            "derived_metadata_bands": ",".join(HLS_CLAY_BANDS),
            "eval_split": EVAL_SPLIT,
            "learning_rate": f"{lr:g}",
            "device": device,
        }
    )

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )

    if freeze_backbone:
        # Clay's frozen features are large (26 GB for T5), so they are cached
        # on disk as float16 memmaps and the head trains from them directly.
        train_cache = _feature_cache(
            model, train_ds, waves, means, stds, device, request, emit, "train"
        )
        eval_cache = _feature_cache(
            model, eval_ds, waves, means, stds, device, request, emit, EVAL_SPLIT
        )
        return _train_from_cache(
            model,
            train_cache,
            eval_cache,
            train_ds,
            eval_ds,
            optimizer,
            criterion,
            epochs,
            limit,
            emit,
            writer,
            run_id,
            head_params,
            freeze_backbone,
            device,
        )

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)

    started = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in loader:
            images = (batch["images"].to(device) - means) / stds
            labels = batch["label"].to(device)
            dates = batch["dates"].to(device)
            if int((labels != IGNORE_INDEX).sum()) == 0:
                continue
            features = model.encode_dates(images, waves)
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
        mean_loss = total_loss / max(steps, 1)
        writer.add_curve("loss", "train", epoch, mean_loss)
        emit(
            run_id,
            "train.epoch",
            f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}",
            progress=(epoch + 1) / epochs,
        )
    wall = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

    emit(run_id, "train.eval", f"evaluating on {len(eval_ds)} {EVAL_SPLIT} chips")
    matrix = ConfusionMatrix(num_classes=NUM_CLASSES)
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2):
            images = (batch["images"].to(device) - means) / stds
            dates = batch["dates"].to(device)
            features = model.encode_dates(images, waves)
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
            logits = model(features, dates, mask)
            matrix.update(
                batch["label"].numpy(),
                logits.argmax(dim=1).cpu().numpy(),
                ignore_index=IGNORE_INDEX,
            )
    metrics = multiclass_metrics(matrix)
    writer.add_metrics(
        to_metric_records(metrics, primary_name=metric_name, direction="maximize", split=EVAL_SPLIT)
    )
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=wall, peak_gpu_memory_mb=peak_mb
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


def _feature_cache(model, dataset, waves, means, stds, device, request, emit, split):
    """Extract per-date frozen features to a float16 memmap; return (path, shape)."""
    import numpy as np

    key = cache_key(
        Path(request.manifest_path), CLAY_LARGE["patch_size"], CHIP_SIZE, split, len(dataset)
    )
    cache_dir = Path(request.cache_root) / "clay_t5_features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.npy"
    grid = CHIP_SIZE // CLAY_LARGE["patch_size"]
    shape = (len(dataset), TIME_STEPS, CLAY_LARGE["dim"], grid, grid)
    if path.is_file():
        emit(emit_run_id(request), "train.cache", f"{split}: reusing feature cache {path.name}")
        return path, shape
    emit(
        emit_run_id(request),
        "train.cache",
        f"{split}: extracting {len(dataset)} chips to {path.name}",
    )
    memmap = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    cursor = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = (batch["images"].to(device) - means) / stds
            features = model.encode_dates(images, waves)
            count = features.shape[0]
            memmap[cursor : cursor + count] = features.to(torch.float16).cpu().numpy()
            cursor += count
    memmap.flush()
    return path, shape


def emit_run_id(request) -> str:
    return request.run_id


def _train_from_cache(
    model,
    train_cache,
    eval_cache,
    train_ds,
    eval_ds,
    optimizer,
    criterion,
    epochs,
    limit,
    emit,
    writer,
    run_id,
    head_params,
    freeze_backbone,
    device,
):
    """Train the temporal head on cached per-date features; evaluate on full chips."""
    import numpy as np

    train_path, train_shape = train_cache
    eval_path, _ = eval_cache
    train_features = np.load(train_path, mmap_mode="r")
    # Train against the FULL label grid (upsample the logits) instead of a
    # downsampled label, which would keep only 1/64 of the pixels.
    train_labels = _read_labels(train_ds, 0, train_shape[0])
    started = time.time()
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
            dates = torch.zeros(features.shape[0], TIME_STEPS, dtype=torch.float32, device=device)
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
    wall = time.time() - started

    emit(run_id, "train.eval", f"evaluating on {len(eval_ds)} cached chips")
    eval_features = np.load(eval_path, mmap_mode="r")
    matrix = ConfusionMatrix(num_classes=NUM_CLASSES)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(eval_ds), 4):
            features = torch.from_numpy(
                np.asarray(eval_features[start : start + 4], dtype=np.float32)
            ).to(device)
            dates = torch.zeros(features.shape[0], TIME_STEPS, dtype=torch.float32, device=device)
            mask = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
            logits = model.head(features, dates, mask, output_size=(CHIP_SIZE, CHIP_SIZE))
            labels = _read_labels(eval_ds, start, start + 4)
            matrix.update(
                labels.numpy(), logits.argmax(dim=1).cpu().numpy(), ignore_index=IGNORE_INDEX
            )
    metrics = multiclass_metrics(matrix)
    writer.add_metrics(
        to_metric_records(metrics, primary_name="mean_iou", direction="maximize", split=EVAL_SPLIT)
    )
    writer.update_training(epochs_completed=epochs, wall_time_seconds=wall)
    summary = (
        f"mean_iou={metrics['mean_iou']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"({epochs} epochs, head {head_params} params, frozen, cached features, "
        f"{wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]" if limit else " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0


def _read_labels(dataset, start: int, stop: int) -> torch.Tensor:
    from rs_backbone_eval.temporal_io import read_multicrop_label

    labels = [
        read_multicrop_label(dataset.data_root / row["label_path"])
        for row in dataset.rows[start:stop]
    ]
    return torch.from_numpy(np.stack(labels))
