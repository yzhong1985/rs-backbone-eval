"""T1 full fine-tuning for Prithvi-EO-2.0 (B1 track) with the shared head.

B1 differs from the B2 probe only in that the backbone is updated. To keep the
three models in one table the decoder is the same shared ``LiteSpatialHead``
(not UPerNet): the *only* variable is the backbone. Input follows the same
contract as the AnySat/Clay tracks: 256 px training crops from the 512 chips
and full-512 evaluation against the same labels.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.groups import GroupedSegmentation
from rs_backbone_eval.evaluation.segmentation import binary_metrics
from rs_backbone_eval.heads import LiteSpatialHead, count_parameters
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.splits import evaluation_split, extra_evaluation_splits
from rs_backbone_eval.worker_runtime import RunWriter

from .t1_frozen import BANDS, band_stats, build_encoder, chip_indices_for_bands

EVAL_SPLIT = evaluation_split("test")

OFFICIAL_EPOCHS = 50
TRAIN_CROP = 256
BATCH_SIZE = 1


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


class PrithviCropDataset(Dataset):
    """Native 512 T1 chips (no resize) with an optional center crop.

    Fine-tuning keeps the native 10 m resolution, matching the AnySat/Clay B1
    input contract; normalization is the same constant_scale + band statistics
    used by the frozen track.
    """

    def __init__(self, manifest_path: Path, data_root: Path, split: str, crop: int | None):
        self.data_root = data_root
        self.crop = crop
        self.indices = chip_indices_for_bands(BANDS)
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
        with rasterio.open(self.data_root / row["image_paths"][0]) as source:
            image = source.read().astype(np.float32)[self.indices]
        with rasterio.open(self.data_root / row["label_path"]) as source:
            label = source.read(1).astype(np.int64)
        image = image * 0.0001
        mean, std = band_stats()
        image = (image - mean[:, None, None]) / std[:, None, None]
        if self.crop is not None and self.crop < image.shape[-1]:
            offset = (image.shape[-1] - self.crop) // 2
            image = image[:, offset : offset + self.crop, offset : offset + self.crop]
            label = label[offset : offset + self.crop, offset : offset + self.crop]
        return {"image": torch.from_numpy(image), "label": label}


class PrithviFineTune(nn.Module):
    def __init__(self, encoder, num_classes: int = 2):
        super().__init__()
        self.encoder = encoder
        self.head = LiteSpatialHead(1024, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        layers = self.encoder(images)
        tokens = layers[-1][:, 1:, :] if isinstance(layers, (list, tuple)) else layers[:, 1:, :]
        batch, length, dim = tokens.shape
        grid = int(length**0.5)
        features = tokens.transpose(1, 2).reshape(batch, dim, grid, grid)
        logits = self.head(features)
        return torch.nn.functional.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )


def train_finetune(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    checkpoint: Path,
) -> int:
    run_id = request.run_id
    data_root = Path(request.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = _int_env("PRITHVI_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("PRITHVI_WORKER_BATCH_SIZE", BATCH_SIZE)
    lr = _float_env("PRITHVI_WORKER_LR", 1e-4)
    workers = _int_env("PRITHVI_WORKER_NUM_WORKERS", 4)
    limit = _int_env("PRITHVI_WORKER_LIMIT_BATCHES", 0) or None
    crop = _int_env("PRITHVI_WORKER_TRAIN_CROP", TRAIN_CROP)

    manifest = Path(request.manifest_path)
    train_ds = PrithviCropDataset(manifest, data_root, "train", crop=crop)
    val_ds = PrithviCropDataset(manifest, data_root, "validation", crop=None)
    eval_ds = PrithviCropDataset(manifest, data_root, EVAL_SPLIT, crop=None)

    encoder = build_encoder(checkpoint)
    for parameter in encoder.parameters():
        parameter.requires_grad = True
    model = PrithviFineTune(encoder).to(device)
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model)
    writer.update_training(
        head_id="lite_spatial_head", trainable_parameters=trainable, total_parameters=total
    )
    writer.update_input(preprocessing_id="native512_crop256_restored_norm")
    writer.update(
        tags={
            "head_parameters": str(count_parameters(model.head)),
            "backbone_frozen": "False",
            "train_crop_px": str(crop),
            "learning_rate": f"{lr:g}",
            "device": device,
        }
    )
    emit(
        run_id,
        "train.prepare",
        f"Prithvi fine-tune (B1): {len(train_ds)} train / {len(eval_ds)} test chips, "
        f"crop {crop}, {trainable / 1e6:.1f}M trainable of {total / 1e6:.1f}M",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)

    best = BestCheckpoint()
    val_every = validation_every()

    def score(dataset) -> tuple[dict, dict[str, list[list[int]]]]:
        model.eval()
        collector = GroupedSegmentation(num_classes=2, ignore_index=-1)
        with torch.inference_mode():
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
            for index, batch in enumerate(loader):
                logits = model(batch["image"].to(device))
                collector.update(
                    [dataset.rows[index]["group_id"]],
                    batch["label"].numpy(),
                    logits.argmax(dim=1).cpu().numpy(),
                )
        result = binary_metrics(collector.global_matrix, foreground=1)
        result["water_iou"] = result.pop("foreground_iou")
        return result, collector.group_confusion()

    started = time.time()
    optimizer_steps = 0
    phases: dict[str, float] = {}
    torch.cuda.reset_peak_memory_stats()
    train_started = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            if int((labels != -1).sum()) == 0:
                continue
            logits = model(images)
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
        if (epoch + 1) % val_every == 0 or epoch + 1 == epochs:
            validation, _ = score(val_ds)
            writer.add_curve("water_iou", "validation", epoch, validation["water_iou"])
            best.offer(validation["water_iou"], epoch + 1, model)
    phases["train"] = time.time() - train_started
    wall = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

    best.restore(model)
    best.save(writer.run_dir / "best_checkpoint.pt")
    writer.add_artifact("checkpoint", "best_checkpoint.pt", "validation-selected weights")
    emit(
        run_id,
        "train.eval",
        f"selected epoch {best.epoch} (validation water_iou {best.score:.4f}); "
        f"evaluating on {len(eval_ds)} test chips at full 512",
    )
    eval_started = time.time()
    metrics, group_json = score(eval_ds)
    phases["eval"] = time.time() - eval_started
    writer.update(
        tags={
            "selection": "best_validation_water_iou",
            "selected_epoch": str(best.epoch),
            "validation_water_iou": f"{best.score:.4f}",
            "validation_every": str(val_every),
        }
    )
    writer.add_metrics(
        to_metric_records(metrics, primary_name="water_iou", direction="maximize", split=EVAL_SPLIT)
    )
    writer.add_confusion(group_json, 2)
    extra_splits = extra_evaluation_splits(EVAL_SPLIT)
    for split in extra_splits:
        holdout_ds = PrithviCropDataset(manifest, data_root, split, crop=None)
        holdout, _ = score(holdout_ds)
        writer.add_metrics(
            to_metric_records(
                holdout, primary_name="water_iou", direction="maximize", split=split, primary=False
            )
        )
        emit(
            run_id,
            "train.holdout",
            f"{split} ({len(holdout_ds)} chips): {holdout['water_iou']:.4f}",
        )
    if extra_splits:
        writer.update(tags={"extra_eval_splits": ",".join(extra_splits)})
    writer.update_training(
        epochs_completed=epochs,
        optimizer_steps=optimizer_steps,
        wall_time_seconds=wall,
        peak_gpu_memory_mb=peak_mb,
        phases=phases,
    )
    summary = (
        f"water_iou={metrics['water_iou']:.4f} mean_iou={metrics['mean_iou']:.4f} "
        f"({epochs} epochs, {trainable / 1e6:.1f}M trainable, fine-tuned, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
