"""T1 full fine-tuning for Prithvi-EO-2.0 (B1 track) with the shared head.

B1 differs from the B2 probe only in that the backbone is updated. To keep the
three models in one table the decoder is the same shared ``LiteSpatialHead``
(not UPerNet): the *only* variable is the backbone. Input follows the same
contract as the AnySat/Clay tracks: 256 px training crops from the 512 chips
and full-512 evaluation against the same labels.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
from rs_backbone_eval.evaluation.segmentation import binary_metrics
from rs_backbone_eval.heads import LiteSpatialHead, count_parameters
from rs_backbone_eval.worker_runtime import RunWriter

from .t1_frozen import T1ChipDataset, build_encoder

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


class PrithviCropDataset(T1ChipDataset):
    """T1 chips with optional center crop for fine-tuning."""

    def __init__(self, manifest_path: Path, data_root: Path, split: str, crop: int | None):
        super().__init__(manifest_path, data_root, split)
        self.crop = crop

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        image, label = item["image"], item["label"]
        if self.crop is not None and self.crop < image.shape[-1]:
            offset = (image.shape[-1] - self.crop) // 2
            image = image[:, offset : offset + self.crop, offset : offset + self.crop]
            label = label[offset : offset + self.crop, offset : offset + self.crop]
        return {"image": image, "label": label}


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
    eval_ds = PrithviCropDataset(manifest, data_root, "test", crop=None)

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

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
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

    emit(run_id, "train.eval", f"evaluating on {len(eval_ds)} test chips at full 512")
    model.eval()
    matrix = ConfusionMatrix(num_classes=2)
    with torch.inference_mode():
        for batch in DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2):
            logits = model(batch["image"].to(device))
            matrix.update(
                batch["label"].numpy(), logits.argmax(dim=1).cpu().numpy(), ignore_index=-1
            )
    metrics = binary_metrics(matrix, foreground=1)
    metrics["water_iou"] = metrics.pop("foreground_iou")
    writer.add_metrics(
        to_metric_records(metrics, primary_name="water_iou", direction="maximize", split="test")
    )
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=wall, peak_gpu_memory_mb=peak_mb
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
