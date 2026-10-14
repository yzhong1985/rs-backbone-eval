"""T8 (Five-Billion-Pixels / GF-2) land cover for the Prithvi worker.

The Prithvi encoder has a six-band patch embedding, so the four GF-2 bands
(B/G/R/NIR) pass through a small explicit 1x1-conv adapter that synthesizes the
two missing bands (SWIR-1, SWIR-2). Zero padding stays forbidden.

Cross-domain by construction: 0.8 m GF-2 without SWIR versus Prithvi's 30 m HLS
pretraining. Scenes are ~6900x7300 px, so chips are read as rasterio windows.
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
from rasterio.windows import Window
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
from rs_backbone_eval.evaluation.segmentation import multiclass_metrics
from rs_backbone_eval.heads import LiteSpatialHead, count_parameters
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.worker_runtime import RunWriter

TASK_BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD"]
# The encoder's native six-band HLS order that the adapter must produce.
ENCODER_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
NUM_CLASSES = 25
IGNORE_INDEX = -1
CHIP_SIZE = 224
ENCODER_INPUT = 224
EMBED_DIM = 1024
ENCODER_PATCH = 16
GRID = ENCODER_INPUT // ENCODER_PATCH
GSD_M = 0.8
OFFICIAL_EPOCHS = 50


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


def scene_root() -> Path:
    return Path(os.environ.get("RS_EVAL_FBP_ROOT", "/geodata/Five-Billion-Pixels")).resolve()


class Gf2WindowDataset(Dataset):
    def __init__(self, manifest_path: Path, split: str, chip: int = CHIP_SIZE, train: bool = False):
        self.root = scene_root()
        self.chip = chip
        self.train = train
        self.rows: list[dict] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["split"] == split:
                    self.rows.append(row)
        self.rows = self.rows[: _int_env("PRITHVI_WORKER_EVAL_LIMIT", 0) or len(self.rows)]

    def __len__(self) -> int:
        return len(self.rows)

    def _window(self, width: int, height: int, index: int) -> Window:
        size = min(self.chip, width, height)
        if self.train:
            cols = max(1, width // size)
            rows = max(1, height // size)
            slot = index % (cols * rows)
            return Window((slot % cols) * size, (slot // cols) * size, size, size)
        return Window((width - size) // 2, (height - size) // 2, size, size)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with rasterio.open(self.root / row["image_paths"][0]) as source:
            window = self._window(source.width, source.height, index)
            image = source.read(window=window).astype(np.float32)
        with rasterio.open(self.root / row["label_path"]) as source:
            label = source.read(1, window=window).astype(np.int64)
        label = np.where(label <= 0, IGNORE_INDEX, label)
        return {"image": image, "label": label}


def train_normalization(manifest_path: Path) -> tuple[np.ndarray, np.ndarray]:
    dataset = Gf2WindowDataset(manifest_path, "train", train=True)
    values = []
    for row in dataset.rows[: _int_env("PRITHVI_WORKER_NORM_SCENES", 20)]:
        with rasterio.open(dataset.root / row["image_paths"][0]) as source:
            window = dataset._window(source.width, source.height, 0)
            values.append(source.read(window=window).astype(np.float32))
    array = np.concatenate([item.reshape(len(TASK_BANDS), -1) for item in values], axis=1)
    return array.mean(axis=1).astype(np.float32), np.maximum(array.std(axis=1), 1e-3).astype(
        np.float32
    )


def build_encoder(checkpoint: Path, freeze: bool):
    from terratorch.registry import BACKBONE_REGISTRY

    encoder = BACKBONE_REGISTRY.build(
        "prithvi_eo_v2_300_tl",
        pretrained=True,
        ckpt_path=str(checkpoint),
        bands=ENCODER_BANDS,
    )
    for parameter in encoder.parameters():
        parameter.requires_grad = not freeze
    if freeze:
        encoder.eval()
    return encoder


class PrithviT8(nn.Module):
    """4->6 band adapter + Prithvi encoder + shared lightweight head."""

    def __init__(self, encoder, freeze_backbone: bool):
        super().__init__()
        self.encoder = encoder
        self.adapter = nn.Sequential(
            nn.Conv2d(len(TASK_BANDS), 16, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(16, len(ENCODER_BANDS), kernel_size=1),
        )
        # A trainable adapter sits upstream of the backbone, so the graph must
        # stay alive through the frozen backbone or it gets no gradient.
        self.adapter_trainable = any(
            parameter.requires_grad for parameter in self.adapter.parameters()
        )
        self.head = LiteSpatialHead(EMBED_DIM, NUM_CLASSES)
        self.freeze_backbone = freeze_backbone

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        adapted = self.adapter(images)
        context = (
            torch.no_grad()
            if self.freeze_backbone and not self.adapter_trainable
            else torch.enable_grad()
        )
        with context:
            layers = self.encoder(adapted)
        tokens = layers[-1][:, 1:, :] if isinstance(layers, (list, tuple)) else layers[:, 1:, :]
        features = tokens.transpose(1, 2).reshape(tokens.shape[0], EMBED_DIM, GRID, GRID)
        logits = self.head(features)
        return torch.nn.functional.interpolate(
            logits, size=(CHIP_SIZE, CHIP_SIZE), mode="bilinear", align_corners=False
        )


def _evaluate(model, dataset, means, stds, device) -> dict[str, float]:
    matrix = ConfusionMatrix(num_classes=NUM_CLASSES)
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2):
            images = (batch["image"].to(device) - means) / stds
            logits = model(images)
            matrix.update(
                batch["label"].numpy(),
                logits.argmax(dim=1).cpu().numpy(),
                ignore_index=IGNORE_INDEX,
            )
    return multiclass_metrics(matrix)


def train_t8(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    checkpoint: Path,
    freeze_backbone: bool = True,
) -> int:
    run_id = request.run_id
    device = os.environ.get("PRITHVI_WORKER_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    epochs = _int_env("PRITHVI_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("PRITHVI_WORKER_BATCH_SIZE", 4)
    lr = _float_env("PRITHVI_WORKER_LR", 1e-3)
    workers = _int_env("PRITHVI_WORKER_NUM_WORKERS", 4)
    limit = _int_env("PRITHVI_WORKER_LIMIT_BATCHES", 0) or None

    manifest = Path(request.manifest_path)
    mean, std = train_normalization(manifest)
    means = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    stds = torch.tensor(std, device=device).view(1, -1, 1, 1)
    encoder = build_encoder(checkpoint, freeze=freeze_backbone)
    model = PrithviT8(encoder, freeze_backbone).to(device)
    adapter_params = count_parameters(model.adapter)
    head_params = count_parameters(model.head)
    train_ds = Gf2WindowDataset(manifest, "train", train=True)
    val_ds = Gf2WindowDataset(manifest, "validation")
    eval_ds = Gf2WindowDataset(manifest, "test")
    emit(
        run_id,
        "train.prepare",
        f"Prithvi T8 (freeze={freeze_backbone}): {len(train_ds)} train / "
        f"{len(eval_ds)} test scenes, "
        f"adapter {adapter_params} + head {head_params}, {NUM_CLASSES} classes",
    )
    writer.update_training(
        head_id="lite_spatial_head+band_adapter",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="gf2_train_stats_four_to_six_band_adapter")
    writer.update(
        tags={
            "adapter_parameters": str(adapter_params),
            "head_parameters": str(head_params),
            "backbone_frozen": str(freeze_backbone),
            "bands": ",".join(TASK_BANDS),
            "encoder_bands": ",".join(ENCODER_BANDS),
            "normalization": f"train-scene mean {mean.tolist()} std {std.tolist()}",
            "label_mapping": "0 -> ignore, 1..24 kept",
            "cross_domain_note": "0.8 m GF-2 without SWIR into a 30 m HLS encoder",
            "band_order_source": "verified via label-conditional band statistics",
            "learning_rate": f"{lr:g}",
            "device": device,
        }
    )

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    select = not freeze_backbone
    best = BestCheckpoint()
    val_every = validation_every()

    started = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in loader:
            images = (batch["image"].to(device) - means) / stds
            labels = batch["label"].to(device)
            if int((labels != IGNORE_INDEX).sum()) == 0:
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
        mean_loss = total_loss / max(steps, 1)
        writer.add_curve("loss", "train", epoch, mean_loss)
        emit(
            run_id,
            "train.epoch",
            f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}",
            progress=(epoch + 1) / epochs,
        )
        if select and ((epoch + 1) % val_every == 0 or epoch + 1 == epochs):
            validation = _evaluate(model, val_ds, means, stds, device)
            writer.add_curve("mean_iou", "validation", epoch, validation["mean_iou"])
            best.offer(validation["mean_iou"], epoch + 1, model)
    wall = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

    if select:
        best.restore(model)
        writer.update(
            tags={
                "selection": "best_validation_mean_iou",
                "selected_epoch": str(best.epoch),
                "validation_mean_iou": f"{best.score:.4f}",
            }
        )
    emit(run_id, "train.eval", f"evaluating on {len(eval_ds)} test scenes")
    metrics = _evaluate(model, eval_ds, means, stds, device)
    writer.add_metrics(
        to_metric_records(metrics, primary_name="mean_iou", direction="maximize", split="test")
    )
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=wall, peak_gpu_memory_mb=peak_mb
    )
    summary = (
        f"mean_iou={metrics['mean_iou']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"({epochs} epochs, adapter {adapter_params} + head {head_params}, "
        f"{'frozen' if freeze_backbone else 'fine-tuned'}, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]" if limit else " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
