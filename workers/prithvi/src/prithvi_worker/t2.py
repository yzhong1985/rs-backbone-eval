"""T2 (HLS Burn Scar) training for the Prithvi-EO-2.0 worker.

The encoder consumes the six HLS bands directly (no adapter needed) at native
512 resolution with 256 training crops; the shared ``LiteSpatialHead`` decodes
the 32x32 patch-token grid. The backbone is frozen for B2 (features cached,
since the grid is small) and fine-tuned for B1, driven by the protocol.
Evaluation runs on the ``official_validation`` split.
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

from rs_backbone_eval.bands import HLS_WAVELENGTHS_UM  # noqa: F401  (documentation)
from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
from rs_backbone_eval.evaluation.segmentation import binary_metrics
from rs_backbone_eval.heads import LiteSpatialHead, count_parameters
from rs_backbone_eval.worker_runtime import RunWriter

BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]
ENCODER_INPUT = 512
LABEL_SIZE = 512
OFFICIAL_EPOCHS = 50
TRAIN_CROP = 256
EVAL_SPLIT = "official_validation"


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


class HlsDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        split: str,
        *,
        crop: int | None = None,
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
        with rasterio.open(self.data_root / row["image_paths"][0]) as source:
            image = source.read().astype(np.float32)
        with rasterio.open(self.data_root / row["label_path"]) as source:
            label = source.read(1).astype(np.int64)
        if self.crop is not None and self.crop < image.shape[-1]:
            offset = (image.shape[-1] - self.crop) // 2
            image = image[:, offset : offset + self.crop, offset : offset + self.crop]
            label = label[offset : offset + self.crop, offset : offset + self.crop]
        return {"image": image, "label": label}


def train_normalization(manifest_path: Path, data_root: Path, cache_dir: Path):
    cache_file = cache_dir / "NORM_hls6_prithvi.json"
    if cache_file.is_file():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return np.asarray(payload["mean"], dtype=np.float32), np.asarray(
            payload["std"], dtype=np.float32
        )
    means, stds = [], []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] != "train":
                continue
            with rasterio.open(data_root / row["image_paths"][0]) as source:
                chip = source.read().astype(np.float32)
            means.append(chip.mean(axis=(1, 2)))
            stds.append(chip.std(axis=(1, 2)))
    mean = np.stack(means).mean(axis=0).astype(float)
    std = np.stack(stds).mean(axis=0).astype(float)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"mean": list(mean), "std": list(std), "chips": len(means)}), encoding="utf-8"
    )
    return mean.astype(np.float32), std.astype(np.float32)


def build_encoder(checkpoint: Path):
    from terratorch.registry import BACKBONE_REGISTRY

    encoder = BACKBONE_REGISTRY.build(
        "prithvi_eo_v2_300_tl", pretrained=True, ckpt_path=str(checkpoint), bands=BANDS
    )
    encoder.eval()
    return encoder


class PrithviT2(nn.Module):
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


def train_t2(
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = _int_env("PRITHVI_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("PRITHVI_WORKER_BATCH_SIZE", 1)
    lr = _float_env("PRITHVI_WORKER_LR", 1e-4)
    workers = _int_env("PRITHVI_WORKER_NUM_WORKERS", 4)
    crop = _int_env("PRITHVI_WORKER_TRAIN_CROP", TRAIN_CROP)
    limit = _int_env("PRITHVI_WORKER_LIMIT_BATCHES", 0) or None

    manifest = Path(request.manifest_path)
    mean, std = train_normalization(manifest, data_root, Path(request.cache_root) / "prithvi_norm")
    means = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    stds = torch.tensor(std, device=device).view(1, -1, 1, 1)

    train_ds = HlsDataset(manifest, data_root, "train", crop=crop)
    eval_ds = HlsDataset(manifest, data_root, EVAL_SPLIT)
    encoder = build_encoder(checkpoint).to(device)
    model = PrithviT2(encoder, freeze_backbone=freeze_backbone).to(device)
    writer.update_training(
        head_id="lite_spatial_head",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="native512_crop256_train_stats_hls6")
    writer.update(
        tags={
            "head_parameters": str(count_parameters(model.head)),
            "backbone_frozen": str(freeze_backbone),
            "encoder_input_px": str(ENCODER_INPUT),
            "feature_grid": "32x32",
            "train_crop_px": str(crop),
            "eval_split": EVAL_SPLIT,
            "learning_rate": f"{lr:g}",
        }
    )
    emit(
        run_id,
        "train.prepare",
        f"Prithvi T2 (freeze={freeze_backbone}): {len(train_ds)} train / {len(eval_ds)} "
        f"{EVAL_SPLIT} chips, crop {crop}",
    )

    if freeze_backbone:
        emit(run_id, "train.cache", "extracting frozen features")
        train_features, train_labels = _extract(model, train_ds, means, stds, device)
        eval_features, eval_labels = _extract(model, eval_ds, means, stds, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.05
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    if freeze_backbone:
        grid = train_features.shape[-1]
        labels_grid = (
            torch.nn.functional.interpolate(
                train_labels.unsqueeze(1).float(), size=(grid, grid), mode="nearest"
            )
            .squeeze(1)
            .long()
            .to(device)
        )
        train_features = train_features.to(device)
        for epoch in range(epochs):
            permutation = torch.randperm(train_features.shape[0], device=device)
            total_loss, steps = 0.0, 0
            for start in range(0, permutation.numel(), 8):
                index = permutation[start : start + 8]
                logits = model.head(train_features[index])
                loss = criterion(logits, labels_grid[index])
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
    wall = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

    emit(run_id, "train.eval", f"evaluating on {len(eval_ds)} {EVAL_SPLIT} chips at full 512")
    matrix = ConfusionMatrix(num_classes=2)
    model.eval()
    with torch.inference_mode():
        if freeze_backbone:
            for start in range(0, eval_features.shape[0], 8):
                logits = model(eval_features[start : start + 8].to(device), size=LABEL_SIZE)
                matrix.update(
                    eval_labels[start : start + 8].numpy(),
                    logits.argmax(dim=1).cpu().numpy(),
                    ignore_index=-1,
                )
        else:
            for batch in DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2):
                images = (batch["image"].to(device) - means) / stds
                logits = model(model.features(images), size=LABEL_SIZE)
                matrix.update(
                    batch["label"].numpy(), logits.argmax(dim=1).cpu().numpy(), ignore_index=-1
                )
    metrics = binary_metrics(matrix, foreground=1)
    metrics[metric_name] = metrics.pop("foreground_iou")
    writer.add_metrics(
        to_metric_records(metrics, primary_name=metric_name, direction="maximize", split=EVAL_SPLIT)
    )
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=wall, peak_gpu_memory_mb=peak_mb
    )
    summary = (
        f"{metric_name}={metrics[metric_name]:.4f} mean_iou={metrics['mean_iou']:.4f} "
        f"({epochs} epochs, head {count_parameters(model.head)} params, "
        f"{'frozen' if freeze_backbone else 'fine-tuned'}, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0


def _extract(model: PrithviT2, dataset: HlsDataset, means, stds, device):
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    features, labels = [], []
    for batch in loader:
        images = (batch["image"].to(device) - means) / stds
        features.append(model.features(images).cpu())
        labels.append(batch["label"])
    return torch.cat(features), torch.cat(labels)
