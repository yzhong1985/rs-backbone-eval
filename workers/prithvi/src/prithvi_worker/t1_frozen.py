"""T1 frozen representation probe for Prithvi-EO-2.0 (B2 track).

Same recipe as the AnySat and Clay frozen tracks so the three backbones can
be compared in one table: frozen encoder, the shared ``LiteSpatialHead``,
identical train/test splits, 50 epochs, AdamW lr 1e-3, metrics from the
project-wide confusion matrix.

Input follows the repo-config convention (resize 512 -> 224) and the restored
band normalization; encoder features are the final ViT layer's patch tokens
(196 tokens -> 14x14 grid at 224). Because the backbone is frozen, features
are extracted once and the head trains on the cache (design 6.4); the 14x14
logits are upsampled to the full 512 label grid so metrics use exactly the
same labels as AnySat and Clay.
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
from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
from rs_backbone_eval.evaluation.segmentation import binary_metrics
from rs_backbone_eval.heads import LiteSpatialHead, count_parameters
from rs_backbone_eval.perturb import (
    apply_band_drop,
    apply_resolution,
    band_indices_to_drop,
    describe,
    perturbations_from_env,
    resolve,
)
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.worker_runtime import RunWriter

BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
CHIP_BAND_INDICES = {"BLUE": 1, "GREEN": 2, "RED": 3, "NIR_NARROW": 8, "SWIR_1": 11, "SWIR_2": 12}
ENCODER_PATCH = 16
# Encoding resolution: the repo-config convention resizes to 224 (14x14 grid),
# but the encoder also runs natively at 512 (32x32 grid) where full-resolution
# comparison with AnySat/Clay is fairer. Overridable for the fairness check.
ENCODER_INPUT = int(os.environ.get("PRITHVI_FROZEN_INPUT", "224"))
GRID = ENCODER_INPUT // ENCODER_PATCH
LABEL_SIZE = 512
EMBED_DIM = 1024
OFFICIAL_EPOCHS = 50
BATCH_SIZE = 8


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


def chip_indices_for_bands(bands: list[str]) -> list[int]:
    """Positions of task bands inside the 13-band Sen1Floods11 L1C chip."""
    return [CHIP_BAND_INDICES[band] for band in bands]


def band_stats() -> tuple[np.ndarray, np.ndarray]:
    from terratorch.datamodules.sen1floods11 import MEANS, STDS

    mean = np.asarray([MEANS[band] for band in BANDS], dtype=np.float32)
    std = np.asarray([STDS[band] for band in BANDS], dtype=np.float32)
    return mean, std


class T1ChipDataset(Dataset):
    def __init__(self, manifest_path: Path, data_root: Path, split: str):
        self.data_root = data_root
        self.indices = [CHIP_BAND_INDICES[band] for band in BANDS]
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
        # constant_scale 0.0001 then the restored band statistics: the recipe
        # that the B0 supervised runs needed to avoid majority-class collapse.
        image = image * 0.0001
        mean, std = band_stats()
        image = (image - mean[:, None, None]) / std[:, None, None]
        tensor = torch.from_numpy(image).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(
            tensor, size=(ENCODER_INPUT, ENCODER_INPUT), mode="bilinear", align_corners=False
        )[0]
        return {"image": tensor, "label": label}


def build_encoder(checkpoint: Path):
    from terratorch.registry import BACKBONE_REGISTRY

    encoder = BACKBONE_REGISTRY.build(
        "prithvi_eo_v2_300_tl",
        pretrained=not random_init_requested(),
        ckpt_path=None if random_init_requested() else str(checkpoint),
        bands=BANDS,
    )
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()
    return encoder


class PrithviFrozenProbe(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = LiteSpatialHead(EMBED_DIM, 2)

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """[B,6,224,224] -> [B,1024,14,14] (final layer patch tokens)."""
        with torch.no_grad():
            layers = self.encoder(images)
        tokens = layers[-1][:, 1:, :] if isinstance(layers, (list, tuple)) else layers[:, 1:, :]
        batch = tokens.shape[0]
        return tokens.transpose(1, 2).reshape(batch, EMBED_DIM, GRID, GRID)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


def _extract_features(
    model, dataset, device, emit, run_id, perturbation: str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    spec = resolve(perturbation) if perturbation else {"kind": "none"}
    drop = band_indices_to_drop(BANDS, spec["bands"]) if spec["kind"] == "band_drop" else []
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    features, labels = [], []
    for batch in loader:
        images = batch["image"].to(device)
        if spec["kind"] == "band_drop":
            images = apply_band_drop(images, drop)
        elif spec["kind"] == "resolution":
            images = apply_resolution(images, int(spec["factor"]))
        features.append(model.extract(images).cpu())
        labels.append(batch["label"])
    emit(run_id, "train.cache", f"cached features for {len(dataset)} chips")
    return torch.cat(features), torch.cat(labels)


def train_frozen(
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
    lr = _float_env("PRITHVI_WORKER_LR", 1e-3)
    limit = _int_env("PRITHVI_WORKER_LIMIT_BATCHES", 0) or None

    manifest = Path(request.manifest_path)
    train_ds = T1ChipDataset(manifest, data_root, "train")
    eval_ds = T1ChipDataset(manifest, data_root, "test")
    encoder = build_encoder(checkpoint).to(device)
    model = PrithviFrozenProbe(encoder).to(device)
    head_params = count_parameters(model.head)
    total = count_parameters(model)
    writer.update_training(
        head_id="lite_spatial_head", trainable_parameters=head_params, total_parameters=total
    )
    writer.update_input(preprocessing_id="repo_config_resize224_restored_norm")
    writer.update(
        tags={
            "head_parameters": str(head_params),
            "backbone_frozen": "True",
            "encoder_input_px": str(ENCODER_INPUT),
            "feature_grid": f"{GRID}x{GRID}",
            "feature_stride_px": "16 (at 224 encoding, evaluated on the 512 grid)",
            "learning_rate": f"{lr:g}",
            "device": device,
        }
    )
    emit(
        run_id,
        "train.prepare",
        f"frozen Prithvi probe: {len(train_ds)} train / {len(eval_ds)} test chips, "
        f"encode {ENCODER_INPUT}px -> {GRID}x{GRID} features, head {head_params} params",
    )

    started = time.time()
    train_features, train_labels = _extract_features(model, train_ds, device, emit, run_id)
    eval_features, eval_labels = _extract_features(model, eval_ds, device, emit, run_id)
    emit(run_id, "train.model", f"feature cache ready ({total} total params, frozen 300M encoder)")

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    train_features = train_features.to(device)
    # Loss is computed on the 14x14 feature grid; metrics still upsample
    # predictions to the full 512 label grid (as for AnySat and Clay).
    train_labels_grid = (
        torch.nn.functional.interpolate(
            train_labels.unsqueeze(1).float(), size=(GRID, GRID), mode="nearest"
        )
        .squeeze(1)
        .long()
        .to(device)
    )
    for epoch in range(epochs):
        permutation = torch.randperm(train_features.shape[0], device=device)
        total_loss, steps = 0.0, 0
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size]
            logits = model(train_features[index])
            loss = criterion(logits, train_labels_grid[index])
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

    emit(run_id, "train.eval", f"evaluating on {len(eval_ds)} test chips")
    matrix = ConfusionMatrix(num_classes=2)
    model.eval()
    with torch.inference_mode():
        for start in range(0, eval_features.shape[0], batch_size):
            features = eval_features[start : start + batch_size].to(device)
            logits = model(features)
            upsampled = torch.nn.functional.interpolate(
                logits, size=(LABEL_SIZE, LABEL_SIZE), mode="bilinear", align_corners=False
            )
            matrix.update(
                eval_labels[start : start + batch_size].numpy(),
                upsampled.argmax(dim=1).cpu().numpy(),
                ignore_index=-1,
            )
    metrics = binary_metrics(matrix, foreground=1)
    metrics["water_iou"] = metrics.pop("foreground_iou")
    writer.add_metrics(
        to_metric_records(metrics, primary_name="water_iou", direction="maximize", split="test")
    )
    for name in perturbations_from_env():
        if resolve(name)["kind"] == "none":
            continue
        perturbed_features, _ = _extract_features(model, eval_ds, device, emit, run_id, name)
        perturbed_matrix = ConfusionMatrix(num_classes=2)
        with torch.inference_mode():
            for start in range(0, perturbed_features.shape[0], batch_size):
                logits = model(perturbed_features[start : start + batch_size].to(device))
                upsampled = torch.nn.functional.interpolate(
                    logits, size=(LABEL_SIZE, LABEL_SIZE), mode="bilinear", align_corners=False
                )
                perturbed_matrix.update(
                    eval_labels[start : start + batch_size].numpy(),
                    upsampled.argmax(dim=1).cpu().numpy(),
                    ignore_index=-1,
                )
        perturbed = binary_metrics(perturbed_matrix, foreground=1)
        perturbed["water_iou"] = perturbed.pop("foreground_iou")
        writer.add_metrics(
            to_metric_records(
                perturbed,
                primary_name="water_iou",
                direction="maximize",
                split=describe(name),
                primary=False,
            )
        )
        emit(
            run_id,
            "train.perturb",
            f"{describe(name)}: water_iou={perturbed['water_iou']:.4f} "
            f"(baseline {metrics['water_iou']:.4f})",
        )
    if any(resolve(name)["kind"] != "none" for name in perturbations_from_env()):
        writer.update(
            tags={
                "perturbations": ",".join(perturbations_from_env()),
                "perturbation_note": "inference-time input robustness, not re-adaptation",
            }
        )
    writer.update_training(
        epochs_completed=epochs,
        wall_time_seconds=wall,
        peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / (1024**2)
        if device == "cuda"
        else 0.0,
    )
    summary = (
        f"water_iou={metrics['water_iou']:.4f} mean_iou={metrics['mean_iou']:.4f} "
        f"({epochs} epochs, head {head_params} params, frozen encoder, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
