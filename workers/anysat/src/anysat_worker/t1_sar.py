"""T1S: the Sentinel-1 counterpart of T1 for the AnySat worker.

AnySat has a native Sentinel-1 entrance that expects three channels per date:
VV, VH and their ratio VV/(VH+eps), which is exactly how the upstream
Sen1Floods11 loader builds them. No spectral adapter is needed here -- the
six-band adapter path in ``t1_adapter`` exists only because the optical task
carries six bands while AnySat's ``s2`` entrance expects ten.

The chips, splits and hand-drawn labels are shared with T1, so the two runs are
directly comparable (optical versus radar input, same targets).
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
from rs_backbone_eval.evaluation.segmentation import binary_metrics
from rs_backbone_eval.heads import LiteSpatialHead, count_parameters
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.worker_runtime import RunWriter

from .s2_probe import read_chip
from .t1_adapter import build_backbone

BANDS = ["VV", "VH"]
NUM_CLASSES = 2
IGNORE_INDEX = -1
PATCH_SIZE_M = 160
FULL_CHIP = 512
OFFICIAL_EPOCHS = 50
RATIO_CLAMP = 1e4


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


def sar_channels(vv_vh: np.ndarray) -> np.ndarray:
    """[2,H,W] VV/VH in dB -> [3,H,W] (VV, VH, ratio) as the upstream loader does."""
    ratio = vv_vh[:1] / (vv_vh[1:] + 1e-10)
    return np.concatenate([vv_vh[:2], np.clip(ratio, -RATIO_CLAMP, RATIO_CLAMP)], axis=0)


class SarChipDataset(Dataset):
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
        self.rows = self.rows[: _int_env("ANYSAT_WORKER_EVAL_LIMIT", 0) or len(self.rows)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        chip = sar_channels(read_chip(self.data_root / row["image_paths"][0]))
        label = read_chip(self.data_root / row["label_path"], band_indices=[0])[0].astype(np.int64)
        if self.crop is not None and self.crop < chip.shape[-1]:
            offset = (chip.shape[-1] - self.crop) // 2
            chip = chip[:, offset : offset + self.crop, offset : offset + self.crop]
            label = label[offset : offset + self.crop, offset : offset + self.crop]
        return {"image": chip, "label": label, "date": int(str(row["timestamps"][0])[:4])}


def train_normalization(manifest_path: Path, data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel train-split mean/std of the three SAR channels."""
    means, stds = [], []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] != "train":
                continue
            chip = sar_channels(read_chip(data_root / row["image_paths"][0]))
            means.append(chip.mean(axis=(1, 2)))
            stds.append(chip.std(axis=(1, 2)))
    mean = np.stack(means).mean(axis=0)
    std = np.stack(stds).mean(axis=0)
    return mean.astype(np.float32), np.maximum(std, 1e-3).astype(np.float32)


class SarProbe(nn.Module):
    """AnySat's native Sentinel-1 entrance (frozen or fine-tuned) + spatial head."""

    def __init__(self, backbone, freeze_backbone: bool):
        super().__init__()
        self.backbone = backbone
        self.head = LiteSpatialHead(2 * 768, NUM_CLASSES)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor, dates: torch.Tensor) -> torch.Tensor:
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            dense = self.backbone(
                {"s1": images, "s1_dates": dates},
                patch_size=PATCH_SIZE_M,
                output="dense",
                output_modality="s1",
            )
        return self.head(dense.permute(0, 3, 1, 2))


def _evaluate(model, dataset, means, stds, device) -> dict[str, float]:
    matrix = ConfusionMatrix(num_classes=NUM_CLASSES)
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2):
            images = (batch["image"].to(device) - means) / stds
            dates = batch["date"].to(device).float().unsqueeze(1)
            logits = model(images, dates)
            matrix.update(
                batch["label"].numpy(), logits.argmax(dim=1).cpu().numpy(), ignore_index=-1
            )
            del images, dates, logits
            torch.cuda.empty_cache()
    result = binary_metrics(matrix, foreground=1)
    result["water_iou"] = result.pop("foreground_iou")
    return result


def train_sar(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    checkpoint: Path,
    freeze_backbone: bool = True,
) -> int:
    run_id = request.run_id
    data_root = Path(request.data_root)
    device = os.environ.get("ANYSAT_WORKER_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    epochs = _int_env("ANYSAT_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("ANYSAT_WORKER_BATCH_SIZE", 1)
    lr = _float_env("ANYSAT_WORKER_LR", 1e-3)
    workers = _int_env("ANYSAT_WORKER_NUM_WORKERS", 4)
    limit = _int_env("ANYSAT_WORKER_LIMIT_BATCHES", 0) or None
    train_crop = _int_env("ANYSAT_WORKER_TRAIN_CROP", 224)
    freeze = freeze_backbone
    if os.environ.get("ANYSAT_WORKER_FINETUNE"):
        freeze = os.environ["ANYSAT_WORKER_FINETUNE"] in {"0", "false"}

    manifest = Path(request.manifest_path)
    mean, std = train_normalization(manifest, data_root)
    means = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    stds = torch.tensor(std, device=device).view(1, 3, 1, 1)

    backbone = build_backbone(checkpoint).to(device)
    model = SarProbe(backbone, freeze_backbone=freeze).to(device)
    head_params = count_parameters(model.head)
    train_ds = SarChipDataset(manifest, data_root, "train", crop=train_crop)
    val_ds = SarChipDataset(manifest, data_root, "validation")
    eval_ds = SarChipDataset(manifest, data_root, "test")
    emit(
        run_id,
        "train.prepare",
        f"AnySat T1S (freeze={freeze}): {len(train_ds)} train / {len(eval_ds)} test, "
        f"crop {train_crop}, s1 entrance (VV, VH, ratio), head {head_params}",
    )
    writer.update_training(
        head_id="lite_spatial_head+anySat_s1_entrance",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="s1_train_stats_vv_vh_ratio")
    writer.update(
        tags={
            "head_parameters": str(head_params),
            "backbone_frozen": str(freeze),
            "s1_channels": "VV, VH, VV/(VH+eps)",
            "normalization": f"train-split mean {mean.tolist()} std {std.tolist()}",
            "learning_rate": f"{lr:g}",
            "device": device,
        }
    )

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    select = not freeze
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
            dates = batch["date"].to(device).float().unsqueeze(1)
            if int((labels != IGNORE_INDEX).sum()) == 0:
                continue
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(images, dates)
            loss = criterion(logits.float(), labels)
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
            torch.cuda.empty_cache()
            validation = _evaluate(model, val_ds, means, stds, device)
            writer.add_curve("water_iou", "validation", epoch, validation["water_iou"])
            best.offer(validation["water_iou"], epoch + 1, model)
            torch.cuda.empty_cache()
    wall = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

    if select:
        best.restore(model)
        writer.update(
            tags={
                "selection": "best_validation_water_iou",
                "selected_epoch": str(best.epoch),
                "validation_water_iou": f"{best.score:.4f}",
            }
        )
    torch.cuda.empty_cache()
    metrics = _evaluate(model, eval_ds, means, stds, device)
    writer.add_metrics(
        to_metric_records(metrics, primary_name="water_iou", direction="maximize", split="test")
    )
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=wall, peak_gpu_memory_mb=peak_mb
    )
    summary = (
        f"water_iou={metrics['water_iou']:.4f} mean_iou={metrics['mean_iou']:.4f} "
        f"({epochs} epochs, head {head_params} params, "
        f"{'frozen' if freeze else 'fine-tuned'}, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]" if limit else " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
