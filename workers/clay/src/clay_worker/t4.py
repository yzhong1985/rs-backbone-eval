"""T4 (Landslide4Sense) training for the Clay v1.5 worker.

Clay is sensor-agnostic, so the six official bands are fed with their central
wavelengths (derived metadata, auditable) and the 128 -> 224 resize from the
official config. The shipped per-band scale does not match TerraTorch's
published statistics, so train-split mean/std are used (identical across
backbones). Both tracks select the head/checkpoint on ``validation`` and report
``test`` once; labels are binary with no ignore index.
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

from rs_backbone_eval.bands import HLS_WAVELENGTHS_UM
from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
from rs_backbone_eval.evaluation.segmentation import binary_metrics
from rs_backbone_eval.heads import count_parameters
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.temporal_io import landslide_statistics, read_landslide_sample
from rs_backbone_eval.worker_runtime import RunWriter

from .clay_probe import add_upstream_to_path
from .t1 import CLAY_LARGE, ClayProbe

BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD", "SWIR_1", "SWIR_2"]
HLS_CLAY_BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]
SENSOR = "landslide"
GSD_M = 10.0
ENCODER_INPUT = 224
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


def _evaluate(model, dataset, means, stds, waves, device) -> dict[str, float]:
    matrix = ConfusionMatrix(num_classes=2)
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2):
            pixels = (batch["image"].to(device) - means) / stds
            logits = model(pixels, waves)
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
    task_bands: list[str],
    freeze_backbone: bool,
    metric_name: str,
) -> int:
    run_id = request.run_id
    data_root = Path(request.data_root)
    device = os.environ.get("CLAY_WORKER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    epochs = _int_env("CLAY_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("CLAY_WORKER_BATCH_SIZE", 4)
    lr = _float_env("CLAY_WORKER_LR", 1e-3)
    workers = _int_env("CLAY_WORKER_NUM_WORKERS", 4)
    limit = _int_env("CLAY_WORKER_LIMIT_BATCHES", 0) or None

    mean, std = landslide_statistics(BANDS)
    means = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    stds = torch.tensor(std, device=device).view(1, -1, 1, 1)
    waves = torch.tensor(
        [HLS_WAVELENGTHS_UM[band] for band in BANDS], dtype=torch.float32, device=device
    )

    add_upstream_to_path()
    from claymodel.finetune.segment.factory import SegmentEncoder  # upstream checkout

    encoder = SegmentEncoder(**CLAY_LARGE, ckpt_path=str(checkpoint)).to(device)
    model = ClayProbe(
        encoder,
        num_classes=2,
        gsd=GSD_M,
        sensor=SENSOR,
        freeze_backbone=freeze_backbone,
    ).to(device)
    head_params = count_parameters(model.head)

    manifest = Path(request.manifest_path)
    train_ds = LandslideDataset(manifest, data_root, "train")
    select_ds = LandslideDataset(manifest, data_root, SELECT_SPLIT)
    eval_ds = LandslideDataset(manifest, data_root, EVAL_SPLIT)
    eval_limit = _int_env("CLAY_WORKER_EVAL_LIMIT", 0)
    if eval_limit:
        select_ds.rows = select_ds.rows[:eval_limit]
        eval_ds.rows = eval_ds.rows[:eval_limit]
    emit(
        run_id,
        "train.prepare",
        f"Clay T4 (freeze={freeze_backbone}): {len(train_ds)} train / {len(select_ds)} "
        f"{SELECT_SPLIT} / {len(eval_ds)} {EVAL_SPLIT}, head {head_params} params",
    )
    writer.update_training(
        head_id="lite_spatial_head",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="landslide_train_split_stats_resize224")
    writer.update(
        tags={
            "head_parameters": str(head_params),
            "backbone_frozen": str(freeze_backbone),
            "encoder_input_px": str(ENCODER_INPUT),
            "derived_metadata_bands": ",".join(HLS_CLAY_BANDS),
            "normalization": "train-split statistics (shipped scale differs from TerraTorch)",
            "select_split": SELECT_SPLIT,
            "eval_split": EVAL_SPLIT,
            "learning_rate": f"{lr:g}",
            "device": device,
        }
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    best = BestCheckpoint()
    val_every = validation_every()

    started = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in loader:
            pixels = (batch["image"].to(device) - means) / stds
            labels = batch["label"].to(device)
            logits = model(pixels, waves)
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
            validation = _evaluate(model, select_ds, means, stds, waves, device)
            writer.add_curve("mean_iou", SELECT_SPLIT, epoch, validation["mean_iou"])
            best.offer(validation["mean_iou"], epoch + 1, model)
    wall = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

    best.restore(model)
    writer.update(
        tags={
            "selection": "best_validation_mean_iou",
            "selected_epoch": str(best.epoch),
            "validation_mean_iou": f"{best.score:.4f}",
        }
    )
    metrics = _evaluate(model, eval_ds, means, stds, waves, device)
    writer.add_metrics(
        to_metric_records(metrics, primary_name=metric_name, direction="maximize", split=EVAL_SPLIT)
    )
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=wall, peak_gpu_memory_mb=peak_mb
    )
    summary = (
        f"{metric_name}={metrics[metric_name]:.4f} "
        f"foreground_iou={metrics['foreground_iou']:.4f} "
        f"({epochs} epochs, head {head_params} params, "
        f"{'frozen' if freeze_backbone else 'fine-tuned'}, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]" if limit else " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
