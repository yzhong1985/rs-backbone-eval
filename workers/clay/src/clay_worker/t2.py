"""T2 (HLS Burn Scar) training for the Clay v1.5 worker.

Clay is sensor-agnostic, so HLS needs no adapter: we build an auditable
derived metadata profile (six HLS bands with their central wavelengths, GSD
and train-split mean/std) and feed it with the same SegmentEncoder +
shared-head recipe used everywhere else.
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

from rs_backbone_eval.bands import HLS_WAVELENGTHS_UM
from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
from rs_backbone_eval.evaluation.segmentation import binary_metrics
from rs_backbone_eval.heads import count_parameters
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.worker_runtime import RunWriter

from .clay_probe import add_upstream_to_path, prepare_encoder
from .t1 import CLAY_LARGE, ClayProbe

# Clay band names for the six HLS bands, in task order.
HLS_CLAY_BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]
SENSOR = "hls-burn-scars"
TRAIN_CROP = 256
FULL_CHIP = 512
OFFICIAL_EPOCHS = 50
OFFICIAL_BATCH_SIZE = 4
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


def derive_hls_metadata(
    manifest_path: Path, data_root: Path, task_bands: list[str], cache_dir: Path
):
    """Derived HLS metadata: wavelengths, GSD and train-split statistics."""
    cache_file = cache_dir / "hls_metadata.json"
    if cache_file.is_file():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        return payload
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
    payload = {
        "bands": HLS_CLAY_BANDS,
        "task_bands": list(task_bands),
        "gsd": 30.0,
        "mean": list(mean),
        "std": list(std),
        "wavelength": [HLS_WAVELENGTHS_UM[band] for band in task_bands],
        "chips": len(means),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def train_t2(
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
    batch_size = _int_env("CLAY_WORKER_BATCH_SIZE", OFFICIAL_BATCH_SIZE)
    lr = _float_env("CLAY_WORKER_LR", 1e-3)
    workers = _int_env("CLAY_WORKER_NUM_WORKERS", 4)
    crop = _int_env("CLAY_WORKER_TRAIN_CROP", TRAIN_CROP)
    limit = _int_env("CLAY_WORKER_LIMIT_BATCHES", 0) or None

    metadata = derive_hls_metadata(
        Path(request.manifest_path), data_root, task_bands, Path(request.cache_root) / "clay_meta"
    )
    means = torch.tensor(metadata["mean"], device=device).view(1, -1, 1, 1)
    stds = torch.tensor(metadata["std"], device=device).view(1, -1, 1, 1)
    waves = torch.tensor(metadata["wavelength"], dtype=torch.float32, device=device)

    add_upstream_to_path()
    from claymodel.finetune.segment.factory import SegmentEncoder  # upstream checkout

    encoder = SegmentEncoder(
        **CLAY_LARGE, ckpt_path=None if random_init_requested() else str(checkpoint)
    ).to(device)
    prepare_encoder(encoder, freeze_backbone)
    model = ClayProbe(
        encoder, num_classes=2, gsd=metadata["gsd"], sensor=SENSOR, freeze_backbone=freeze_backbone
    ).to(device)
    head_params = count_parameters(model.head)
    trainable = count_parameters(model, trainable_only=True)

    train_ds = HlsDataset(Path(request.manifest_path), data_root, "train", crop=crop)
    eval_ds = HlsDataset(Path(request.manifest_path), data_root, EVAL_SPLIT)
    emit(
        run_id,
        "train.prepare",
        f"Clay T2: {len(train_ds)} train / {len(eval_ds)} {EVAL_SPLIT} chips, crop {crop}, "
        f"derived HLS metadata bands {metadata['bands']}, freeze={freeze_backbone}",
    )
    writer.update_training(
        head_id="lite_spatial_head",
        trainable_parameters=trainable,
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="derived_hls_metadata")
    writer.update(
        tags={
            "head_parameters": str(head_params),
            "backbone_frozen": str(freeze_backbone),
            "patch_size_px": str(CLAY_LARGE["patch_size"]),
            "patch_stride_m": f"{CLAY_LARGE['patch_size'] * metadata['gsd']:g}",
            "train_crop_px": str(crop),
            "derived_metadata_bands": ",".join(metadata["bands"]),
            "derived_metadata_wavelength_um": ",".join(str(w) for w in metadata["wavelength"]),
            "eval_split": EVAL_SPLIT,
            "learning_rate": f"{lr:g}",
            "device": device,
        }
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)

    started = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in loader:
            pixels = (batch["image"].to(device) - means) / stds
            labels = batch["label"].to(device)
            if int((labels != -1).sum()) == 0:
                continue
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
    wall = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

    emit(run_id, "train.eval", f"evaluating on {len(eval_ds)} {EVAL_SPLIT} chips at full 512")
    matrix = ConfusionMatrix(num_classes=2)
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2):
            pixels = (batch["image"].to(device) - means) / stds
            logits = model(pixels, waves)
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
        f"({epochs} epochs, head {head_params} params, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
