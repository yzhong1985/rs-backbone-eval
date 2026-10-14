"""T8 (Five-Billion-Pixels / GF-2) land cover for the AnySat worker.

Four GF-2 bands (B/G/R/NIR) must reach AnySat's fixed ten-band ``s2`` entrance,
so the shared explicit spectral adapter synthesizes the six missing bands; zero
padding stays forbidden. GF-2 is 0.8 m with no SWIR while AnySat's S2 entrance
was pretrained on 10 m Sentinel-2, so the run is cross-domain by construction
and the adapter parameters are reported separately from the task head.

Scenes are ~6900x7300 px, so chips are read as rasterio windows.
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

from .spectral_adapter import SpectralAdapter
from .t1_adapter import build_backbone

BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD"]
NUM_CLASSES = 25
IGNORE_INDEX = -1
CHIP_SIZE = 224
PATCH_SIZE_M = 160
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
        self.rows = self.rows[: _int_env("ANYSAT_WORKER_EVAL_LIMIT", 0) or len(self.rows)]

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
        return {"image": image, "label": label, "date": 2016}


def train_normalization(manifest_path: Path) -> tuple[np.ndarray, np.ndarray]:
    dataset = Gf2WindowDataset(manifest_path, "train", train=True)
    values = []
    for row in dataset.rows[: _int_env("ANYSAT_WORKER_NORM_SCENES", 20)]:
        with rasterio.open(dataset.root / row["image_paths"][0]) as source:
            window = dataset._window(source.width, source.height, 0)
            values.append(source.read(window=window).astype(np.float32))
    array = np.concatenate([item.reshape(len(BANDS), -1) for item in values], axis=1)
    return array.mean(axis=1).astype(np.float32), np.maximum(array.std(axis=1), 1e-3).astype(
        np.float32
    )


class AdapterProbeT8(nn.Module):
    """Spectral adapter + AnySat's S2 entrance (frozen or fine-tuned) + spatial head."""

    def __init__(self, backbone, task_bands: list[str], freeze_backbone: bool):
        super().__init__()
        self.backbone = backbone
        self.adapter = SpectralAdapter(task_bands)
        # A trainable adapter sits upstream of the backbone, so the graph
        # must stay alive through the frozen backbone or it gets no gradient.
        self.adapter_trainable = any(
            parameter.requires_grad for parameter in self.adapter.parameters()
        )
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
        adapted = self.adapter(images)
        context = (
            torch.no_grad()
            if self.freeze_backbone and not self.adapter_trainable
            else torch.enable_grad()
        )
        with context:
            dense = self.backbone(
                {"s2": adapted, "s2_dates": dates},
                patch_size=PATCH_SIZE_M,
                output="dense",
                output_modality="s2",
            )
        return self.head(dense.permute(0, 3, 1, 2))


def _evaluate(model, dataset, means, stds, device) -> dict[str, float]:
    matrix = ConfusionMatrix(num_classes=NUM_CLASSES)
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2):
            images = (batch["image"].to(device) - means) / stds
            images = images.unsqueeze(1)
            dates = batch["date"].to(device).float().unsqueeze(1)
            logits = model(images, dates)
            matrix.update(
                batch["label"].numpy(),
                logits.argmax(dim=1).cpu().numpy(),
                ignore_index=IGNORE_INDEX,
            )
            del images, dates, logits
            torch.cuda.empty_cache()
    return multiclass_metrics(matrix)


def train_t8(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    checkpoint: Path,
    task_bands: list[str],
    freeze_backbone: bool = True,
) -> int:
    run_id = request.run_id
    device = os.environ.get("ANYSAT_WORKER_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    epochs = _int_env("ANYSAT_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("ANYSAT_WORKER_BATCH_SIZE", 1)
    lr = _float_env("ANYSAT_WORKER_LR", 1e-3)
    workers = _int_env("ANYSAT_WORKER_NUM_WORKERS", 4)
    limit = _int_env("ANYSAT_WORKER_LIMIT_BATCHES", 0) or None
    freeze = freeze_backbone
    if os.environ.get("ANYSAT_WORKER_FINETUNE"):
        freeze = os.environ["ANYSAT_WORKER_FINETUNE"] in {"0", "false"}

    manifest = Path(request.manifest_path)
    mean, std = train_normalization(manifest)
    means = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    stds = torch.tensor(std, device=device).view(1, -1, 1, 1)
    backbone = build_backbone(checkpoint).to(device)
    model = AdapterProbeT8(backbone, task_bands, freeze_backbone=freeze).to(device)
    adapter_params = count_parameters(model.adapter)
    head_params = count_parameters(model.head)
    train_ds = Gf2WindowDataset(manifest, "train", train=True)
    val_ds = Gf2WindowDataset(manifest, "validation")
    eval_ds = Gf2WindowDataset(manifest, "test")
    emit(
        run_id,
        "train.prepare",
        f"AnySat T8 (freeze={freeze}): {len(train_ds)} train / {len(eval_ds)} test scenes, "
        f"adapter {adapter_params} + head {head_params}, {NUM_CLASSES} classes",
    )
    writer.update_training(
        head_id="lite_spatial_head+spectral_adapter",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="gf2_train_stats_four_band")
    writer.update(
        tags={
            "adapter_parameters": str(adapter_params),
            "head_parameters": str(head_params),
            "backbone_frozen": str(freeze),
            "bands": ",".join(task_bands),
            "normalization": f"train-scene mean {mean.tolist()} std {std.tolist()}",
            "label_mapping": "0 -> ignore, 1..24 kept",
            "cross_domain_note": "0.8 m GF-2 with no SWIR into a 10 m S2 entrance",
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
            images = images.unsqueeze(1)
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
            writer.add_curve("mean_iou", "validation", epoch, validation["mean_iou"])
            best.offer(validation["mean_iou"], epoch + 1, model)
            torch.cuda.empty_cache()
    wall = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

    if select:
        best.restore(model)
        best.save(writer.run_dir / "best_checkpoint.pt")
        writer.add_artifact("checkpoint", "best_checkpoint.pt", "validation-selected weights")
        writer.update(
            tags={
                "selection": "best_validation_mean_iou",
                "selected_epoch": str(best.epoch),
                "validation_mean_iou": f"{best.score:.4f}",
            }
        )
    torch.cuda.empty_cache()
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
        f"{'frozen' if freeze else 'fine-tuned'}, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]" if limit else " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
