"""T2 (HLS Burn Scar) training for the AnySat worker.

Same recipe as the T1 tracks: an explicit spectral adapter maps the six HLS
bands onto AnySat's fixed ten-band S2 entrance (HLS.S30 is Sentinel-2 derived,
so the S2 projector is the closest pretrained entrance; the design forbids
zero-padding), then the shared LiteSpatialHead decodes the dense features.
The backbone is frozen for the B2 probe and fine-tuned for B1, driven by the
protocol. Evaluation runs at full 512 resolution on the ``official_validation``
split, which the task configuration locks for final reporting.
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
from rs_backbone_eval.worker_runtime import RunWriter

from .s2_probe import add_upstream_to_path
from .spectral_adapter import SpectralAdapter

PATCH_SIZE_M = 160
TRAIN_CROP = 256
FULL_CHIP = 512
OFFICIAL_EPOCHS = 50
OFFICIAL_BATCH_SIZE = 2
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
    """HLS chips: the raster is already the task's six bands in order."""

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
        doy = row["timestamps"][0]
        if isinstance(doy, str):
            doy = int(np.datetime64(doy, "D").astype("datetime64[D]").astype(int) % 365)
        return {"image": image, "label": label, "date": int(doy)}


def train_normalization(manifest_path: Path, data_root: Path, cache_dir: Path):
    """Per-band train-split mean/std, computed the same way as for T1."""
    cache_file = cache_dir / "NORM_hls6_anysat.json"
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
        json.dumps({"mean": list(mean), "std": list(std), "chips": len(means)}),
        encoding="utf-8",
    )
    return mean.astype(np.float32), std.astype(np.float32)


def build_backbone(checkpoint: Path):
    add_upstream_to_path()
    from hubconf import AnySat  # type: ignore[import-not-found]  # upstream checkout

    model = AnySat(model_size="base", flash_attn=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.model.load_state_dict(state)
    return model


class AdapterProbe(nn.Module):
    def __init__(self, backbone, task_bands: list[str], num_classes: int, freeze_backbone: bool):
        super().__init__()
        self.backbone = backbone
        self.adapter = SpectralAdapter(task_bands)
        self.head = LiteSpatialHead(2 * 768, num_classes)
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
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            dense = self.backbone(
                {"s2": adapted, "s2_dates": dates},
                patch_size=PATCH_SIZE_M,
                output="dense",
                output_modality="s2",
            )
        dense = dense.permute(0, 3, 1, 2)
        return self.head(dense)


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = _int_env("ANYSAT_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("ANYSAT_WORKER_BATCH_SIZE", OFFICIAL_BATCH_SIZE)
    lr = _float_env("ANYSAT_WORKER_LR", 1e-3)
    workers = _int_env("ANYSAT_WORKER_NUM_WORKERS", 4)
    limit = _int_env("ANYSAT_WORKER_LIMIT_BATCHES", 0) or None

    manifest = Path(request.manifest_path)
    mean, std = train_normalization(manifest, data_root, Path(request.cache_root) / "anysat_norm")
    means = torch.tensor(mean, device=device).view(1, 1, -1, 1, 1)
    stds = torch.tensor(std, device=device).view(1, 1, -1, 1, 1)

    train_ds = HlsDataset(manifest, data_root, "train", crop=TRAIN_CROP)
    eval_ds = HlsDataset(manifest, data_root, EVAL_SPLIT)
    emit(
        run_id,
        "train.prepare",
        f"T2 adapter probe: {len(train_ds)} train / {len(eval_ds)} {EVAL_SPLIT} chips, "
        f"crop {TRAIN_CROP}, patch {PATCH_SIZE_M}m, freeze_backbone={freeze_backbone}",
    )

    backbone = build_backbone(checkpoint).to(device)
    model = AdapterProbe(backbone, task_bands, 2, freeze_backbone=freeze_backbone).to(device)
    adapter_params = count_parameters(model.adapter)
    head_params = count_parameters(model.head)
    writer.update_training(
        head_id="lite_spatial_head+spectral_adapter",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="train_stats_hls6+learned_spectral_adapter")
    writer.update(
        tags={
            "adapter_parameters": str(adapter_params),
            "head_parameters": str(head_params),
            "synthesized_anysat_bands": ",".join(model.adapter.synthesized_bands),
            "backbone_frozen": str(freeze_backbone),
            "patch_size_m": str(PATCH_SIZE_M),
            "train_crop_px": str(TRAIN_CROP),
            "eval_split": EVAL_SPLIT,
            "learning_rate": f"{lr:g}",
        }
    )
    emit(
        run_id,
        "train.model",
        f"adapter {adapter_params} (synthesizes {', '.join(model.adapter.synthesized_bands)}) "
        f"+ head {head_params}",
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in loader:
            images = batch["image"].to(device).unsqueeze(1)
            images = (images - means) / stds
            labels = batch["label"].to(device)
            dates = batch["date"].to(device).float().unsqueeze(1)
            if int((labels != -1).sum()) == 0:
                continue
            logits = model(images, dates)
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
            images = batch["image"].to(device).unsqueeze(1)
            images = (images - means) / stds
            dates = batch["date"].to(device).float().unsqueeze(1)
            logits = model(images, dates)
            matrix.update(
                batch["label"].numpy(), logits.argmax(dim=1).cpu().numpy(), ignore_index=-1
            )
            torch.cuda.empty_cache()
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
        f"({epochs} epochs, adapter {adapter_params} + head {head_params}, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
