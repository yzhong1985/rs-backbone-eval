"""T1 (Sen1Floods11) training for the AnySat worker: six-band spectral adapter
plus a shared lightweight head over AnySat's native S2 dense features.

Track: representation probe with an explicitly trained input adapter (the
design forbids zero-padding the four missing bands). The backbone is frozen by
default; the adapter and the head are the only trainable modules and their
parameter counts are reported separately from the task head.
"""

from __future__ import annotations

import gc
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
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.timeutil import day_of_year
from rs_backbone_eval.worker_runtime import RunWriter

from .s2_probe import add_upstream_to_path, read_chip
from .spectral_adapter import SpectralAdapter, chip_indices_for_bands

PATCH_SIZE_M = 160  # the only patch size that fits a full 512 chip (probe sweep)
TRAIN_CROP = 256  # training crops; evaluation runs on full 512 chips
FULL_CHIP = 512
OFFICIAL_EPOCHS = 50
OFFICIAL_BATCH_SIZE = 2


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


class T1SixBandDataset(Dataset):
    """Six task bands plus the label chip; returns numpy for worker-safe loading."""

    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        split: str,
        task_bands: list[str],
        *,
        crop: int | None = None,
    ):
        self.data_root = data_root
        self.crop = crop
        self.band_indices = chip_indices_for_bands(task_bands)
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
        image = read_chip(self.data_root / row["image_paths"][0], band_indices=self.band_indices)
        with rasterio.open(self.data_root / row["label_path"]) as source:
            label = source.read(1).astype(np.int64)
        if self.crop is not None and self.crop < image.shape[-1]:
            offset = (image.shape[-1] - self.crop) // 2
            image = image[:, offset : offset + self.crop, offset : offset + self.crop]
            label = label[offset : offset + self.crop, offset : offset + self.crop]
        doy = row["timestamps"][0]
        if isinstance(doy, str):
            doy = day_of_year(doy)
        return {"image": image, "label": label, "date": int(doy)}


def train_normalization(manifest_path: Path, data_root: Path, bands: list[str], cache_dir: Path):
    """Per-band train-split mean/std for the task's six bands (AnySat's method)."""
    cache_file = cache_dir / "NORM_task6_anysat.json"
    if cache_file.is_file():
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if payload.get("bands") == bands:
            return (
                np.asarray(payload["mean"], dtype=np.float32),
                np.asarray(payload["std"], dtype=np.float32),
            )
    indices = chip_indices_for_bands(bands)
    means, stds = [], []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["split"] != "train":
                continue
            chip = read_chip(data_root / row["image_paths"][0], band_indices=indices)
            means.append(chip.mean(axis=(1, 2)))
            stds.append(chip.std(axis=(1, 2)))
    mean = np.stack(means).mean(axis=0).astype(float)
    std = np.stack(stds).mean(axis=0).astype(float)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"bands": bands, "mean": list(mean), "std": list(std)}), encoding="utf-8"
    )
    return mean.astype(np.float32), std.astype(np.float32)


def build_backbone(checkpoint: Path):
    add_upstream_to_path()
    from hubconf import AnySat  # type: ignore[import-not-found]  # upstream checkout

    model = AnySat(model_size="base", flash_attn=False)
    if not random_init_requested():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.model.load_state_dict(state)
    return model


class AdapterProbe(nn.Module):
    """Backbone (frozen or fine-tuned) + trainable spectral adapter + head."""

    def __init__(self, backbone, task_bands: list[str], num_classes: int, freeze_backbone: bool):
        super().__init__()
        self.backbone = backbone
        self.adapter = SpectralAdapter(task_bands)
        # A trainable adapter sits upstream of the backbone, so the graph
        # must stay alive through the frozen backbone or it gets no gradient.
        self.adapter_trainable = any(
            parameter.requires_grad for parameter in self.adapter.parameters()
        )
        self.head = LiteSpatialHead(2 * 768, num_classes)  # AnySat dense S2 is 2*embed_dim
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
        adapted = self.adapter(images)  # [B,T,10,H,W]
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
        dense = dense.permute(0, 3, 1, 2)  # [B,H,W,2D] -> [B,2D,H,W]
        return self.head(dense)


def _evaluate(
    model, dataset, means, stds, device, task_bands, perturbation: str | None = None
) -> dict[str, float]:
    spec = resolve(perturbation) if perturbation else {"kind": "none"}
    drop = band_indices_to_drop(task_bands, spec["bands"]) if spec["kind"] == "band_drop" else []
    means4 = means.reshape(1, -1, 1, 1)
    stds4 = stds.reshape(1, -1, 1, 1)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    matrix = ConfusionMatrix(num_classes=2)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = (batch["image"].to(device) - means4) / stds4
            if spec["kind"] == "band_drop":
                images = apply_band_drop(images, drop)
            elif spec["kind"] == "resolution":
                images = apply_resolution(images, int(spec["factor"]))
            images = images.unsqueeze(1)
            dates = batch["date"].to(device).float().unsqueeze(1)
            logits = model(images, dates)
            matrix.update(
                batch["label"].numpy(), logits.argmax(dim=1).cpu().numpy(), ignore_index=-1
            )
            del images, dates, logits
            torch.cuda.empty_cache()
    metrics = binary_metrics(matrix, foreground=1)
    metrics["water_iou"] = metrics.pop("foreground_iou")
    return metrics


def train_adapter(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    checkpoint: Path,
    task_bands: list[str],
    freeze_backbone: bool = True,
) -> int:
    run_id = request.run_id
    data_root = Path(request.data_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = _int_env("ANYSAT_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("ANYSAT_WORKER_BATCH_SIZE", OFFICIAL_BATCH_SIZE)
    lr = _float_env("ANYSAT_WORKER_LR", 1e-3)
    workers = _int_env("ANYSAT_WORKER_NUM_WORKERS", 4)
    limit = _int_env("ANYSAT_WORKER_LIMIT_BATCHES", 0) or None
    train_crop = _int_env("ANYSAT_WORKER_TRAIN_CROP", TRAIN_CROP)
    # The track is protocol-driven: B2 freezes the backbone, B1 fine-tunes it.
    freeze = freeze_backbone
    if os.environ.get("ANYSAT_WORKER_FINETUNE"):
        freeze = os.environ["ANYSAT_WORKER_FINETUNE"] in {"0", "false"}

    mean, std = train_normalization(
        Path(request.manifest_path),
        data_root,
        task_bands,
        Path(request.cache_root) / "anysat_norm",
    )
    means = torch.tensor(mean, device=device).view(1, 1, -1, 1, 1)
    stds = torch.tensor(std, device=device).view(1, 1, -1, 1, 1)

    train_ds = T1SixBandDataset(
        Path(request.manifest_path), data_root, "train", task_bands, crop=train_crop
    )
    val_ds = T1SixBandDataset(Path(request.manifest_path), data_root, "validation", task_bands)
    eval_ds = T1SixBandDataset(Path(request.manifest_path), data_root, "test", task_bands)
    emit(
        run_id,
        "train.prepare",
        f"adapter probe: {len(train_ds)} train / {len(eval_ds)} test chips, "
        f"crop {train_crop}, patch {PATCH_SIZE_M}m, freeze_backbone={freeze}",
    )

    backbone = build_backbone(checkpoint).to(device)
    model = AdapterProbe(backbone, task_bands, num_classes=2, freeze_backbone=freeze).to(device)
    adapter_params = count_parameters(model.adapter)
    head_params = count_parameters(model.head)
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model)
    writer.update_training(
        head_id="lite_spatial_head+spectral_adapter",
        trainable_parameters=trainable,
        total_parameters=total,
    )
    writer.update_input(preprocessing_id="train_stats_6band+learned_spectral_adapter")
    writer.update(
        tags={
            "adapter_parameters": str(adapter_params),
            "head_parameters": str(head_params),
            "backbone_frozen": str(freeze),
            "patch_size_m": str(PATCH_SIZE_M),
            "train_crop_px": str(train_crop),
            "learning_rate": f"{lr:g}",
        }
    )
    emit(
        run_id,
        "train.model",
        f"adapter {adapter_params} + head {head_params} = {trainable} trainable of {total} total "
        f"({'frozen' if freeze else 'fine-tuned'} backbone)",
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)

    select = not freeze
    best = BestCheckpoint()
    val_every = validation_every()

    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            images = batch["image"].to(device).unsqueeze(1)
            images = (images - means) / stds
            labels = batch["label"].to(device)
            dates = batch["date"].to(device).float().unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            if int((labels != -1).sum()) == 0:
                continue
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(images, dates)
            loss = criterion(logits.float(), labels)
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
            optimizer.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            validation = _evaluate(model, val_ds, means, stds, device, task_bands)
            writer.add_curve("water_iou", "validation", epoch, validation["water_iou"])
            best.offer(validation["water_iou"], epoch + 1, model)
            torch.cuda.empty_cache()
    wall = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

    if select:
        best.restore(model)
        writer.update(
            tags={
                "selection": "best_validation_water_iou",
                "selected_epoch": str(best.epoch),
                "validation_water_iou": f"{best.score:.4f}",
                "validation_every": str(val_every),
            }
        )
    torch.cuda.empty_cache()
    emit(run_id, "train.eval", f"evaluating on {len(eval_ds)} test chips at full 512 resolution")
    metrics = _evaluate(model, eval_ds, means, stds, device, task_bands)
    for name in perturbations_from_env():
        if resolve(name)["kind"] == "none":
            continue
        perturbed = _evaluate(model, eval_ds, means, stds, device, task_bands, name)
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
    records = to_metric_records(
        metrics, primary_name="water_iou", direction="maximize", split="test"
    )
    writer.add_metrics(records)
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=wall, peak_gpu_memory_mb=peak_mb
    )
    summary = (
        f"water_iou={metrics['water_iou']:.4f} mean_iou={metrics['mean_iou']:.4f} "
        f"({epochs} epochs, adapter {adapter_params} + head {head_params} params, "
        f"{wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
