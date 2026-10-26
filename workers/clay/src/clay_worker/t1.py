"""T1 (Sen1Floods11) training for the Clay v1.5 worker.

Clay is sensor-agnostic: it consumes the official six-band protocol directly
with a *derived* metadata profile (the six S2 bands selected from Clay's own
``sentinel-2-l2a`` metadata, keeping its wavelengths, mean, std and GSD), so
no spectral adapter is needed. Features come from Clay's own segmentation
recipe (SegmentEncoder loading ``model.encoder`` weights) and are decoded by
the shared ``LiteSpatialHead`` so every backbone is compared with the same
head. The backbone is frozen (B2-style representation probe).
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
from rs_backbone_eval.evaluation.groups import GroupedSegmentation
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
from rs_backbone_eval.splits import evaluation_split, extra_evaluation_splits
from rs_backbone_eval.worker_runtime import RunWriter

from .clay_probe import SENSOR, add_upstream_to_path, load_sensor_metadata, prepare_encoder

EVAL_SPLIT = evaluation_split("test")

# Task band -> Clay metadata band name for the derived S2 six-band profile.
TASK_BAND_TO_CLAY = {
    "BLUE": "blue",
    "GREEN": "green",
    "RED": "red",
    "NIR_NARROW": "nir08",
    "SWIR_1": "swir16",
    "SWIR_2": "swir22",
}
# Task band -> index into the 13-band Sen1Floods11 L1C chip.
CHIP_BAND_INDICES = {"BLUE": 1, "GREEN": 2, "RED": 3, "NIR_NARROW": 8, "SWIR_1": 11, "SWIR_2": 12}

# Sentinel-1 counterpart of T1: two polarisations, backscatter in dB. Clay v1.5
# was pretrained on Sentinel-1 RTC under the `sentinel-1-rtc` metadata entry,
# whose placeholder wavelengths (vv 3.5, vh 4.0) are what the encoder saw for
# radar; the physical C-band wavelength (~5.5e4 um) would be out of distribution.
# Sen1Floods11 ships GRD (not terrain-corrected) dB backscatter, recorded as a
# product difference in the run tags.
S1_SENSOR = "sentinel-1-rtc"
S1_CLAY_BANDS = {"VV": "vv", "VH": "vh"}
S1_BAND_INDICES = {"VV": 0, "VH": 1}
CLAY_LARGE = {
    "mask_ratio": 0,
    "patch_size": 8,
    "shuffle": False,
    "dim": 1024,
    "depth": 24,
    "heads": 16,
    "dim_head": 64,
    "mlp_ratio": 4,
}
TRAIN_CROP = 256
FULL_CHIP = 512
OFFICIAL_EPOCHS = 50
OFFICIAL_BATCH_SIZE = 4


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


def derive_metadata(source_root: Path, task_bands: list[str], sensor_kind: str = "optical") -> dict:
    """Auditable band subset of Clay's official sensor metadata.

    ``sensor_kind="sar"`` uses Clay's own ``sentinel-1-rtc`` entry (wavelengths
    and dB statistics), the same way the optical path uses ``sentinel-2-l2a``.
    """
    sensor = S1_SENSOR if sensor_kind == "sar" else SENSOR
    band_map = S1_CLAY_BANDS if sensor_kind == "sar" else TASK_BAND_TO_CLAY
    base = load_sensor_metadata(source_root, sensor)
    clay_bands = [band_map[band] for band in task_bands]
    return {
        "sensor": sensor,
        "bands": clay_bands,
        "gsd": base["gsd"],
        "band_order": clay_bands,
        "bands_meta": {
            "mean": {band: base["bands"]["mean"][band] for band in clay_bands},
            "std": {band: base["bands"]["std"][band] for band in clay_bands},
            "wavelength": {band: base["bands"]["wavelength"][band] for band in clay_bands},
        },
    }


class T1SixBandDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        split: str,
        task_bands: list[str],
        *,
        crop: int | None = None,
        sensor_kind: str = "optical",
        fill: list[float] | None = None,
    ):
        self.data_root = data_root
        self.crop = crop
        # Sen1Floods11 S1 chips hold NaN where the radar has no observation.
        # Those pixels get the band mean (0 after normalisation) and an ignore
        # label: an unobserved pixel is not scored.
        self.fill = None if fill is None else np.asarray(fill, dtype=np.float32)
        index_map = S1_BAND_INDICES if sensor_kind == "sar" else CHIP_BAND_INDICES
        self.band_indices = [index_map[band] for band in task_bands]
        self.rows: list[dict] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["split"] == split:
                    self.rows.append(row)
        self.rows = self.rows[: _int_env("CLAY_WORKER_EVAL_LIMIT", 0) or len(self.rows)]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        with rasterio.open(self.data_root / row["image_paths"][0]) as source:
            image = source.read().astype(np.float32)[self.band_indices]
        with rasterio.open(self.data_root / row["label_path"]) as source:
            label = source.read(1).astype(np.int64)
        if self.fill is not None:
            invalid = ~np.isfinite(image).all(axis=0)
            if invalid.any():
                image = np.where(invalid[None], self.fill[:, None, None], image).astype(np.float32)
                label = np.where(invalid, -1, label)
        if self.crop is not None and self.crop < image.shape[-1]:
            offset = (image.shape[-1] - self.crop) // 2
            image = image[:, offset : offset + self.crop, offset : offset + self.crop]
            label = label[offset : offset + self.crop, offset : offset + self.crop]
        return {"image": image, "label": label}


class ClayProbe(nn.Module):
    """Clay SegmentEncoder (frozen or fine-tuned) + shared lightweight head."""

    def __init__(self, encoder, num_classes: int, gsd: float, sensor: str, freeze_backbone: bool):
        super().__init__()
        self.encoder = encoder
        self.sensor = sensor
        self.gsd = gsd
        self.freeze_backbone = freeze_backbone
        self.head = LiteSpatialHead(CLAY_LARGE["dim"], num_classes)
        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def forward(self, pixels: torch.Tensor, waves: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = pixels.shape
        datacube = {
            "pixels": pixels,
            "time": torch.zeros(batch, 4, device=pixels.device),
            "latlon": torch.zeros(batch, 4, device=pixels.device),
            "gsd": torch.tensor(self.gsd, device=pixels.device),
            "waves": waves,
            "platform": self.sensor,
        }
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            tokens = self.encoder(datacube)  # [B, L, D]
        patch = CLAY_LARGE["patch_size"]
        grid_h, grid_w = height // patch, width // patch
        features = tokens.transpose(1, 2).reshape(batch, CLAY_LARGE["dim"], grid_h, grid_w)
        logits = self.head(features)  # [B, K, grid_h, grid_w]
        return torch.nn.functional.interpolate(
            logits, size=(height, width), mode="bilinear", align_corners=False
        )


def train_clay_t1(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    checkpoint: Path,
    task_bands: list[str],
    freeze_backbone: bool = True,
    sensor_kind: str = "optical",
) -> int:
    run_id = request.run_id
    data_root = Path(request.data_root)
    source_root = add_upstream_to_path()
    metadata = derive_metadata(source_root, task_bands, sensor_kind)
    device = os.environ.get("CLAY_WORKER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    epochs = _int_env("CLAY_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("CLAY_WORKER_BATCH_SIZE", OFFICIAL_BATCH_SIZE)
    lr = _float_env("CLAY_WORKER_LR", 1e-3)
    workers = _int_env("CLAY_WORKER_NUM_WORKERS", 4)
    crop = _int_env("CLAY_WORKER_TRAIN_CROP", TRAIN_CROP)
    limit = _int_env("CLAY_WORKER_LIMIT_BATCHES", 0) or None

    from claymodel.finetune.segment.factory import SegmentEncoder  # upstream checkout

    encoder = SegmentEncoder(
        **CLAY_LARGE, ckpt_path=None if random_init_requested() else str(checkpoint)
    ).to(device)
    prepare_encoder(encoder, freeze_backbone)
    model = ClayProbe(
        encoder,
        num_classes=2,
        gsd=metadata["gsd"],
        sensor=metadata["sensor"],
        freeze_backbone=freeze_backbone,
    ).to(device)
    head_params = count_parameters(model.head)
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model)

    band_order = metadata["bands"]
    means = torch.tensor(
        [metadata["bands_meta"]["mean"][b] for b in band_order], device=device
    ).view(1, -1, 1, 1)
    stds = torch.tensor([metadata["bands_meta"]["std"][b] for b in band_order], device=device).view(
        1, -1, 1, 1
    )
    fill = [metadata["bands_meta"]["mean"][b] for b in band_order] if sensor_kind == "sar" else None
    waves = torch.tensor(
        [metadata["bands_meta"]["wavelength"][b] for b in band_order],
        dtype=torch.float32,
        device=device,
    )

    train_ds = T1SixBandDataset(
        Path(request.manifest_path),
        data_root,
        "train",
        task_bands,
        crop=crop,
        sensor_kind=sensor_kind,
        fill=fill,
    )
    eval_ds = T1SixBandDataset(
        Path(request.manifest_path),
        data_root,
        EVAL_SPLIT,
        task_bands,
        sensor_kind=sensor_kind,
        fill=fill,
    )
    emit(
        run_id,
        "train.prepare",
        f"Clay probe: {len(train_ds)} train / {len(eval_ds)} test chips, crop {crop}, "
        f"derived meta bands {band_order}, device {device}",
    )
    writer.update_training(
        head_id="lite_spatial_head", trainable_parameters=trainable, total_parameters=total
    )
    writer.update_input(
        preprocessing_id="clay_official_s1_rtc_metadata"
        if sensor_kind == "sar"
        else "clay_official_s2_6band_subset"
    )
    writer.update(
        tags={
            "head_parameters": str(head_params),
            "backbone_frozen": str(freeze_backbone),
            "patch_size_px": str(CLAY_LARGE["patch_size"]),
            "patch_stride_m": f"{CLAY_LARGE['patch_size'] * metadata['gsd']:g}",
            "train_crop_px": str(crop),
            "derived_metadata_bands": ",".join(band_order),
            "learning_rate": f"{lr:g}",
            "sensor_kind": sensor_kind,
            "clay_metadata_sensor": metadata["sensor"],
            **(
                {
                    "s1_product": "Sen1Floods11 GRD dB; Clay pretrained on S1 RTC",
                    "nonfinite_policy": "band-mean fill; label ignored (no radar observation)",
                }
                if sensor_kind == "sar"
                else {}
            ),
            "device": device,
        }
    )
    emit(
        run_id,
        "train.model",
        f"head {head_params} trainable of {total} total (frozen Clay encoder)",
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)

    select = not freeze_backbone
    best = BestCheckpoint()
    val_every = validation_every()
    val_ds = T1SixBandDataset(
        Path(request.manifest_path),
        data_root,
        "validation",
        task_bands,
        sensor_kind=sensor_kind,
        fill=fill,
    )

    def score(dataset, perturbation: str | None = None) -> tuple[dict, dict[str, list[list[int]]]]:
        spec = resolve(perturbation) if perturbation else {"kind": "none"}
        drop = (
            band_indices_to_drop(task_bands, spec["bands"]) if spec["kind"] == "band_drop" else []
        )
        model.eval()
        collector = GroupedSegmentation(num_classes=2, ignore_index=-1)
        with torch.inference_mode():
            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
            for index, batch in enumerate(loader):
                pixels = (batch["image"].to(device) - means) / stds
                if spec["kind"] == "band_drop":
                    pixels = apply_band_drop(pixels, drop)
                elif spec["kind"] == "resolution":
                    pixels = apply_resolution(pixels, int(spec["factor"]))
                logits = model(pixels, waves)
                collector.update(
                    [dataset.rows[index]["group_id"]],
                    batch["label"].numpy(),
                    logits.argmax(dim=1).cpu().numpy(),
                )
        result = binary_metrics(collector.global_matrix, foreground=1)
        result["water_iou"] = result.pop("foreground_iou")
        return result, collector.group_confusion()

    started = time.time()
    optimizer_steps = 0
    phases: dict[str, float] = {}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    train_started = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            pixels = (batch["image"].to(device) - means) / stds
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            if int((labels != -1).sum()) == 0:
                continue
            logits = model(pixels, waves)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
            if limit and steps >= limit:
                break
        optimizer_steps += steps
        mean_loss = total_loss / max(steps, 1)
        writer.add_curve("loss", "train", epoch, mean_loss)
        emit(
            run_id,
            "train.epoch",
            f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}",
            progress=(epoch + 1) / epochs,
        )
        if select and ((epoch + 1) % val_every == 0 or epoch + 1 == epochs):
            validation, _ = score(val_ds)
            writer.add_curve("water_iou", "validation", epoch, validation["water_iou"])
            best.offer(validation["water_iou"], epoch + 1, model)
    phases["train"] = time.time() - train_started
    wall = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

    if select:
        best.restore(model)
        best.save(writer.run_dir / "best_checkpoint.pt")
        writer.add_artifact("checkpoint", "best_checkpoint.pt", "validation-selected weights")
        writer.update(
            tags={
                "selection": "best_validation_water_iou",
                "selected_epoch": str(best.epoch),
                "validation_water_iou": f"{best.score:.4f}",
                "validation_every": str(val_every),
            }
        )
    emit(
        run_id, "train.eval", f"evaluating on {len(eval_ds)} {EVAL_SPLIT} chips at full {FULL_CHIP}"
    )
    eval_started = time.time()
    metrics, group_json = score(eval_ds)
    phases["eval"] = time.time() - eval_started
    for name in perturbations_from_env():
        spec = resolve(name)
        if spec["kind"] == "none" or name == "none":
            continue
        perturbed, _ = score(eval_ds, name)
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
    extra_splits = extra_evaluation_splits(EVAL_SPLIT)
    for split in extra_splits:
        holdout_ds = T1SixBandDataset(
            Path(request.manifest_path),
            data_root,
            split,
            task_bands,
            sensor_kind=sensor_kind,
            fill=fill,
        )
        holdout, _ = score(holdout_ds)
        writer.add_metrics(
            to_metric_records(
                holdout, primary_name="water_iou", direction="maximize", split=split, primary=False
            )
        )
        emit(
            run_id,
            "train.holdout",
            f"{split} ({len(holdout_ds)} chips): {holdout['water_iou']:.4f}",
        )
    if extra_splits:
        writer.update(tags={"extra_eval_splits": ",".join(extra_splits)})
    writer.add_metrics(
        to_metric_records(metrics, primary_name="water_iou", direction="maximize", split=EVAL_SPLIT)
    )
    writer.add_confusion(group_json, 2)
    writer.update_training(
        epochs_completed=epochs,
        optimizer_steps=optimizer_steps,
        wall_time_seconds=wall,
        peak_gpu_memory_mb=peak_mb,
        phases=phases,
    )
    summary = (
        f"water_iou={metrics['water_iou']:.4f} mean_iou={metrics['mean_iou']:.4f} "
        f"({epochs} epochs, head {head_params} params, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
