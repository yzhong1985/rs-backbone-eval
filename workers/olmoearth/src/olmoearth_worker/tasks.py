"""T2/T3/T4 frozen-head training for OlmoEarth.

Shares the T1 recipe (six-band chips -> constant-fill 12-band adapter -> frozen
encoder -> ``LiteSpatialHead``) but generalises three axes:

* single- vs multi-date chips: the encoder returns one token grid per date and
  the head reads the **concatenation over dates**, so the order pre/during/post
  is preserved (T1's single date reduces to the same tensor as before);
* the objective: plain CE for T2, class-weighted CE for T3 (the design doc's
  fixed objective), focal loss for T4;
* the output space: binary IoU for T2/T4, five-class mIoU for T3.

Read-out choice (documented for review): dates are concatenated, not averaged
and not fused by a learned temporal module. A temporal head is a follow-up if
T3/T5 show the frozen per-date tokens need it; averaging would discard the
pre/during/post order that defines burn intensity.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rs_backbone_eval.data.manifest import audit_task_manifest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
from rs_backbone_eval.evaluation.segmentation import binary_metrics, multiclass_metrics
from rs_backbone_eval.heads import LiteSpatialHead, count_parameters
from rs_backbone_eval.losses import FocalLoss, WeightedCrossEntropy, inverse_frequency_weights
from rs_backbone_eval.registry import TaskSpec
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.temporal_io import read_hls_dates, read_landslide_sample
from rs_backbone_eval.worker_runtime import RunWriter

from .bands import (
    OLMOEARTH_S2L2A_BANDS,
    bandset_usable,
    to_olmoearth_s2l2a,
)
from .t1_frozen import (
    ENCODER_INPUT,
    PATCH_SIZE,
    _fail,
    _int_env,
    _timestamps,
    build_encoder,
    frozen_backbone_protocol,
    record_checkpoint_hash,
)

BATCH_SIZE = 4
OFFICIAL_EPOCHS = 50
IGNORE_INDEX = -1


@dataclass(frozen=True)
class Recipe:
    task_id: str
    num_classes: int
    eval_split: str
    select_split: str
    epochs: int
    loss: str
    metric_name: str
    multiclass: bool


RECIPES: dict[str, Recipe] = {
    "T2": Recipe("T2", 2, "official_validation", "validation", 50, "ce", "burn_scar_iou", False),
    "T3": Recipe("T3", 5, "test", "validation", 50, "weighted_ce", "mean_iou", True),
    "T4": Recipe("T4", 2, "test", "validation", 50, "focal", "mean_iou", False),
}


class HlsDataset(Dataset):
    """HLS chips (one raster per date) and integer labels from one split.

    Band order follows ``row["bands"]``; ``read_hls_dates`` keeps the raster's
    channel order, and ``to_olmoearth_s2l2a`` maps each name to its OlmoEarth
    position, so T2 (``NIR_BROAD``) and T3 (``NIR``) both land on B08.
    """

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
        images = read_hls_dates([self.data_root / path for path in row["image_paths"]])
        label = _read_label(self.data_root / row["label_path"])
        return {
            "images": images,
            "label": label,
            "group_id": row["group_id"],
            "bands": list(row["bands"]),
            # ``pre``/``during``/``post`` are not dates; use a stable increasing
            # sequence so the encoder has an order but no false calendar claim.
            "timestamps": torch.tensor(
                _relative_timestamps(row.get("timestamps") or [], len(row["image_paths"])),
                dtype=torch.long,
            ),
        }


class LandslideDataset(Dataset):
    """Landslide4Sense patches: the 14-band H5 subset to the task's six bands."""

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
            list(row["bands"]),
            size=ENCODER_INPUT,
        )
        return {
            "images": image[None, ...],
            "label": label.astype(np.int64),
            "group_id": row["group_id"],
            "bands": list(row["bands"]),
            "timestamps": torch.tensor([[1, 0, 2020]], dtype=torch.long),
        }


def _read_label(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as source:
        return source.read(1).astype(np.int64)


def _relative_timestamps(stamps: list[str], count: int) -> list[list[int]]:
    parsed = _timestamps([str(stamp) for stamp in stamps]) if _looks_like_dates(stamps) else []
    if len(parsed) == count:
        return parsed
    # Order-only sequence (day 1..T of a fixed month/year); documented in the
    # run tags so no reader mistakes it for real acquisition dates.
    return [[index + 1, 0, 2020] for index in range(count)]


def _looks_like_dates(stamps: list[str]) -> bool:
    import datetime

    if not stamps:
        return False
    try:
        for stamp in stamps:
            datetime.date.fromisoformat(str(stamp)[:10])
    except ValueError:
        return False
    return True


def band_statistics_of(dataset: Dataset, *, limit: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-band mean/std over a sample of the training split (all dates pooled)."""
    collected: list[torch.Tensor] = []
    for index in range(min(limit, len(dataset))):
        images = np.asarray(dataset[index]["images"])  # [T, C, H, W]
        collected.append(torch.as_tensor(images).permute(1, 0, 2, 3).reshape(images.shape[1], -1))
    if not collected:
        raise ValueError("cannot compute band statistics from an empty training split")
    pixels = torch.cat(collected, dim=1)
    return pixels.mean(dim=1), pixels.std(dim=1).clamp_min(1e-6)


def dataset_for(recipe: Recipe, manifest: Path, data_root: Path, split: str) -> Dataset:
    if recipe.task_id == "T4":
        return LandslideDataset(manifest, data_root, split)
    return HlsDataset(manifest, data_root, split)


def encode_dataset(
    dataset: Dataset,
    encoder,
    means: torch.Tensor,
    stds: torch.Tensor,
    device: str,
    emit_event: Callable[..., None],
    run_id: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Frozen tokens for every chip: ``(B, T*D, grid, grid)``.

    Dates are concatenated along the channel axis, preserving pre/during/post
    order. The mask is band-set level and identical for every date (the adapter
    fills the same bands), so one mask tensor is expanded over dates.
    """
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

    first = dataset[0]
    bands = list(first["bands"])
    usable = bandset_usable(bands)
    online = [MaskValue.ONLINE_ENCODER.value if ok else MaskValue.MISSING.value for ok in usable]

    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    groups: list[str] = []
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    for batch in loader:
        # Move before the standardisation: means/stds live on the compute device
        # and mixing them with a CPU chip raises at the first subtraction.
        chips = batch["images"].float().to(device)  # [B, T, C, H, W]
        width, dates = chips.shape[0], chips.shape[1]
        channels = chips.shape[2]
        chips = torch.nn.functional.interpolate(
            chips.reshape(width * dates, channels, *chips.shape[3:]),
            size=(ENCODER_INPUT, ENCODER_INPUT),
            mode="bilinear",
            align_corners=False,
        ).reshape(width, dates, channels, ENCODER_INPUT, ENCODER_INPUT)
        chips = (chips - means.view(1, 1, -1, 1, 1)) / stds.view(1, 1, -1, 1, 1)
        expanded = torch.empty(
            width, ENCODER_INPUT, ENCODER_INPUT, dates, len(OLMOEARTH_S2L2A_BANDS), device=device
        )
        for sample in range(width):
            for date in range(dates):
                expanded[sample, :, :, date] = (
                    torch.from_numpy(
                        to_olmoearth_s2l2a(chips[sample, date].cpu().numpy(), bands, fill=0.0)
                    )
                    .to(device)
                    .permute(1, 2, 0)
                )
        mask = (
            torch.tensor(online, dtype=torch.long, device=device)
            .view(1, 1, 1, 1, len(online))
            .expand(width, ENCODER_INPUT, ENCODER_INPUT, dates, len(online))
        )
        sample = MaskedOlmoEarthSample(
            timestamps=batch["timestamps"].to(device),
            sentinel2_l2a=expanded,
            sentinel2_l2a_mask=mask,
        )
        with torch.inference_mode():
            tokens = encoder(sample, patch_size=PATCH_SIZE)["tokens_and_masks"]
            dense = tokens.sentinel2_l2a  # [B, P_H, P_W, T, BS, D]
            # (B, P, P, T, D) -> (B, T*D, P, P): order-preserving date axis.
            fused = dense[:, :, :, :, 0, :].permute(0, 3, 4, 1, 2).flatten(1, 2)
            features.append(fused.contiguous().cpu())
        labels.append(batch["label"])
        groups.extend(batch["group_id"])
    emit_event(
        run_id,
        "probe.encode",
        f"encoded {len(dataset)} chips x {dates} date(s) into {features[0].shape[-1]}x"
        f"{features[0].shape[-1]} frozen tokens",
    )
    return torch.cat(features), torch.cat(labels), groups


def build_criterion(recipe: Recipe, dataset: Dataset, device: str) -> nn.Module:
    if recipe.loss == "focal":
        return FocalLoss(gamma=2.0, ignore_index=IGNORE_INDEX)
    if recipe.loss == "weighted_ce":
        counts = [0] * recipe.num_classes
        for index in range(len(dataset)):
            label = np.asarray(dataset[index]["label"]).ravel()
            valid = label[label != IGNORE_INDEX]
            if valid.size:
                binned = np.bincount(valid, minlength=recipe.num_classes)
                for class_index in range(recipe.num_classes):
                    counts[class_index] += int(binned[class_index])
        weights = inverse_frequency_weights(counts).to(device)
        return WeightedCrossEntropy(weight=weights, ignore_index=IGNORE_INDEX)
    return nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)


def _predict(model, features: torch.Tensor, label_size: int) -> torch.Tensor:
    logits = model(features)
    return torch.nn.functional.interpolate(
        logits, size=(label_size, label_size), mode="bilinear", align_corners=False
    )


def _metrics(recipe: Recipe, matrix: ConfusionMatrix) -> dict[str, float]:
    if recipe.multiclass:
        return multiclass_metrics(matrix)
    values = binary_metrics(matrix, foreground=1)
    values[recipe.metric_name] = values.pop("foreground_iou")
    return values


def train_task(
    request,
    writer: RunWriter,
    event_log: Callable[..., None],
    *,
    task_spec: TaskSpec,
    checkpoint_dir: Path,
) -> int:
    run_id = request.run_id
    recipe = RECIPES[task_spec.task_id]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = _int_env("OLMOEARTH_WORKER_EPOCHS", recipe.epochs)
    batch_size = _int_env("OLMOEARTH_WORKER_BATCH_SIZE", BATCH_SIZE)
    lr = _float_env_olmoearth(1e-3)
    limit = _int_env("OLMOEARTH_WORKER_LIMIT_BATCHES", 0) or None
    stats_rows = _int_env("OLMOEARTH_WORKER_STATS_ROWS", 64)

    ok, message = frozen_backbone_protocol(Path(request.protocol_config_path))
    if not ok:
        return _fail(writer, run_id, "train.protocol", message)
    audit = audit_task_manifest(Path(request.manifest_path), task_spec)
    if audit.error_count - len(audit.leakage):
        return _fail(
            writer, run_id, "train.manifest", f"manifest audit failed: {audit.all_errors()[:2]}"
        )

    record_checkpoint_hash(writer, checkpoint_dir)
    data_root = Path(request.data_root)
    manifest = Path(request.manifest_path)
    train_ds = dataset_for(recipe, manifest, data_root, "train")
    eval_ds = dataset_for(recipe, manifest, data_root, recipe.eval_split)
    if not train_ds or not eval_ds:
        return _fail(writer, run_id, "train.data", "train or evaluation split is empty")

    encoder = build_encoder(checkpoint_dir, load_weights=not random_init_requested()).to(device)
    encoder_params = count_parameters(encoder)
    mean, std = band_statistics_of(train_ds, limit=stats_rows)
    means = mean.to(device)
    stds = std.to(device)

    started = time.time()
    extract_started = time.time()
    train_features, train_labels, _ = encode_dataset(
        train_ds, encoder, means, stds, device, event_log, run_id
    )
    eval_features, eval_labels, groups = encode_dataset(
        eval_ds, encoder, means, stds, device, event_log, run_id
    )
    phases = {"extract": time.time() - extract_started}

    model = OlmoEarthTaskHead(encoder, train_features.shape[1], recipe.num_classes).to(device)
    head_params = count_parameters(model.head)
    writer.update_training(
        head_id="lite_spatial_head",
        trainable_parameters=head_params,
        total_parameters=encoder_params + head_params,
    )
    writer.update_input(preprocessing_id="train_stats_6band+constant_fill_adapter")
    writer.update(
        tags={
            "track": "frozen_adapter",
            "task": recipe.task_id,
            "head_parameters": str(head_params),
            "encoder_parameters": str(encoder_params),
            "backbone_frozen": "True",
            "band_adapter": "constant_fill_12band",
            "adapter_note": "missing OlmoEarth bands filled with the standardised mean; "
            "band-sets with no real band masked MISSING",
            "date_readout": "concat over dates (T*D channels); order preserved",
            "loss": recipe.loss,
            "eval_split": recipe.eval_split,
            "learning_rate": f"{lr:g}",
            "device": device,
        }
    )
    event_log(
        run_id,
        "train.model",
        f"frozen encoder {encoder_params} + head {head_params} params, "
        f"{recipe.task_id} {recipe.num_classes} classes",
        progress=0.1,
    )

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = build_criterion(recipe, train_ds, device)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    label_size = train_labels.shape[-1]

    train_started = time.time()
    optimizer_steps = 0
    for epoch in range(epochs):
        permutation = torch.randperm(train_features.shape[0], device=device)
        total_loss, steps = 0.0, 0
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size]
            logits = _predict(model, train_features[index], label_size)
            loss = criterion(logits, train_labels[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
            if limit and steps >= limit:
                break
        optimizer_steps += steps
        mean_loss = total_loss / max(steps, 1)
        writer.add_curve("loss", "train", epoch, mean_loss)
        event_log(
            run_id,
            "train.epoch",
            f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}",
            progress=0.4 + 0.5 * (epoch + 1) / epochs,
        )
    phases["train"] = time.time() - train_started
    wall = time.time() - started

    event_log(run_id, "train.eval", f"evaluating on {len(eval_ds)} {recipe.eval_split} chips")
    eval_started = time.time()
    matrix = ConfusionMatrix(num_classes=recipe.num_classes)
    per_group: dict[str, ConfusionMatrix] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, eval_features.shape[0], 8):
            index = slice(start, start + 8)
            logits = _predict(model, eval_features[index].to(device), label_size)
            predictions = logits.argmax(dim=1).cpu().numpy()
            targets = eval_labels[index].numpy()
            matrix.update(targets, predictions, ignore_index=IGNORE_INDEX)
            for offset in range(targets.shape[0]):
                row = eval_ds.rows[start + offset]
                local = per_group.setdefault(
                    row["group_id"], ConfusionMatrix(num_classes=recipe.num_classes)
                )
                local.update(
                    targets[offset][None],
                    predictions[offset][None],
                    ignore_index=IGNORE_INDEX,
                )
    phases["eval"] = time.time() - eval_started
    values = _metrics(recipe, matrix)
    writer.add_metrics(
        to_metric_records(
            values, primary_name=recipe.metric_name, direction="maximize", split=recipe.eval_split
        )
    )
    writer.add_confusion(
        {group: local.counts.tolist() for group, local in per_group.items()},
        recipe.num_classes,
    )
    _ = groups
    writer.update_training(
        epochs_completed=epochs,
        optimizer_steps=optimizer_steps,
        wall_time_seconds=wall,
        peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / (1024**2)
        if device == "cuda"
        else 0.0,
        phases=phases,
    )
    summary = (
        f"{recipe.metric_name}={values[recipe.metric_name]:.4f} "
        f"({epochs} epochs, frozen encoder {encoder_params} + head {head_params} params, "
        f"{wall / 60:.1f} min)"
    )
    if epochs != recipe.epochs or limit:
        summary += " [SMOKE]"
    writer.complete(summary)
    event_log(run_id, "train.metrics", summary)
    return 0


class OlmoEarthTaskHead(nn.Module):
    """Frozen encoder container plus the shared head over concatenated dates."""

    def __init__(self, encoder, feature_channels: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = LiteSpatialHead(feature_channels, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


def _float_env_olmoearth(default: float) -> float:
    return float(os.environ.get("OLMOEARTH_WORKER_LR", default))
