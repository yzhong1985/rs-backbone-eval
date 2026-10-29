"""T9 (iSAID) for the LIM worker: tile training and whole-image sliding-window test.

Data comes from the tile-level manifest (one row per window), so training reads
a window and pads its label with the task's ignore index (255). Evaluation must
not do that: the official mIoU is computed on the whole image, so every pixel is
scored once through ``rs_backbone_eval.evaluation.windows`` -- overlapping
logits are averaged before the argmax and padding never enters the confusion
matrix.

Batch: the protocol fixes the *effective* batch at 8 (micro 1 x accumulation 8).
A different product is a protocol error, not a tuning choice, so this module
refuses to run (design 2026-10-29 §2.2).
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
from rs_backbone_eval.data.isaid_io import band_counts, read_label, read_rgb
from rs_backbone_eval.data.tiling import pad_tile
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.confusion import GroupConfusion
from rs_backbone_eval.evaluation.segmentation import multiclass_metrics
from rs_backbone_eval.evaluation.windows import evaluate_image
from rs_backbone_eval.heads import UPerNetHead, count_parameters
from rs_backbone_eval.matrix import stopped_short_of_budget
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.training import scale_accumulated_gradients
from rs_backbone_eval.worker_runtime import RunWriter

from .mit import MixMIMEncoder, load_encoder, mit_huge

# The budget gate lives in the control plane (matrix.stopped_short_of_budget):
# T9 is measured in optimizer steps, and a second copy here is how a run that
# did its full 25,000 steps got stamped REDUCED-BUDGET (review 2026-10-29).
OFFICIAL_EFFECTIVE_BATCH = 8
IGNORE_INDEX = 255


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


def load_rows(manifest_path: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["split"] == split:
                    rows.append(row)
    return rows


def tile_geometry(rows: list[dict]) -> tuple[int, int]:
    """Infer ``(crop, stride)`` from the manifest so the run matches the build."""
    crop = int(rows[0]["window"][2])
    xs: dict[str, set[int]] = {}
    for row in rows:
        xs.setdefault(row["group_id"], set()).add(int(row["window"][0]))
    steps = [
        sorted(b - a for a, b in zip(sorted(origins), sorted(origins)[1:], strict=False) if b > a)
        for origins in xs.values()
        if len(origins) > 1
    ]
    strides = [step[0] for step in steps if step]
    stride = min(strides) if strides else crop // 2
    return crop, stride


class IsaidTiles(Dataset):
    """Training tiles: read the manifest window, pad label with ignore."""

    def __init__(
        self,
        rows: list[dict],
        data_root: Path,
        crop: int,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool = True,
    ):
        self.rows = rows
        self.data_root = data_root
        self.crop = crop
        self.mean = mean.reshape(1, 1, 3)
        self.std = std.reshape(1, 1, 3)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        window = tuple(int(value) for value in row["window"])
        image = read_rgb(self.data_root / row["image_paths"][0], window).astype(np.float32)
        label = read_label(self.data_root / row["label_path"], window)
        if image.shape[:2] != label.shape:
            raise ValueError(
                f"{row['sample_id']}: image {image.shape[:2]} != label {label.shape} "
                "(the mask and the DOTA image disagree)"
            )
        if self.augment and np.random.random() < 0.5:
            image = image[:, ::-1].copy()
            label = label[:, ::-1].copy()
        if self.augment and np.random.random() < 0.5:
            image = image[::-1].copy()
            label = label[::-1].copy()
        image = (image - self.mean) / self.std
        image = pad_tile(image, self.crop, fill=0.0)
        label = pad_tile(label, self.crop, fill=IGNORE_INDEX)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "label": torch.from_numpy(label.astype(np.int64)),
            "group": row["group_id"],
        }


class LimSegmentation(nn.Module):
    """MiT-Huge encoder + shared UPerNet head."""

    def __init__(self, encoder: MixMIMEncoder, num_classes: int, fpn_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.head = UPerNetHead(list(encoder.embed_dims), num_classes=num_classes, fpn_dim=fpn_dim)

    def forward(self, pixels: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        return self.head(self.encoder(pixels), output_size=output_size)


def _norm_statistics(
    rows: list[dict], data_root: Path, crop: int, sample: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(rows), size=min(sample, len(rows)), replace=False)
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for index in picks:
        row = rows[int(index)]
        window = tuple(int(value) for value in row["window"])
        image = read_rgb(data_root / row["image_paths"][0], window).astype(np.float64)
        if image.size == 0:
            continue
        total += image.reshape(-1, image.shape[-1]).sum(axis=0)
        total_sq += (image.reshape(-1, image.shape[-1]) ** 2).sum(axis=0)
        count += image.shape[0] * image.shape[1]
    if not count:
        raise ValueError("no training pixels to compute normalisation statistics")
    mean = total / count
    variance = np.maximum(total_sq / count - mean**2, 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def train_t9(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    checkpoint: Path,
    freeze_backbone: bool,
    num_classes: int,
    task_id: str,
    metric_name: str = "mean_iou",
) -> int:
    run_id = request.run_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Only an upper bound on the loop when LIM_WORKER_TOTAL_STEPS is set.
    epochs = _int_env("LIM_WORKER_EPOCHS", 50)
    micro_batch = _int_env("LIM_WORKER_MICRO_BATCH", 1)
    accumulation = _int_env("LIM_WORKER_GRAD_ACCUM", OFFICIAL_EFFECTIVE_BATCH)
    if micro_batch * accumulation != OFFICIAL_EFFECTIVE_BATCH:
        emit(
            run_id,
            "train.batch",
            f"effective batch {micro_batch}x{accumulation}={micro_batch * accumulation} "
            f"!= {OFFICIAL_EFFECTIVE_BATCH} (protocol)",
            level="error",
        )
        writer.fail(
            f"T9 requires an effective batch of {OFFICIAL_EFFECTIVE_BATCH}; got "
            f"{micro_batch} x {accumulation}"
        )
        return 1
    lr = _float_env("LIM_WORKER_LR", 1e-3)
    workers = _int_env("LIM_WORKER_NUM_WORKERS", 4)
    limit = _int_env("LIM_WORKER_LIMIT_BATCHES", 0) or None
    # The protocol budget is 25,000 optimizer steps x 8 = 200,000 tiles (design
    # §5.1), which is not a whole number of epochs; LIM_WORKER_TOTAL_STEPS is the
    # authoritative stop condition, LIM_WORKER_EPOCHS only bounds the loop.
    total_steps = _int_env("LIM_WORKER_TOTAL_STEPS", 0)
    eval_limit = _int_env("LIM_WORKER_EVAL_IMAGES", 0)
    norm_samples = _int_env("LIM_WORKER_NORM_SAMPLES", 64)
    fpn_dim = _int_env("LIM_WORKER_FPN_DIM", 256)

    data_root = Path(request.data_root)
    manifest = Path(request.manifest_path)
    train_rows = load_rows(manifest, "train")
    val_rows = load_rows(manifest, "validation")
    test_rows = load_rows(manifest, "test")
    if not train_rows or not test_rows:
        emit(run_id, "train.manifest", "manifest has no train/test tiles", level="error")
        writer.fail("manifest has no train/test tiles; run rs-eval build-manifest --task T9")
        return 1
    if not freeze_backbone and not val_rows:
        # A fine-tune selects its checkpoint on a held-out image split; without
        # one, "best epoch" would be the last epoch and the run would pretend to
        # have selected something (design §4.1).
        emit(
            run_id, "train.manifest", "no validation split for checkpoint selection", level="error"
        )
        writer.fail(
            "upernet_finetune needs a validation split: run rs-eval holdout-manifest "
            "--fraction 0.2 --seed 17 --split validation"
        )
        return 1
    crop, stride = tile_geometry(train_rows)
    bands_seen = band_counts(train_rows + test_rows, data_root)

    mean, std = _norm_statistics(train_rows, data_root, crop, norm_samples, request.seed)
    means = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    stds = torch.tensor(std, device=device).view(1, 3, 1, 1)

    encoder = mit_huge(in_chans=3, drop_path_rate=0.1 if not freeze_backbone else 0.0)
    if random_init_requested():
        load_stats = {"loaded": 0, "missing": 0, "unexpected": 0, "dropped": 0}
    else:
        load_stats = load_encoder(checkpoint, encoder)
        if load_stats["missing"] or load_stats["unexpected"]:
            emit(
                run_id,
                "train.checkpoint",
                f"encoder load mismatch: {load_stats}",
                level="error",
            )
            writer.fail(f"LIM encoder did not load cleanly: {load_stats}")
            return 1
    encoder.use_checkpoint = bool(_int_env("LIM_WORKER_CHECKPOINT", 0))
    model = LimSegmentation(encoder, num_classes, fpn_dim=fpn_dim).to(device)
    if freeze_backbone:
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
    head_params = count_parameters(model.head)
    writer.update_training(
        head_id="upernet_gn",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="isaid_rgb_train_stats")
    writer.update(
        tags={
            "head_parameters": str(head_params),
            "head_id": "upernet_gn",
            "norm_layer": "GroupNorm",
            "backbone_frozen": str(freeze_backbone),
            "encoder_params": str(count_parameters(model.encoder)),
            "pretrained_epoch": "150",
            "encoder_keys_loaded": str(load_stats["loaded"]),
            "encoder_keys_dropped": str(load_stats["dropped"]),
            "normalization": (
                f"train-tile mean {mean.round(4).tolist()} std {std.round(4).tolist()}"
            ),
            "normalization_samples": str(norm_samples),
            "learning_rate": f"{lr:g}",
            "micro_batch": str(micro_batch),
            "grad_accum": str(accumulation),
            "effective_batch": str(micro_batch * accumulation),
            "train_crop_px": str(crop),
            "test_stride_px": str(stride),
            "test_protocol": "whole-image sliding window, logits averaged, padding dropped",
            "label_mapping": "16-colour iSAID palette -> category ids, pad = 255",
            "image_bands": ",".join(f"{k}:{v}" for k, v in sorted(bands_seen.items())),
            "grayscale_policy": "1-band and 2-band source PNGs replicated to 3 channels",
            "activation_checkpointing": str(encoder.use_checkpoint),
            "device": device,
            "gsd_source": "declared",
            "gsd_m": "1.0",
        }
    )
    emit(
        run_id,
        "train.prepare",
        f"LIM MiT-Huge ({'frozen' if freeze_backbone else 'fine-tuned'}): "
        f"{len(train_rows)} train tiles, {len(val_rows)} validation tiles, "
        f"{len(test_rows)} test tiles, crop {crop} stride {stride}, "
        f"head {head_params} params ({fpn_dim} fpn), {load_stats['loaded']} encoder keys",
    )

    train_ds = IsaidTiles(train_rows, data_root, crop, mean, std)
    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
        persistent_workers=workers > 0,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)

    eval_stats = {"seconds": 0.0, "images": 0, "tiles": 0}

    def score(rows: list[dict], which: str) -> tuple[dict, dict[str, list[list[int]]]]:
        started_eval = time.time()
        collector = GroupConfusion(num_classes)
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(row["group_id"], []).append(row)
        keys = sorted(groups)
        if eval_limit:
            keys = keys[:eval_limit]
        model.eval()
        with torch.inference_mode():
            for key in keys:
                group = groups[key]
                label_path = data_root / group[0]["label_path"]
                image_path = data_root / group[0]["image_paths"][0]
                label = read_label(label_path)

                tile_calls = 0

                def predict(
                    window: tuple[int, int, int, int],
                    image_path: Path = image_path,
                ) -> np.ndarray:
                    nonlocal tile_calls
                    tile_calls += 1
                    pixels = read_rgb(image_path, window).astype(np.float32)
                    tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
                    # Add the batch axis first: means/stds are [1, C, 1, 1], so
                    # normalising a 3-D tensor would broadcast it up to 4-D.
                    tensor = (tensor[None].to(device) - means) / stds
                    logits = model(tensor, (window[3], window[2]))
                    return logits[0].float().cpu().numpy()

                matrix = evaluate_image(
                    label,
                    num_classes=num_classes,
                    crop=crop,
                    stride=stride,
                    predict=predict,
                    ignore_index=None,
                )
                collector.add_counts(key, matrix.counts)
                eval_stats["tiles"] += tile_calls
        eval_stats["seconds"] += time.time() - started_eval
        eval_stats["images"] += len(keys)
        return multiclass_metrics(collector.combined()), collector.to_json()

    select = not freeze_backbone
    best = BestCheckpoint()
    val_every = validation_every()
    started = time.time()
    train_seconds = 0.0
    train_tiles = 0
    steps_total = 0
    completed_epochs = 0
    budget_hit = False
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        epoch_started = time.time()
        total_loss, steps, window_valid = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            labels = batch["label"].to(device)
            pixels = batch["image"].to(device)
            valid = int((labels != IGNORE_INDEX).sum())
            if valid == 0:
                continue
            logits = model(pixels, tuple(labels.shape[-2:]))
            loss = criterion(logits, labels)
            loss.backward()
            total_loss += float(loss.detach()) / valid
            window_valid += valid
            steps += 1
            steps_total += 1
            if steps % accumulation == 0:
                scale_accumulated_gradients(parameters, window_valid)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                window_valid = 0
            if limit and steps >= limit:
                break
            if total_steps and steps_total >= total_steps:
                budget_hit = True
                break
        if window_valid:
            scale_accumulated_gradients(parameters, window_valid)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        epoch_seconds = time.time() - epoch_started
        train_seconds += epoch_seconds
        train_tiles += steps
        if not budget_hit:
            completed_epochs = epoch + 1
        mean_loss = total_loss / max(steps // accumulation, 1)
        writer.add_curve("loss", "train", epoch, mean_loss)
        emit(
            run_id,
            "train.epoch",
            f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}",
            progress=(epoch + 1) / epochs,
        )
        if budget_hit:
            break
        if select and ((epoch + 1) % val_every == 0 or epoch + 1 == epochs):
            validation, _ = score(val_rows, "validation")
            writer.add_curve("mean_iou", "validation", epoch, validation["mean_iou"])
            best.offer(validation["mean_iou"], epoch + 1, model)
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
    eval_images = len({row["group_id"] for row in test_rows})
    if eval_limit:
        eval_images = min(eval_images, eval_limit)
    emit(run_id, "train.eval", f"evaluating {eval_images} test images (whole-image sliding window)")
    metrics, group_json = score(test_rows, "test")
    writer.add_metrics(
        to_metric_records(metrics, primary_name=metric_name, direction="maximize", split="test")
    )
    writer.add_confusion(group_json, num_classes)
    writer.update_training(
        epochs_completed=completed_epochs, wall_time_seconds=wall, peak_gpu_memory_mb=peak_mb
    )
    writer.update(
        tags={
            "train_tiles": str(train_tiles),
            "train_steps": str(steps_total),
            "total_steps_budget": str(total_steps) if total_steps else "epochs",
            "train_seconds": f"{train_seconds:.1f}",
            "train_tiles_per_second": (
                f"{train_tiles / train_seconds:.2f}" if train_seconds else "0"
            ),
            "eval_seconds": f"{eval_stats['seconds']:.1f}",
            "eval_images": str(eval_stats["images"]),
            "eval_tiles": str(eval_stats["tiles"]),
            "eval_tiles_per_second": (
                f"{eval_stats['tiles'] / eval_stats['seconds']:.2f}"
                if eval_stats["seconds"]
                else "0"
            ),
            "eval_seconds_per_image": (
                f"{eval_stats['seconds'] / eval_stats['images']:.2f}"
                if eval_stats["images"]
                else "0"
            ),
        }
    )
    summary = (
        f"mean_iou={metrics['mean_iou']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"({completed_epochs} epochs / {steps_total} steps, head {head_params} params, "
        f"{'frozen' if freeze_backbone else 'fine-tuned'}, {wall / 60:.1f} min)"
    )
    if limit:
        summary += " [SMOKE]"
    elif stopped_short_of_budget(
        task_id,
        epochs_completed=completed_epochs,
        steps_total=steps_total,
        total_steps=total_steps or None,
    ):
        summary += " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
