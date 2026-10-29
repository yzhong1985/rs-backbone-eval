"""T9 (iSAID) for the Prithvi worker: RGB adapter + Prithvi v2 + shared UPerNet.

Prithvi is the reference backbone of the Prithvi suite and, on iSAID, an **away**
model: it was pretrained on HLS at 30 m with a /16 patch, so a 1024 aerial tile
becomes a 64x64 token grid and every object is resolved by very few tokens.

What the adapter is and is not: Prithvi wants six HLS bands. The 1x1 conv below
maps the three real RGB bands onto those six slots; the two missing bands are
**synthesized by a trainable adapter**, never claimed as observations. The run
records `input_adapter=trainable_1x1_3to6` so a reader cannot mistake it for a
multispectral input.

The feature pyramid is the one honest option for a single-scale ViT: four depths
of the encoder (1/4, 2/4, 3/4, 4/4), all on the same /16 grid. UPerNet fuses
them and upsamples the logits to the tile; the grid itself stays coarse, which is
part of what makes this an away result and is recorded in the run tags.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from rs_backbone_eval.checkpoint import TrainingCheckpoint
from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.data.isaid_io import band_counts, read_label, read_rgb
from rs_backbone_eval.data.isaid_tiles import (
    IGNORE_INDEX,
    IsaidTiles,
    load_rows,
    norm_statistics,
    tile_geometry,
)
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

OFFICIAL_EFFECTIVE_BATCH = 8
# Prithvi's native six HLS bands; the adapter is what supplies four of the six
# slots plus the two bands iSAID does not have.
ENCODER_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
ENCODER_DIM = 1024
PATCH = 16
# Depth indices (0-based) used as the four pyramid levels: 1/4, 2/4, 3/4, 4/4 of
# the encoder. All are /16 grids; the head fuses them and upsamples the logits.
LEVEL_LAYERS = (5, 11, 17, 23)


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


def fixed_band_mapping() -> torch.Tensor:
    """6x3 matrix placing R, G, B into Prithvi's HLS slots, zeros elsewhere.

    Prithvi expects six HLS bands; iSAID has three. A *learned* adapter would
    need the whole frozen encoder's graph alive (all 24 layers kept for backward),
    which measured 0.85 tiles/s and turns a representation probe into a partial
    fine-tune. The fixed mapping keeps B2U a probe of Prithvi's representation and
    is 3x cheaper; the two missing bands are zero-filled and the run tags say so.
    """
    mapping = torch.zeros(len(ENCODER_BANDS), 3)
    for slot, band in enumerate(ENCODER_BANDS):
        if band == "RED":
            mapping[slot, 0] = 1.0
        elif band == "GREEN":
            mapping[slot, 1] = 1.0
        elif band == "BLUE":
            mapping[slot, 2] = 1.0
    return mapping


class PrithviSegmentation(nn.Module):
    """Fixed 3->6 band mapping + Prithvi v2 encoder + shared UPerNet head."""

    def __init__(
        self,
        encoder,
        num_classes: int,
        fpn_dim: int = 256,
        freeze_backbone: bool = True,
        train_adapter: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.freeze_backbone = freeze_backbone
        self.train_adapter = train_adapter
        self.adapter = nn.Conv2d(3, len(ENCODER_BANDS), kernel_size=1)
        with torch.no_grad():
            self.adapter.weight.copy_(fixed_band_mapping()[:, :, None, None])
            self.adapter.bias.zero_()
        for parameter in self.adapter.parameters():
            parameter.requires_grad = train_adapter
        self.head = UPerNetHead(
            [ENCODER_DIM] * len(LEVEL_LAYERS), num_classes=num_classes, fpn_dim=fpn_dim
        )
        if freeze_backbone:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.encoder.eval()
        return self

    def forward(self, pixels: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        adapted = self.adapter(pixels)
        # A learned adapter upstream of a frozen encoder needs the encoder's graph
        # kept alive; a fixed mapping does not, so the probe runs under no_grad and
        # is ~3x faster (measured on the second host, review 2026-10-29).
        if self.freeze_backbone and not self.train_adapter:
            with torch.no_grad():
                layers = self.encoder(adapted)
        else:
            layers = self.encoder(adapted)
        grid = pixels.shape[-1] // PATCH
        features = []
        for index in LEVEL_LAYERS:
            tokens = layers[index][:, 1:, :]  # drop the CLS token
            features.append(
                tokens.transpose(1, 2).reshape(tokens.shape[0], ENCODER_DIM, grid, grid)
            )
        return self.head(features, output_size=output_size)


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
    epochs = _int_env("PRITHVI_WORKER_EPOCHS", 50)
    micro_batch = _int_env("PRITHVI_WORKER_MICRO_BATCH", 1)
    accumulation = _int_env("PRITHVI_WORKER_GRAD_ACCUM", OFFICIAL_EFFECTIVE_BATCH)
    if micro_batch * accumulation != OFFICIAL_EFFECTIVE_BATCH:
        emit(
            run_id,
            "train.batch",
            f"effective batch {micro_batch}x{accumulation} != {OFFICIAL_EFFECTIVE_BATCH}",
            level="error",
        )
        writer.fail(
            f"T9 requires an effective batch of {OFFICIAL_EFFECTIVE_BATCH}; got "
            f"{micro_batch} x {accumulation}"
        )
        return 1
    lr = _float_env("PRITHVI_WORKER_LR", 1e-3)
    workers = _int_env("PRITHVI_WORKER_NUM_WORKERS", 4)
    limit = _int_env("PRITHVI_WORKER_LIMIT_BATCHES", 0) or None
    total_steps = _int_env("PRITHVI_WORKER_TOTAL_STEPS", 0)
    eval_limit = _int_env("PRITHVI_WORKER_EVAL_IMAGES", 0)
    norm_samples = _int_env("PRITHVI_WORKER_NORM_SAMPLES", 64)
    fpn_dim = _int_env("PRITHVI_WORKER_FPN_DIM", 256)

    data_root = Path(request.data_root)
    manifest = Path(request.manifest_path)
    train_rows = load_rows(manifest, "train")
    val_rows = load_rows(manifest, "validation")
    test_rows = load_rows(manifest, "test")
    if not train_rows or not test_rows:
        writer.fail("manifest has no train/test tiles; run rs-eval build-manifest --task T9")
        return 1
    if not freeze_backbone and not val_rows:
        writer.fail(
            "upernet_finetune needs a validation split: run rs-eval holdout-manifest "
            "--fraction 0.2 --seed 17 --split validation"
        )
        return 1
    crop, stride = tile_geometry(train_rows)
    bands_seen = band_counts(train_rows + test_rows, data_root)

    mean, std = norm_statistics(train_rows, data_root, norm_samples, request.seed)
    means = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    stds = torch.tensor(std, device=device).view(1, 3, 1, 1)

    from terratorch.registry import BACKBONE_REGISTRY

    encoder = BACKBONE_REGISTRY.build(
        "prithvi_eo_v2_300_tl",
        pretrained=not random_init_requested(),
        ckpt_path=None if random_init_requested() else str(checkpoint),
        bands=ENCODER_BANDS,
    )
    train_adapter = os.environ.get("PRITHVI_WORKER_TRAIN_ADAPTER", "0") not in ("", "0", "false")
    model = PrithviSegmentation(
        encoder,
        num_classes,
        fpn_dim=fpn_dim,
        freeze_backbone=freeze_backbone,
        train_adapter=train_adapter,
    ).to(device)
    head_params = count_parameters(model.head)
    writer.update_training(
        head_id="upernet_gn",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="isaid_rgb_train_stats")
    writer.update(
        tags={
            "head_id": "upernet_gn",
            "head_parameters": str(head_params),
            "norm_layer": "GroupNorm",
            "backbone_frozen": str(freeze_backbone),
            "encoder_params": str(count_parameters(model.encoder)),
            "input_adapter": ("trainable_1x1_3to6" if train_adapter else "fixed_3to6_band_mapping"),
            "input_adapter_parameters": str(count_parameters(model.adapter, trainable_only=True)),
            # The input is normalised with iSAID train statistics first, so a
            # zero in the adapter output is the *mean* of that band, not a
            # reflectance of zero. Say what is true: the two bands carry no
            # observation, they sit at the per-band mean.
            "missing_bands": (
                "NIR_NARROW,SWIR_1,SWIR_2 carry no observation; the fixed mapping "
                "leaves them at the normalised mean (0 after standardisation)"
                if not train_adapter
                else "NIR_NARROW,SWIR_1,SWIR_2 synthesized by a trainable adapter, not observed"
            ),
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
            "pyramid": f"encoder depths {LEVEL_LAYERS} on the /{PATCH} grid, fused by UPerNet",
            "token_grid": f"{crop // PATCH}x{crop // PATCH} per {crop}px tile",
            "device": device,
            "gsd_source": "declared",
            "gsd_m": f"{_float_env('RS_EVAL_ISAID_GSD', 1.0):g}",
        }
    )
    emit(
        run_id,
        "train.prepare",
        f"Prithvi T9 ({'frozen' if freeze_backbone else 'fine-tuned'}): "
        f"{len(train_rows)} train tiles, {len(val_rows)} validation, {len(test_rows)} test, "
        f"crop {crop} stride {stride}, /{PATCH} grid {crop // PATCH}x{crop // PATCH}, "
        f"head {head_params} params ({fpn_dim} fpn)",
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

    select = not freeze_backbone
    best = BestCheckpoint()
    val_every = validation_every()
    # Epoch-boundary recovery: an 8-21 h run must not restart from zero because the
    # box rebooted. --resume-from points at a latest.pt; TrainingCheckpoint refuses
    # a checkpoint whose settings differ.
    checkpoints = TrainingCheckpoint(
        request,
        writer,
        model,
        optimizer,
        settings={
            "epochs": epochs,
            "crop": crop,
            "lr": lr,
            "total_steps": total_steps,
            "effective_batch": OFFICIAL_EFFECTIVE_BATCH,
            "fpn_dim": fpn_dim,
        },
        best=best if select else None,
    )
    optimizer_steps = checkpoints.optimizer_steps
    completed_epochs = checkpoints.start_epoch

    eval_stats = {"seconds": 0.0, "images": 0, "tiles": 0}

    def score(rows: list[dict]) -> tuple[dict, dict[str, list[list[int]]]]:
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
                    window: tuple[int, int, int, int], image_path: Path = image_path
                ) -> np.ndarray:
                    nonlocal tile_calls
                    tile_calls += 1
                    pixels = read_rgb(image_path, window).astype(np.float32)
                    tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
                    tensor = (tensor[None].to(device) - means) / stds
                    logits = model(tensor, (crop, crop))
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

    train_seconds = 0.0
    train_tiles = 0
    # optimizer_steps / completed_epochs come from the checkpoint (0 when fresh)
    budget_hit = False
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(checkpoints.start_epoch, epochs):
        if total_steps and optimizer_steps >= total_steps:
            budget_hit = True
            break
        model.train()
        epoch_started = time.time()
        total_loss, steps, window_valid = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            pixels = batch["image"].to(device)
            labels = batch["label"].to(device)
            valid = int((labels != IGNORE_INDEX).sum())
            if valid == 0:
                continue
            logits = model(pixels, tuple(labels.shape[-2:]))
            loss = criterion(logits, labels)
            loss.backward()
            total_loss += float(loss.detach()) / valid
            window_valid += valid
            steps += 1
            train_tiles += 1
            if steps % accumulation == 0:
                scale_accumulated_gradients(parameters, window_valid)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                window_valid = 0
                optimizer_steps += 1
                if total_steps and optimizer_steps >= total_steps:
                    budget_hit = True
                    break
            if limit and steps >= limit:
                break
        if window_valid:
            scale_accumulated_gradients(parameters, window_valid)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        epoch_seconds = time.time() - epoch_started
        train_seconds += epoch_seconds
        if not budget_hit:
            completed_epochs = epoch + 1
        mean_loss = total_loss / max(steps // accumulation, 1)
        writer.add_curve("loss", "train", epoch, mean_loss)
        emit(
            run_id,
            "train.epoch",
            f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f} steps={optimizer_steps}",
            progress=(epoch + 1) / epochs,
        )
        if budget_hit:
            break
        if budget_hit:
            checkpoints.save(completed_epochs, optimizer_steps)
            break
        if select and ((epoch + 1) % val_every == 0 or epoch + 1 == epochs):
            validation, _ = score(val_rows)
            writer.add_curve("mean_iou", "validation", epoch, validation["mean_iou"])
            best.offer(validation["mean_iou"], epoch + 1, model)
    wall = checkpoints.elapsed
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
    emit(run_id, "train.eval", "evaluating the test split (whole-image sliding window)")
    metrics, group_json = score(test_rows)
    writer.add_metrics(
        to_metric_records(metrics, primary_name=metric_name, direction="maximize", split="test")
    )
    writer.add_confusion(group_json, num_classes)
    writer.update_training(
        epochs_completed=completed_epochs,
        optimizer_steps=optimizer_steps,
        wall_time_seconds=wall,
        peak_gpu_memory_mb=peak_mb,
    )
    writer.update(
        tags={
            "train_tiles": str(train_tiles),
            "train_steps": str(train_tiles),
            "optimizer_steps": str(optimizer_steps),
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
        }
    )
    summary = (
        f"mean_iou={metrics['mean_iou']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"({completed_epochs} epochs / {optimizer_steps} steps, head {head_params} params, "
        f"{'frozen' if freeze_backbone else 'fine-tuned'}, {wall / 60:.1f} min)"
    )
    if limit:
        summary += " [SMOKE]"
    elif stopped_short_of_budget(
        task_id,
        epochs_completed=completed_epochs,
        steps_total=optimizer_steps,
        total_steps=total_steps or None,
    ):
        summary += " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
