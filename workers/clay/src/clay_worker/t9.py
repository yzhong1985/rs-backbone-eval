"""T9 (iSAID) for the Clay worker: UPerNet over Clay's patch-8 token grid.

Clay is the closest "away" backbone to LIM: three optical channels, and the only
one that can be told a GSD explicitly (declared 1.0 m here, ``platform=naip`` --
Clay's aerial RGB entry point, whose wavelengths 0.48/0.56/0.65 um match iSAID's
RGB). Everything else follows the shared iSAID protocol: manifest tiles for
training (label padded with 255), whole-image sliding windows for evaluation.

Clay emits one token grid at stride 8 rather than a four-level pyramid, so the
UPerNet head gets four levels by average-pooling it (stride 8/16/32/64). That is
a parameter-free choice, recorded in the run tags -- it is part of what makes
this an "away" result, next to the GSD mismatch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

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

from .clay_probe import add_upstream_to_path, prepare_encoder
from .t1 import CLAY_LARGE, _float_env, _int_env

OFFICIAL_EFFECTIVE_BATCH = 8
# Clay's aerial RGB entry point: 3 bands, 1.0 m, wavelengths 0.48/0.56/0.65 um.
PLATFORM = "naip"
GSD_M = 1.0
WAVELENGTHS_UM = (0.48, 0.56, 0.65)


class ClaySegmentation(nn.Module):
    """Clay SegmentEncoder (frozen or fine-tuned) + shared UPerNet head."""

    def __init__(self, encoder, num_classes: int, fpn_dim: int = 256, freeze_backbone: bool = True):
        super().__init__()
        self.encoder = encoder
        self.freeze_backbone = freeze_backbone
        dim = CLAY_LARGE["dim"]
        levels = [dim] * 4
        self.head = UPerNetHead(levels, num_classes=num_classes, fpn_dim=fpn_dim)
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
            "gsd": torch.tensor(GSD_M, device=pixels.device),
            "waves": waves,
            "platform": PLATFORM,
        }
        context = torch.no_grad() if self.freeze_backbone else torch.enable_grad()
        with context:
            tokens = self.encoder(datacube)  # [B, L, D]
        patch = CLAY_LARGE["patch_size"]
        grid_h, grid_w = height // patch, width // patch
        features = tokens.transpose(1, 2).reshape(batch, CLAY_LARGE["dim"], grid_h, grid_w)
        levels = [features]
        for _ in range(3):
            levels.append(F.avg_pool2d(levels[-1], 2))
        return self.head(levels, output_size=(height, width))


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
    add_upstream_to_path()
    from claymodel.finetune.segment.factory import SegmentEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = _int_env("CLAY_WORKER_EPOCHS", 50)
    micro_batch = _int_env("CLAY_WORKER_MICRO_BATCH", 1)
    accumulation = _int_env("CLAY_WORKER_GRAD_ACCUM", OFFICIAL_EFFECTIVE_BATCH)
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
    lr = _float_env("CLAY_WORKER_LR", 1e-3)
    workers = _int_env("CLAY_WORKER_NUM_WORKERS", 4)
    limit = _int_env("CLAY_WORKER_LIMIT_BATCHES", 0) or None
    total_steps = _int_env("CLAY_WORKER_TOTAL_STEPS", 0)
    eval_limit = _int_env("CLAY_WORKER_EVAL_IMAGES", 0)
    norm_samples = _int_env("CLAY_WORKER_NORM_SAMPLES", 64)
    fpn_dim = _int_env("CLAY_WORKER_FPN_DIM", 256)

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
        emit(run_id, "train.manifest", "no validation split", level="error")
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
    waves = torch.tensor(WAVELENGTHS_UM, dtype=torch.float32, device=device)

    encoder = SegmentEncoder(
        **CLAY_LARGE, ckpt_path=None if random_init_requested() else str(checkpoint)
    ).to(device)
    prepare_encoder(encoder, freeze_backbone)
    model = ClaySegmentation(encoder, num_classes, fpn_dim=fpn_dim, freeze_backbone=freeze_backbone)
    model = model.to(device)
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
            "device": device,
            "gsd_source": "declared",
            "gsd_m": f"{GSD_M:g}",
            "clay_platform": PLATFORM,
            "wavelengths_um": ",".join(f"{value:g}" for value in WAVELENGTHS_UM),
            "pyramid": "stride 8 token grid, avg-pooled to 8/16/32/64",
        }
    )
    emit(
        run_id,
        "train.prepare",
        f"Clay T9 ({'frozen' if freeze_backbone else 'fine-tuned'}): "
        f"{len(train_rows)} train tiles, {len(val_rows)} validation, {len(test_rows)} test, "
        f"crop {crop} stride {stride}, platform {PLATFORM} @ {GSD_M:g} m, "
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
                    logits = model(tensor, waves)
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
    optimizer_steps = 0
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
            pixels = batch["image"].to(device)
            labels = batch["label"].to(device)
            valid = int((labels != IGNORE_INDEX).sum())
            if valid == 0:
                continue
            logits = model(pixels, waves)
            loss = criterion(logits, labels)
            loss.backward()
            total_loss += float(loss.detach()) / valid
            window_valid += valid
            steps += 1
            if steps % accumulation == 0:
                scale_accumulated_gradients(parameters, window_valid)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                window_valid = 0
                # The protocol budget counts *optimizer* steps (25,000 x 8
                # samples); one accumulator window is one step, not one tile.
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
            validation, _ = score(val_rows)
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
