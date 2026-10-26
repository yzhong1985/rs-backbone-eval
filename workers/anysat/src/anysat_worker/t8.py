"""T8 (Five-Billion-Pixels / GF-2) land cover for the AnySat worker.

Four GF-2 bands (B/G/R/NIR) must reach AnySat's fixed ten-band ``s2`` entrance,
so the shared explicit spectral adapter synthesizes the six missing bands; zero
padding stays forbidden. GF-2 is 4 m with no SWIR while AnySat's S2 entrance
was pretrained on 10 m Sentinel-2, so the run is cross-domain by construction
and the adapter parameters are reported separately from the task head.

Scenes are ~6900x7300 px; chips follow the shared protocol in
``rs_backbone_eval.gf2_chips``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.groups import GroupedSegmentation
from rs_backbone_eval.evaluation.segmentation import multiclass_metrics
from rs_backbone_eval.gf2_chips import (
    CHIP,
    TRAIN_CHIPS_PER_SCENE,
    VALIDATION_CHIPS_PER_SCENE,
    VALIDATION_DRAW_SEED,
    Gf2Chips,
    channel_statistics,
    epoch_seed,
    load_split,
)
from rs_backbone_eval.heads import LiteSpatialHead, count_parameters
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.timeutil import day_of_year
from rs_backbone_eval.training import scale_accumulated_gradients
from rs_backbone_eval.worker_runtime import RunWriter

from .spectral_adapter import SpectralAdapter
from .t1_adapter import build_backbone

BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD"]
NUM_CLASSES = 25
IGNORE_INDEX = -1
PATCH_SIZE_M = 160
GSD_M = 4.0  # MSS product; 0.8 m is the panchromatic band, not in this release
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


def split_chips(manifest_path: Path, split: str) -> Gf2Chips:
    """Chips of one split; ``ANYSAT_WORKER_EVAL_LIMIT`` caps the scenes (smoke runs)."""
    rows = load_split(manifest_path, split)
    rows = rows[: _int_env("ANYSAT_WORKER_EVAL_LIMIT", 0) or len(rows)]
    return Gf2Chips(scene_root(), rows, bands=len(BANDS))


def placeholder_day_of_year(dataset: Gf2Chips) -> int:
    """AnySat needs a date per image; the release has no acquisition dates.

    The manifest carries one placeholder date for every scene, so the model sees
    a constant day of year (not the year, which the first version passed).
    """
    stamps = {row["timestamps"][0] for row in dataset.rows}
    if len(stamps) != 1:
        raise ValueError(f"expected one placeholder date, found {sorted(stamps)}")
    return day_of_year(next(iter(stamps)))


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


def _evaluate(
    model, dataset, means, stds, device, doy: int, workers: int
) -> tuple[dict[str, float], dict[str, list[list[int]]]]:
    collector = GroupedSegmentation(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=4, shuffle=False, num_workers=workers):
            images = ((batch["image"].to(device) - means) / stds).unsqueeze(1)
            dates = torch.full((images.shape[0], 1), float(doy), device=device)
            logits = model(images, dates)
            collector.update(
                list(batch["group"]),
                batch["label"].numpy(),
                logits.argmax(dim=1).cpu().numpy(),
            )
            del images, dates, logits
    return multiclass_metrics(collector.global_matrix), collector.group_confusion()


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
    grad_accum = max(1, _int_env("ANYSAT_WORKER_GRAD_ACCUM", 1))
    lr = _float_env("ANYSAT_WORKER_LR", 1e-3)
    workers = _int_env("ANYSAT_WORKER_NUM_WORKERS", 4)
    limit = _int_env("ANYSAT_WORKER_LIMIT_BATCHES", 0) or None
    freeze = freeze_backbone
    if os.environ.get("ANYSAT_WORKER_FINETUNE"):
        freeze = os.environ["ANYSAT_WORKER_FINETUNE"] in {"0", "false"}

    manifest = Path(request.manifest_path)
    train_ds = split_chips(manifest, "train")
    val_ds = split_chips(manifest, "validation")
    val_ds.set_chips(val_ds.draw(VALIDATION_CHIPS_PER_SCENE, VALIDATION_DRAW_SEED))
    eval_ds = split_chips(manifest, "test")
    eval_ds.set_chips(eval_ds.full_grid())
    doy = placeholder_day_of_year(train_ds)
    norm_chips = [
        chip
        for chip in train_ds.draw(4, VALIDATION_DRAW_SEED)
        if chip.scene < _int_env("ANYSAT_WORKER_NORM_SCENES", 20)
    ]
    mean, std = channel_statistics(train_ds, norm_chips)
    means = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    stds = torch.tensor(std, device=device).view(1, -1, 1, 1)
    backbone = build_backbone(checkpoint).to(device)
    model = AdapterProbeT8(backbone, task_bands, freeze_backbone=freeze).to(device)
    adapter_params = count_parameters(model.adapter)
    head_params = count_parameters(model.head)
    emit(
        run_id,
        "train.prepare",
        f"AnySat T8 (freeze={freeze}): {len(train_ds.rows)} train scenes "
        f"({TRAIN_CHIPS_PER_SCENE} chips each per epoch), {len(val_ds)} validation chips, "
        f"{len(eval_ds)} test chips (full grid of {len(eval_ds.rows)} scenes), "
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
            "cross_domain_note": "4 m GF-2 with no SWIR into a 10 m S2 entrance",
            "band_order_source": "verified via label-conditional band statistics",
            "learning_rate": f"{lr:g}",
            "chip_protocol": (
                f"{CHIP}px; train {TRAIN_CHIPS_PER_SCENE}/scene/epoch redrawn, "
                f"validation {VALIDATION_CHIPS_PER_SCENE}/scene fixed, test full grid"
            ),
            "dates": f"no acquisition dates; placeholder day of year {doy}",
            "micro_batch": str(batch_size),
            "grad_accum": str(grad_accum),
            "effective_batch": str(batch_size * grad_accum),
            "device": device,
        }
    )

    # Sum + one scaling by the window's valid pixels equals a single
    # mean-reduced batch over the accumulation window (same helper as t2/t3/t4).
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    select = not freeze
    best = BestCheckpoint()
    val_every = validation_every()

    started = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optimizer_steps = 0
    for epoch in range(epochs):
        model.train()
        train_ds.set_chips(train_ds.draw(TRAIN_CHIPS_PER_SCENE, epoch_seed(request.seed, epoch)))
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
        epoch_loss_sum, epoch_valid = 0.0, 0
        steps, pending, window_valid = 0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch in loader:
            images = ((batch["image"].to(device) - means) / stds).unsqueeze(1)
            labels = batch["label"].to(device)
            dates = torch.full((images.shape[0], 1), float(doy), device=device)
            valid = int((labels != IGNORE_INDEX).sum())
            if valid == 0:
                continue
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(images, dates)
            loss_sum = criterion(logits.float(), labels)
            loss_sum.backward()
            epoch_loss_sum += float(loss_sum.detach())
            epoch_valid += valid
            pending += 1
            window_valid += valid
            if pending == grad_accum:
                scale_accumulated_gradients(params, window_valid)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                steps += 1
                pending, window_valid = 0, 0
                if limit and steps >= limit:
                    break
        if pending:  # flush a partial accumulation window at epoch end
            scale_accumulated_gradients(params, window_valid)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            steps += 1
        mean_loss = epoch_loss_sum / epoch_valid if epoch_valid else 0.0
        writer.add_curve("loss", "train", epoch, mean_loss)
        emit(
            run_id,
            "train.epoch",
            f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}",
            progress=(epoch + 1) / epochs,
        )
        if select and ((epoch + 1) % val_every == 0 or epoch + 1 == epochs):
            torch.cuda.empty_cache()
            validation, _ = _evaluate(model, val_ds, means, stds, device, doy, workers)
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
    emit(run_id, "train.eval", f"evaluating on {len(eval_ds)} test chips")
    metrics, group_json = _evaluate(model, eval_ds, means, stds, device, doy, workers)
    writer.add_metrics(
        to_metric_records(metrics, primary_name="mean_iou", direction="maximize", split="test")
    )
    writer.add_confusion(group_json, NUM_CLASSES)
    writer.update_training(
        epochs_completed=epochs,
        optimizer_steps=optimizer_steps,
        wall_time_seconds=wall,
        peak_gpu_memory_mb=peak_mb,
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
