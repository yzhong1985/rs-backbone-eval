"""T8 (Five-Billion-Pixels / GF-2) land cover for the Clay worker.

Out-of-domain by construction: 4 m, four bands (B/G/R/NIR), no SWIR, versus
Clay's optical 10-30 m pretraining. The scenes are ~6900x7300 px; chips follow
the shared protocol in ``rs_backbone_eval.gf2_chips`` (re-drawn training chips,
fixed validation draw, full test grid).

Clay can be handed the four band wavelengths directly, so no adapter is needed;
Prithvi and AnySat need one and are not part of this module.
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
from rs_backbone_eval.heads import count_parameters
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.worker_runtime import RunWriter

from .clay_probe import add_upstream_to_path, prepare_encoder
from .t1 import CLAY_LARGE, ClayProbe

# The 16-bit product is used because the 8-bit NIR/R/G/B product is RGBA with a
# constant alpha band and carries no usable NIR. Order follows the readme
# (B, G, R, NIR); it is recorded as readme-sourced rather than independently
# verified, so check a scene visually before quoting final numbers.
BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD"]
# Central wavelengths of the GF-2 bands in micrometres (NIR is broad).
WAVELENGTHS_UM = {"BLUE": 0.48, "GREEN": 0.56, "RED": 0.66, "NIR_BROAD": 0.83}
NUM_CLASSES = 25
IGNORE_INDEX = -1
OFFICIAL_BATCH_SIZE = 2
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
    """Chips of one split; ``CLAY_WORKER_EVAL_LIMIT`` caps the scenes (smoke runs)."""
    rows = load_split(manifest_path, split)
    rows = rows[: _int_env("CLAY_WORKER_EVAL_LIMIT", 0) or len(rows)]
    return Gf2Chips(scene_root(), rows, bands=len(BANDS))


def train_t8(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    checkpoint: Path,
    freeze_backbone: bool = True,
) -> int:
    run_id = request.run_id
    add_upstream_to_path()
    from claymodel.finetune.segment.factory import SegmentEncoder  # upstream checkout

    device = os.environ.get("CLAY_WORKER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    epochs = _int_env("CLAY_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("CLAY_WORKER_BATCH_SIZE", OFFICIAL_BATCH_SIZE)
    lr = _float_env("CLAY_WORKER_LR", 1e-3)
    workers = _int_env("CLAY_WORKER_NUM_WORKERS", 4)
    limit = _int_env("CLAY_WORKER_LIMIT_BATCHES", 0) or None

    manifest = Path(request.manifest_path)
    train_ds = split_chips(manifest, "train")
    val_ds = split_chips(manifest, "validation")
    val_ds.set_chips(val_ds.draw(VALIDATION_CHIPS_PER_SCENE, VALIDATION_DRAW_SEED))
    eval_ds = split_chips(manifest, "test")
    eval_ds.set_chips(eval_ds.full_grid())
    norm_chips = [
        chip
        for chip in train_ds.draw(4, VALIDATION_DRAW_SEED)
        if chip.scene < _int_env("CLAY_WORKER_NORM_SCENES", 20)
    ]
    mean_array, std_array = channel_statistics(train_ds, norm_chips)
    mean = [round(float(v), 4) for v in mean_array]
    std = [round(float(v), 4) for v in std_array]
    means = torch.tensor(mean, device=device).view(1, -1, 1, 1)
    stds = torch.tensor(std, device=device).view(1, -1, 1, 1)
    waves = torch.tensor([WAVELENGTHS_UM[b] for b in BANDS], dtype=torch.float32, device=device)

    encoder = SegmentEncoder(
        **CLAY_LARGE, ckpt_path=None if random_init_requested() else str(checkpoint)
    ).to(device)
    prepare_encoder(encoder, freeze_backbone)
    model = ClayProbe(encoder, NUM_CLASSES, GSD_M, "gf-2", freeze_backbone).to(device)
    head_params = count_parameters(model.head)
    emit(
        run_id,
        "train.prepare",
        f"Clay T8 (freeze={freeze_backbone}): {len(train_ds.rows)} train scenes "
        f"({TRAIN_CHIPS_PER_SCENE} chips each per epoch), {len(val_ds)} validation chips, "
        f"{len(eval_ds)} test chips (full grid of {len(eval_ds.rows)} scenes), "
        f"{len(BANDS)} bands, {NUM_CLASSES} classes, head {head_params}",
    )
    writer.update_training(
        head_id="lite_spatial_head",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="gf2_train_stats_four_band")
    writer.update(
        tags={
            "head_parameters": str(head_params),
            "backbone_frozen": str(freeze_backbone),
            "bands": ",".join(BANDS),
            "wavelengths_um": ",".join(f"{WAVELENGTHS_UM[b]}" for b in BANDS),
            "normalization": f"train-scene mean {mean} std {std}",
            "label_mapping": "0 -> ignore, 1..24 kept",
            "cross_domain_note": "4 m GF-2 with no SWIR, far from the 10-30 m pretraining",
            "band_order_source": "readme (16-bit B/G/R/NIR), not independently verified",
            "source_product": "Image_16bit_BGRNir",
            "learning_rate": f"{lr:g}",
            "chip_protocol": (
                f"{CHIP}px; train {TRAIN_CHIPS_PER_SCENE}/scene/epoch redrawn, "
                f"validation {VALIDATION_CHIPS_PER_SCENE}/scene fixed, test full grid"
            ),
            "micro_batch": str(batch_size),
            "device": device,
        }
    )

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )

    def score(dataset) -> tuple[dict, dict[str, list[list[int]]]]:
        collector = GroupedSegmentation(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
        model.eval()
        with torch.inference_mode():
            for batch in DataLoader(dataset, batch_size=8, shuffle=False, num_workers=workers):
                pixels = (batch["image"].to(device) - means) / stds
                logits = model(pixels, waves)
                collector.update(
                    list(batch["group"]),
                    batch["label"].numpy(),
                    logits.argmax(dim=1).cpu().numpy(),
                )
        return multiclass_metrics(collector.global_matrix), collector.group_confusion()

    select = not freeze_backbone
    best = BestCheckpoint()
    val_every = validation_every()
    started = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(epochs):
        model.train()
        train_ds.set_chips(train_ds.draw(TRAIN_CHIPS_PER_SCENE, epoch_seed(request.seed, epoch)))
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
        total_loss, steps = 0.0, 0
        for batch in loader:
            pixels = (batch["image"].to(device) - means) / stds
            labels = batch["label"].to(device)
            if int((labels != IGNORE_INDEX).sum()) == 0:
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
        if select and ((epoch + 1) % val_every == 0 or epoch + 1 == epochs):
            validation, _ = score(val_ds)
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
    emit(run_id, "train.eval", f"evaluating on {len(eval_ds)} test chips")
    metrics, group_json = score(eval_ds)
    writer.add_metrics(
        to_metric_records(metrics, primary_name="mean_iou", direction="maximize", split="test")
    )
    writer.add_confusion(group_json, NUM_CLASSES)
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=wall, peak_gpu_memory_mb=peak_mb
    )
    summary = (
        f"mean_iou={metrics['mean_iou']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"({epochs} epochs, head {head_params} params, "
        f"{'frozen' if freeze_backbone else 'fine-tuned'}, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]" if limit else " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
