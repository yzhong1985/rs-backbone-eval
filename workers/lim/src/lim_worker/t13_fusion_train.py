"""T13 fusion trainer: residual branches on a frozen FT-B0 LIM and head (plan v3 §3).

Only the branch parameters train. The LIM encoder and UPerNet head come from the
pinned FT-B0 full checkpoint of the same seed and stay frozen in eval mode.
Validation tiles score the model every ``validation_every()`` epochs and select
the epoch. The test split is never read here: the test score is a separate,
explicit ``evaluate`` step.

NIR-only in this step. The SAR branch needs a SAR tile provider in
``t13.score_sheets``, so SAR recipes raise ``NotImplementedError`` for now.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from pathlib import Path

import torch

from rs_backbone_eval.data.manifest import manifest_sha256
from rs_backbone_eval.normalization import normalization_seed
from rs_backbone_eval.schedule import build_scheduler
from rs_backbone_eval.selection import validation_every

from .arms import ArmModel, safe_cross_entropy
from .fusion_branches import FusionBranches
from .mit import mit_huge
from .t13 import (
    CHANNELS,
    EFFECTIVE_BATCH,
    NUM_CLASSES,
    Inputs,
    _int_env,
    load_full_checkpoint,
    run_train_epoch,
    score_sheets,
    train_loader,
)
from .t13_data import IGNORE_INDEX, T13Tiles, band_statistics, load_rows
from .t13_fusion import FUSION_RECIPES, branch_parameters, find_full_dependency, fused_logits

FUSION_FORMAT = "t13-fusion-plugin-v1"
# This file is <repo>/workers/lim/src/lim_worker/t13_fusion_train.py.
REPO_ROOT = Path(__file__).resolve().parents[4]


def train_fusion(
    request,
    writer,
    emit: Callable[..., None],
    *,
    protocol,
    task_spec,
    model_id: str,
) -> int:
    recipe = FUSION_RECIPES.get(protocol.protocol_id)
    if recipe is None:
        raise ValueError(f"T13 fusion does not implement protocol {protocol.protocol_id}")
    if recipe["sar"]:
        raise NotImplementedError("SAR validation needs a SAR tile provider in score_sheets")
    run_id, seed = request.run_id, request.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(request.data_root)
    runs_root = Path(request.runs_root)
    manifest = Path(request.manifest_path)

    epochs = int(protocol.epochs or 0)
    lr = float(protocol.learning_rate or 0)
    if not epochs or not lr:
        raise ValueError(
            f"{protocol.protocol_id}: fusion protocols must declare epochs and learning_rate"
        )
    micro = _int_env("LIM_WORKER_MICRO_BATCH", 2)
    if EFFECTIVE_BATCH % micro:
        raise ValueError(
            f"micro batch {micro} does not divide the effective batch {EFFECTIVE_BATCH}"
        )
    accumulation = EFFECTIVE_BATCH // micro
    limit = _int_env("LIM_WORKER_LIMIT_BATCHES", 0)
    eval_sheets = _int_env("LIM_WORKER_EVAL_IMAGES", 0)
    workers = _int_env("LIM_WORKER_NUM_WORKERS", 6)
    smoke = bool(limit or eval_sheets)

    train_rows = load_rows(manifest, "train")
    val_rows = load_rows(manifest, "validation")
    if not (train_rows and val_rows):
        raise ValueError("T13 fusion needs train and validation tiles")

    # Identity and admission of the baseline; the request's manifest must match it.
    dependency = find_full_dependency(
        runs_root,
        seed=seed,
        manifest_sha=manifest_sha256(manifest),
        model_id=model_id,
        batch=EFFECTIVE_BATCH,
        repo=REPO_ROOT,
    )

    torch.manual_seed(seed)
    band_mean, band_std = band_statistics(train_rows, data_root, 256, normalization_seed())
    encoder = mit_huge(in_chans=3)
    model = ArmModel("B0", CHANNELS, NUM_CLASSES)
    load_full_checkpoint(dependency, encoder, model)
    encoder = encoder.to(device).eval()
    model = model.to(device).eval()
    for param in list(encoder.parameters()) + list(model.parameters()):
        param.requires_grad_(False)
    if device == "cuda":
        encoder = encoder.half()
    inputs = Inputs(encoder, band_mean, band_std, device, finetune=False)

    branches = FusionBranches(CHANNELS, use_nir=recipe["nir"], use_sar=False).to(device).train()
    trainable = branch_parameters(branches)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    dataset = T13Tiles(train_rows, data_root, with_aef=False, augment=True, seed=seed)
    loader = train_loader(dataset, micro, workers, seed)
    steps_per_epoch = math.ceil((limit or len(loader)) / accumulation)
    scheduler = (
        build_scheduler(optimizer, epochs * steps_per_epoch, steps_per_epoch)
        if protocol.lr_schedule == "warmup1-cosine"
        else None
    )

    def logits_for(image_u8: torch.Tensor) -> torch.Tensor:
        image = image_u8.to(device)
        feats = inputs.features(image)  # frozen encoder, no grad
        nir = None
        nir_valid = None
        if recipe["nir"]:
            nir = inputs.plugin_image(image)[:, 3:4]
            nir_valid = torch.ones_like(nir)
        logits = fused_logits(
            model.head, feats, branches, tuple(image.shape[-2:]), nir=nir, nir_valid=nir_valid
        )
        return logits.float()

    def micro_loss(batch: dict):
        labels = batch["label"].to(device, non_blocking=True)
        ce = safe_cross_entropy(logits_for(batch["image"]), labels, IGNORE_INDEX)
        aux = ce.new_zeros(())
        return ce, ce, aux

    def evaluate(rows: list[dict], sheet_limit: int) -> dict:
        branches.eval()
        with torch.inference_mode():
            metrics, _ = score_sheets(
                rows,
                data_root,
                lambda image, _codes: logits_for(image),
                with_aef=False,
                batch=micro,
                limit=sheet_limit,
            )
        branches.train()
        return metrics

    val_every = validation_every()
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict | None = None
    optimizer_steps = 0
    started = time.time()
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        branches.train()
        totals, batches, steps = run_train_epoch(
            loader, micro_loss, optimizer, scheduler, accumulation, limit
        )
        optimizer_steps += steps
        writer.add_curve("ce", "train", epoch, totals["ce"] / max(batches, 1))
        if (epoch + 1) % val_every == 0 or epoch + 1 == epochs:
            val = evaluate(val_rows, eval_sheets)
            writer.add_curve("mean_iou", "validation", epoch, val["mean_iou"])
            if val["mean_iou"] > best_score:
                best_score = val["mean_iou"]
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in branches.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no validation pass ran; nothing to select")
    branches.load_state_dict(best_state)
    # The plugin file names the baseline it was trained on, by run ID and checkpoint hash.
    torch.save(
        {
            "format": FUSION_FORMAT,
            "protocol": protocol.protocol_id,
            "baseline_run": dependency.parent.name,
            "baseline_checkpoint": str(dependency),
            "config": {"nir": recipe["nir"], "sar": recipe["sar"], "channels": CHANNELS},
            "band_mean": band_mean.tolist(),
            "band_std": band_std.tolist(),
            "state_dict": best_state,
        },
        writer.run_dir / "fusion_plugin.pt",
    )
    writer.add_artifact("checkpoint", "fusion_plugin.pt", "validation-selected residual branches")
    writer.update(
        tags={
            "selection": "best_validation_mean_iou",
            "selected_epoch": str(best_epoch),
            "validation_selection_score": f"{best_score:.4f}",
            "baseline_run": dependency.parent.name,
            "optimizer_steps": str(optimizer_steps),
            "test_evaluated": "no",
        }
    )
    writer.update_training(
        epochs_completed=epochs,
        optimizer_steps=optimizer_steps,
        wall_time_seconds=time.time() - started,
    )
    summary = (
        f"val_mean_iou={best_score:.4f} ({protocol.protocol_id}, {epochs} epochs, "
        f"best epoch {best_epoch}; test not evaluated)"
    )
    if smoke:
        summary += " [SMOKE]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
