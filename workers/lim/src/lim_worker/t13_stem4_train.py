"""Frozen four-band first layer on FT-B0 (T13 plan §4, frozen version).

Only the NIR convolution of the first layer trains. The LIM encoder, its RGB stem
and the UPerNet head come from the pinned FT-B0 full checkpoint of the same seed
and stay frozen. Gradients still pass through the whole frozen encoder to reach the
first layer, so the cost is close to full fine-tuning.

The test split is never read here. Validation picks the epoch.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F

from rs_backbone_eval.data.manifest import manifest_sha256
from rs_backbone_eval.normalization import normalization_seed
from rs_backbone_eval.schedule import build_scheduler
from rs_backbone_eval.selection import validation_every

from .arms import ArmModel
from .mit import mit_huge
from .stem4 import add_nir_stem
from .t13 import (
    CHANNELS,
    EFFECTIVE_BATCH,
    NUM_CLASSES,
    Inputs,
    _int_env,
    load_full_checkpoint,
    score_sheets,
    train_loader,
)
from .t13_data import IGNORE_INDEX, T13Tiles, band_statistics, load_rows
from .t13_fusion import FT_B0_BASELINES, find_full_dependency
from .t13_fusion_train import REPO_ROOT, _require_finite

STEM4_FORMAT = "t13-stem4-plugin-v1"
STEM4_PROTOCOL = "whu_ft_nir4"
# A change of the NIR weights below this is not training (see t13_fusion_train).
ZERO_UPDATE_TOLERANCE = 1e-6


def _max_change(initial: torch.Tensor, current: torch.Tensor) -> float:
    return float((initial - current).abs().max())


def train_stem4(
    request,
    writer,
    emit: Callable[..., None],
    *,
    protocol,
    task_spec,
    model_id: str,
) -> int:
    if protocol.protocol_id != STEM4_PROTOCOL:
        raise ValueError(f"T13 stem4 does not implement protocol {protocol.protocol_id}")
    run_id, seed = request.run_id, request.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(request.data_root)
    runs_root = Path(request.runs_root)
    manifest = Path(request.manifest_path)

    epochs = int(protocol.epochs or 0)
    lr = float(protocol.learning_rate or 0)
    if not epochs or not lr:
        raise ValueError(f"{protocol.protocol_id}: must declare epochs and learning_rate")
    micro = _int_env("LIM_WORKER_MICRO_BATCH", 2)
    if EFFECTIVE_BATCH % micro:
        raise ValueError(f"micro batch {micro} does not divide the effective batch")
    accumulation = EFFECTIVE_BATCH // micro
    limit = _int_env("LIM_WORKER_LIMIT_BATCHES", 0)
    eval_sheets = _int_env("LIM_WORKER_EVAL_IMAGES", 0)
    workers = _int_env("LIM_WORKER_NUM_WORKERS", 6)
    smoke = bool(limit or eval_sheets)

    train_rows = load_rows(manifest, "train")
    val_rows = load_rows(manifest, "validation")
    if not (train_rows and val_rows):
        raise ValueError("T13 stem4 needs train and validation tiles")

    dependency = find_full_dependency(
        runs_root,
        seed=seed,
        manifest_sha=manifest_sha256(manifest),
        model_id=model_id,
        batch=EFFECTIVE_BATCH,
        repo=REPO_ROOT,
    )
    writer.attach_checkpoint(dependency)

    torch.manual_seed(seed)
    band_mean, band_std = band_statistics(train_rows, data_root, 256, normalization_seed())
    encoder = mit_huge(in_chans=3)
    model = ArmModel("B0", CHANNELS, NUM_CLASSES)
    load_full_checkpoint(dependency, encoder, model)
    stem = add_nir_stem(encoder)  # zero-initialised NIR convolution
    encoder = encoder.to(device).eval()
    model = model.to(device).eval()
    for param in list(encoder.parameters()) + list(model.parameters()):
        param.requires_grad_(False)
    stem.nir.weight.requires_grad_(True)
    inputs = Inputs(encoder, band_mean, band_std, device, finetune=True)
    trainable = [stem.nir.weight]
    initial = stem.nir.weight.detach().cpu().clone()

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    dataset = T13Tiles(train_rows, data_root, with_aef=False, augment=True, seed=seed)
    loader = train_loader(dataset, micro, workers, seed)
    full_groups_per_epoch = min(limit or len(loader), len(loader)) // accumulation
    if full_groups_per_epoch < 1:
        raise ValueError("an epoch holds no complete accumulation group")
    steps_per_epoch = full_groups_per_epoch
    scheduler = (
        build_scheduler(optimizer, epochs * steps_per_epoch, steps_per_epoch)
        if protocol.lr_schedule == "warmup1-cosine"
        else None
    )

    def logits_for(image_u8: torch.Tensor, *, nir_on: bool = True) -> torch.Tensor:
        """Four-band input through the frozen encoder and head, with graph kept."""
        image = image_u8.to(device)
        rgb = (image[:, :3].float() - inputs.rgb_mean) / inputs.rgb_std
        nir = inputs.plugin_image(image)[:, 3:4] if nir_on else torch.zeros_like(rgb[:, :1])
        x = torch.cat([rgb, nir], dim=1)
        with torch.set_grad_enabled(True), inputs.autocast():
            levels = encoder(x)
            logits = model.head(levels, output_size=tuple(image.shape[-2:]))
        logits = logits.float()
        _require_finite("stem4 logits", logits)
        return logits

    def step_group(group: list[dict]) -> tuple[float, int, bool]:
        """One optimizer step over a complete accumulation group (see t13_fusion_train)."""
        labels_list = [batch["label"].to(device, non_blocking=True) for batch in group]
        valid = sum(int((labels != IGNORE_INDEX).sum()) for labels in labels_list)
        if valid == 0:
            return 0.0, 0, False
        optimizer.zero_grad(set_to_none=True)
        ce_sum = 0.0
        for batch, labels in zip(group, labels_list, strict=True):
            logits = logits_for(batch["image"])
            loss_sum = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="sum")
            _require_finite("training loss", loss_sum)
            (loss_sum / valid).backward()
            ce_sum += float(loss_sum.detach())
        _require_finite("NIR gradient", stem.nir.weight.grad)
        optimizer.step()
        _require_finite("NIR weight", stem.nir.weight.detach())
        if scheduler is not None:
            scheduler.step()
        return ce_sum, valid, True

    def train_epoch(epoch_loader) -> tuple[float, int, int]:
        ce_total = 0.0
        valid_total = 0
        batches = 0
        steps = 0
        group: list[dict] = []
        for batch in epoch_loader:
            group.append(batch)
            batches += 1
            if len(group) == accumulation:
                ce, valid, stepped = step_group(group)
                ce_total += ce
                valid_total += valid
                steps += int(stepped)
                group = []
            if limit and batches >= limit:
                break
        mean_ce = ce_total / valid_total if valid_total else 0.0
        return mean_ce, batches, steps

    def evaluate(rows: list[dict], sheet_limit: int) -> dict:
        with torch.inference_mode():
            metrics, _ = score_sheets(
                rows,
                data_root,
                lambda image, _codes: logits_for(image),
                with_aef=False,
                batch=micro,
                limit=sheet_limit,
            )
        return metrics

    # Step-0 check: with zero NIR weights, the four-band path equals the same path
    # with the NIR channel zeroed, i.e. the RGB-only FT-B0 model.
    probe = next(iter(loader))["image"][:2]
    with torch.inference_mode():
        same = torch.equal(logits_for(probe), logits_for(probe, nir_on=False))
    if not same:
        raise RuntimeError("step-0 logits differ from the FT-B0 reference; refusing to train")

    val_every = validation_every()
    best_score = float("-inf")
    best_epoch = 0
    best_state: torch.Tensor | None = None
    optimizer_steps = 0
    started = time.time()
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        mean_ce, _, steps = train_epoch(loader)
        optimizer_steps += steps
        writer.add_curve("ce", "train", epoch, mean_ce)
        if (epoch + 1) % val_every == 0 or epoch + 1 == epochs:
            val = evaluate(val_rows, eval_sheets)
            writer.add_curve("mean_iou", "validation", epoch, val["mean_iou"])
            current = stem.nir.weight.detach().cpu()
            # A candidate must already differ from the initial weights.
            if (
                _max_change(initial, current) > ZERO_UPDATE_TOLERANCE
                and val["mean_iou"] > best_score
            ):
                best_score = val["mean_iou"]
                best_epoch = epoch + 1
                best_state = current.clone()
    if optimizer_steps == 0:
        raise RuntimeError("no optimizer step was taken; this run cannot complete")
    moved = _max_change(initial, stem.nir.weight.detach().cpu())
    if moved <= ZERO_UPDATE_TOLERANCE:
        raise RuntimeError(
            f"NIR weights moved by at most {moved:.1e}; the plugin has no trained update"
        )
    if best_state is None:
        raise RuntimeError("no validation pass ran after a trained update; nothing to export")
    if _max_change(initial, best_state) <= ZERO_UPDATE_TOLERANCE:
        raise RuntimeError("the selected checkpoint has no trained update; refusing to export it")
    _require_finite("selected NIR weight", best_state)
    pin = FT_B0_BASELINES[seed]
    manifest_sha = manifest_sha256(manifest)
    torch.save(
        {
            "format": STEM4_FORMAT,
            "protocol": protocol.protocol_id,
            "baseline_run": dependency.parent.name,
            "baseline_checkpoint_sha256": pin["checkpoint_sha256"],
            "manifest_sha256": manifest_sha,
            "precision": "fp32 weights, bf16 autocast on CUDA (FT-B0 path)",
            "trainable_parameters": int(best_state.numel()),
            "config": {"nir": True, "sar": False, "channels": CHANNELS, "frozen_rgb_stem": True},
            "band_mean": band_mean.tolist(),
            "band_std": band_std.tolist(),
            "state_dict": {"nir.weight": best_state},
        },
        writer.run_dir / "stem4_plugin.pt",
    )
    writer.add_artifact("checkpoint", "stem4_plugin.pt", "validation-selected NIR first layer")
    writer.update(
        tags={
            "selection": "best_validation_mean_iou",
            "selected_epoch": str(best_epoch),
            "validation_selection_score": f"{best_score:.4f}",
            "baseline_run": dependency.parent.name,
            "baseline_checkpoint_sha256": pin["checkpoint_sha256"],
            "manifest_sha256": manifest_sha,
            "optimizer_steps": str(optimizer_steps),
            "test_evaluated": "no",
        }
    )
    writer.update_training(
        epochs_completed=epochs,
        optimizer_steps=optimizer_steps,
        wall_time_seconds=time.time() - started,
        trainable_parameters=int(best_state.numel()),
        total_parameters=sum(p.numel() for p in encoder.parameters()),
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
