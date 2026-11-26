"""Frozen DOFA-Base injector at F3 with the UPerNet head trainable (whu_ft_dofa_head).

The FT-B0 LIM and DOFA-Base stay frozen. DOFA reads the raw NIR band (wavelength
0.83 um, provisional) at 512 px and gives 32x32 tokens. A zero-initialised 1x1
convolution maps them to the 1024 channels of LIM F3, and the result is added to the
RGB F3 before the UPerNet head. The projection trains at ``learning_rate`` and the
head at ``backbone_learning_rate``. Validation selects the epoch; the test split is
never read here.
"""

from __future__ import annotations

import hashlib
import io
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rs_backbone_eval.data.manifest import manifest_sha256
from rs_backbone_eval.normalization import normalization_seed
from rs_backbone_eval.schedule import build_scheduler
from rs_backbone_eval.selection import validation_every

from .arms import ArmModel
from .dofa import DofaViTBase
from .mit import mit_huge
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
from .t13_fusion_train import REPO_ROOT, _require_finite, _residual_update

DOFA_FORMAT = "t13-dofa-plugin-v1"
DOFA_PROTOCOL = "whu_ft_dofa_head"
DOFA_WEIGHTS = Path("/geodata/rs-eval/weights/dofa/DOFA_ViT_base_e100.pth")
DOFA_WEIGHTS_SHA256 = "4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1"
# Provisional: centre of the 770-890 nm range for GF-1 band 4 (secondary source only).
DOFA_WAVE_UM = 0.83
F3_LEVEL = 2  # LIM outputs F1..F4 at indices 0..3; F3 is index 2, 1024 channels
ZERO_UPDATE_TOLERANCE = 1e-6


class DofaF3Plugin(nn.Module):
    """Zero-initialised 1x1 projection from DOFA tokens (768 ch) to LIM F3 (1024 ch)."""

    def __init__(self, in_channels: int = 768, out_channels: int = CHANNELS[F3_LEVEL]):
        super().__init__()
        self.dofa = nn.Module()
        self.dofa.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        nn.init.zeros_(self.dofa.proj.weight)
        nn.init.zeros_(self.dofa.proj.bias)


def load_dofa() -> DofaViTBase:
    """Pinned DOFA-Base weights; refuses any file that differs from the recorded SHA-256."""
    raw = DOFA_WEIGHTS.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DOFA_WEIGHTS_SHA256:
        raise RuntimeError("DOFA weights differ from the pinned SHA-256")
    # Load the same bytes that were hashed, and only as tensors (no pickled objects).
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    model = DofaViTBase()
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"DOFA checkpoint is missing keys: {result.missing_keys[:5]}")
    return model.eval()


def train_dofa(
    request,
    writer,
    emit: Callable[..., None],
    *,
    protocol,
    task_spec,
    model_id: str,
) -> int:
    if protocol.protocol_id != DOFA_PROTOCOL:
        raise ValueError(f"T13 DOFA does not implement protocol {protocol.protocol_id}")
    run_id, seed = request.run_id, request.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(request.data_root)
    runs_root = Path(request.runs_root)
    manifest = Path(request.manifest_path)

    epochs = int(protocol.epochs or 0)
    lr = float(protocol.learning_rate or 0)
    head_lr = float(protocol.backbone_learning_rate or 0)
    if not (epochs and lr and head_lr):
        raise ValueError(
            f"{protocol.protocol_id}: must declare epochs, learning_rate and backbone_learning_rate"
        )
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
        raise ValueError("T13 DOFA needs train and validation tiles")
    manifest_sha = manifest_sha256(manifest)
    dependency = find_full_dependency(
        runs_root,
        seed=seed,
        manifest_sha=manifest_sha,
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
    encoder = encoder.to(device).eval()
    model = model.to(device).eval()
    for param in list(encoder.parameters()):
        param.requires_grad_(False)
    dofa = load_dofa().to(device)
    for param in dofa.parameters():
        param.requires_grad_(False)
    plugin = DofaF3Plugin().to(device).train()
    head_params = list(model.head.parameters())
    for param in head_params:
        param.requires_grad_(True)
    inputs = Inputs(encoder, band_mean, band_std, device, finetune=True)

    plugin_params = list(plugin.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": plugin_params, "lr": lr}, {"params": head_params, "lr": head_lr}],
        weight_decay=1e-4,
    )
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

    def logits_for(image_u8: torch.Tensor, *, plugin_on: bool = True) -> torch.Tensor:
        """LIM RGB features, the DOFA residual at F3, then the trainable UPerNet head."""
        image = image_u8.to(device)
        feats = list(inputs.features(image, grad=False))
        size = tuple(image.shape[-2:])
        if plugin_on:
            nir = inputs.plugin_image(image)[:, 3:4]
            with torch.no_grad():
                tokens = dofa.forward_map(nir, [DOFA_WAVE_UM])
            residual = plugin.dofa.proj(tokens.float())
            if residual.shape[-2:] != feats[F3_LEVEL].shape[-2:]:
                raise RuntimeError(
                    f"DOFA grid {tuple(residual.shape[-2:])} differs from LIM F3 "
                    f"{tuple(feats[F3_LEVEL].shape[-2:])}"
                )
            feats[F3_LEVEL] = feats[F3_LEVEL].float() + residual
        with inputs.autocast():
            logits = model.head(feats, output_size=size)
        logits = logits.float()
        _require_finite("dofa logits", logits)
        return logits

    def step_group(group: list[dict]) -> tuple[float, int, bool]:
        """One optimizer step over a complete accumulation group (as in t13_fusion_train)."""
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
        for param in plugin_params:
            if param.grad is not None:
                _require_finite("DOFA projection gradient", param.grad)
        optimizer.step()
        for param in plugin_params + head_params:
            _require_finite("DOFA trainable weight", param.detach())
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
        plugin.eval()
        model.head.eval()
        with torch.inference_mode():
            metrics, _ = score_sheets(
                rows,
                data_root,
                lambda image, _codes: logits_for(image),
                with_aef=False,
                batch=micro,
                limit=sheet_limit,
            )
        plugin.train()
        model.head.train()
        return metrics

    # Step-0 check: the zero projection must reproduce the RGB-only head exactly.
    probe = next(iter(loader))["image"][:2]
    model.head.eval()
    with torch.inference_mode():
        same = torch.equal(logits_for(probe), logits_for(probe, plugin_on=False))
    model.head.train()
    if not same:
        raise RuntimeError("step-0 logits differ from the FT-B0 reference; refusing to train")

    def snapshot_plugin() -> dict:
        return {k: v.detach().cpu().clone() for k, v in plugin.state_dict().items()}

    initial_state = snapshot_plugin()
    val_every = validation_every()
    best_score = float("-inf")
    best_epoch = 0
    best_state: dict | None = None
    best_head: dict | None = None
    optimizer_steps = 0
    started = time.time()
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        plugin.train()
        model.head.train()
        mean_ce, _, steps = train_epoch(loader)
        optimizer_steps += steps
        writer.add_curve("ce", "train", epoch, mean_ce)
        if (epoch + 1) % val_every == 0 or epoch + 1 == epochs:
            val = evaluate(val_rows, eval_sheets)
            writer.add_curve("mean_iou", "validation", epoch, val["mean_iou"])
            current = snapshot_plugin()
            # The projection must already be non-zero for a candidate to count.
            if (
                _residual_update(initial_state, current) > ZERO_UPDATE_TOLERANCE
                and val["mean_iou"] > best_score
            ):
                best_score = val["mean_iou"]
                best_epoch = epoch + 1
                best_state = current
                best_head = {
                    k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()
                }
    if optimizer_steps == 0:
        raise RuntimeError("no optimizer step was taken; this run cannot complete")
    moved = _residual_update(initial_state, snapshot_plugin())
    if moved <= ZERO_UPDATE_TOLERANCE:
        raise RuntimeError(
            f"DOFA projection moved by at most {moved:.1e}; the plugin has no trained update"
        )
    if best_state is None or best_head is None:
        raise RuntimeError("no validation pass ran after a trained update; nothing to export")
    if _residual_update(initial_state, best_state) <= ZERO_UPDATE_TOLERANCE:
        raise RuntimeError("the selected checkpoint has no trained update; refusing to export it")
    for name, tensor in best_state.items():
        _require_finite(f"selected weight {name}", tensor)
    pin = FT_B0_BASELINES[seed]
    out_path = writer.run_dir / "dofa_plugin.pt"
    torch.save(
        {
            "format": DOFA_FORMAT,
            "protocol": protocol.protocol_id,
            "baseline_run": dependency.parent.name,
            "baseline_checkpoint_sha256": pin["checkpoint_sha256"],
            "manifest_sha256": manifest_sha,
            "dofa_weights_sha256": DOFA_WEIGHTS_SHA256,
            "dofa_wave_um": DOFA_WAVE_UM,
            "precision": "fp32 weights, bf16 autocast on CUDA (FT-B0 path)",
            "config": {
                "nir": True,
                "sar": False,
                "head": True,
                "dofa": True,
                "level": F3_LEVEL,
                "channels": CHANNELS,
            },
            "band_mean": band_mean.tolist(),
            "band_std": band_std.tolist(),
            "smoke": smoke,
            "state_dict": best_state,
            "head_state": best_head,
        },
        out_path,
    )
    plugin_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    writer.add_artifact(
        "checkpoint", "dofa_plugin.pt", "validation-selected DOFA F3 injector and head"
    )
    writer.update(
        tags={
            "selection": "best_validation_mean_iou",
            "selected_epoch": str(best_epoch),
            "validation_selection_score": f"{best_score:.4f}",
            "baseline_run": dependency.parent.name,
            "baseline_checkpoint_sha256": pin["checkpoint_sha256"],
            "manifest_sha256": manifest_sha,
            "plugin_sha256": plugin_sha,
            "optimizer_steps": str(optimizer_steps),
            "test_evaluated": "no",
        }
    )
    trainable_count = sum(p.numel() for p in plugin_params) + sum(p.numel() for p in head_params)
    writer.update_training(
        epochs_completed=epochs,
        optimizer_steps=optimizer_steps,
        wall_time_seconds=time.time() - started,
        trainable_parameters=trainable_count,
        total_parameters=sum(p.numel() for p in dofa.parameters())
        + sum(p.numel() for p in encoder.parameters()),
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
