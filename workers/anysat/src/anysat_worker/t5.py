"""T5 (US multi-temporal crop) training for the AnySat worker.

AnySat's public ``s2`` entrance natively consumes a time series
(``[B,T,10,H,W]`` plus day-of-year dates) and fuses it internally (LTAE +
temporal attention), so its output is a single fused dense map. The six task
bands are mapped to the ten-band entrance by the explicit spectral adapter
(zero padding is forbidden by the design).

Protocol note: unlike Clay and Prithvi, which encode each date separately and
let the shared ``TemporalSegmentationHead`` fuse them, AnySat uses its native
temporal fusion and the shared spatial head. That difference is recorded in
run tags and must be stated in any comparison table.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from rs_backbone_eval.checkpoint import TrainingCheckpoint
from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.groups import GroupedSegmentation
from rs_backbone_eval.evaluation.segmentation import multiclass_metrics
from rs_backbone_eval.heads import LiteSpatialHead, count_parameters
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.splits import evaluation_split
from rs_backbone_eval.temporal_io import (
    multicrop_statistics,
    read_multicrop_label,
    read_stacked_dates,
)
from rs_backbone_eval.timeutil import day_of_year
from rs_backbone_eval.training import scale_accumulated_gradients
from rs_backbone_eval.worker_runtime import RunWriter

from .s2_probe import add_upstream_to_path
from .spectral_adapter import SpectralAdapter

BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
NUM_CLASSES = 13
IGNORE_INDEX = -1
TIME_STEPS = 3
CHIP_SIZE = 224
PATCH_SIZE_M = 160
OFFICIAL_EPOCHS = 125
EVAL_SPLIT = evaluation_split("official_validation")


class _CacheIncomplete(RuntimeError):
    """The feature cache does not cover the whole training split."""


def seed_loader_for_epoch(loader: DataLoader, seed: int, epoch: int) -> None:
    """Rewrite a shuffled loader's permutation as a pure function of (seed, epoch).

    Review P2: the online and cached paths must iterate the same samples in the
    same order. A ``shuffle=True`` loader draws its permutation from whatever
    random state exists when the epoch starts, and the model's own dropout draws
    from the same state, so the two paths diverge after the first batch. Seeding
    an explicit generator per epoch removes that coupling and makes the cached
    sequence provably identical to the online one.
    """
    generator = getattr(getattr(loader, "sampler", None), "generator", None)
    if generator is None:
        raise ValueError("loader must be built with shuffle=True and an explicit generator")
    generator.manual_seed(seed * 100003 + epoch)


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


class MulticropDataset(Dataset):
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
        images = read_stacked_dates(self.data_root / row["image_paths"][0], channels=len(BANDS))
        label = read_multicrop_label(self.data_root / row["label_path"])
        doy = [day_of_year(stamp) for stamp in row["timestamps"]]
        return {"images": images, "label": label, "dates": np.asarray(doy, dtype=np.int64)}


def build_backbone(checkpoint: Path):
    add_upstream_to_path()
    from hubconf import AnySat  # type: ignore[import-not-found]  # upstream checkout

    model = AnySat(model_size="base", flash_attn=False)
    if not random_init_requested():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.model.load_state_dict(state)
    return model


class AnySatT5(nn.Module):
    """Spectral adapter + AnySat's native time-series entrance + spatial head."""

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
        return self.head(self.fused_features(images, dates))

    def fused_features(self, images: torch.Tensor, dates: torch.Tensor) -> torch.Tensor:
        """Fused dense map [B, 2D, H, W]: adapter + backbone + permute."""
        adapted = self.adapter(images)  # [B,T,10,H,W]
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
        return dense.permute(0, 3, 1, 2)  # [B,H,W,2D] -> [B,2D,H,W]


def train_t5(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    checkpoint: Path,
    task_bands: list[str],
    freeze_backbone: bool,
    metric_name: str,
) -> int:
    run_id = request.run_id
    data_root = Path(request.data_root)
    device = os.environ.get("ANYSAT_WORKER_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    epochs = _int_env("ANYSAT_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("ANYSAT_WORKER_BATCH_SIZE", 2)
    # Micro-batch x accumulation = effective batch, so a smaller micro-batch can
    # be used when memory is tight without changing the effective batch.
    grad_accum = max(1, _int_env("ANYSAT_WORKER_GRAD_ACCUM", 1))
    lr = _float_env("ANYSAT_WORKER_LR", 1e-3)
    workers = _int_env("ANYSAT_WORKER_NUM_WORKERS", 4)
    limit = _int_env("ANYSAT_WORKER_LIMIT_BATCHES", 0) or None
    # Review P2 (second round): the cap changes which rows are trained on, so it
    # belongs to the checkpoint identity. Reading it here (not inside the cache
    # prelude) is what lets a resume with a different cap be rejected.
    cache_max_rows = _int_env("ANYSAT_WORKER_CACHE_MAX_ROWS", 0)
    if cache_max_rows and not limit:
        # The cap is a small-sample smoke knob. It now also binds the online
        # fallback, so the guard has to sit here rather than inside the cache
        # prelude: a formal run must never shrink its training set silently.
        raise ValueError(
            "ANYSAT_WORKER_CACHE_MAX_ROWS caps the training set; allowed only "
            "together with ANYSAT_WORKER_LIMIT_BATCHES (smoke runs)"
        )

    mean, std = multicrop_statistics(BANDS)
    means = torch.tensor(mean, device=device).view(1, 1, -1, 1, 1)
    stds = torch.tensor(std, device=device).view(1, 1, -1, 1, 1)

    backbone = build_backbone(checkpoint).to(device)
    model = AnySatT5(backbone, task_bands, freeze_backbone=freeze_backbone).to(device)
    adapter_params = count_parameters(model.adapter)
    head_params = count_parameters(model.head)

    manifest = Path(request.manifest_path)
    train_ds = MulticropDataset(manifest, data_root, "train")
    eval_ds = MulticropDataset(manifest, data_root, EVAL_SPLIT)
    eval_limit = _int_env("ANYSAT_WORKER_EVAL_LIMIT", 0)
    if eval_limit:
        eval_ds.rows = eval_ds.rows[:eval_limit]
    emit(
        run_id,
        "train.prepare",
        f"AnySat T5 native time series (freeze={freeze_backbone}): {len(train_ds)} train / "
        f"{len(eval_ds)} {EVAL_SPLIT}, {TIME_STEPS} dates, "
        f"adapter {adapter_params} + head {head_params}",
    )
    writer.update_training(
        head_id="lite_spatial_head+spectral_adapter(native_time_series)",
        trainable_parameters=count_parameters(model, trainable_only=True),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="official_multicrop_stats_native_time_series")
    writer.update(
        tags={
            "adapter_parameters": str(adapter_params),
            "head_parameters": str(head_params),
            "backbone_frozen": str(freeze_backbone),
            "time_steps": str(TIME_STEPS),
            "band_layout": "date-major (T x C)",
            "label_mapping": "reduce_zero_label (0->ignore)",
            "temporal_fusion": "native (model-internal); spatial head on the fused map",
            "head_note": "differs from Clay/Prithvi (per-date encoding + temporal head)",
            "eval_split": EVAL_SPLIT,
            "micro_batch": str(batch_size),
            "grad_accum": str(grad_accum),
            "effective_batch": str(batch_size * grad_accum),
            "learning_rate": f"{lr:g}",
            "device": device,
        }
    )

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    params = [p for p in model.parameters() if p.requires_grad]

    def _scale_and_step(valid_pixels: int) -> None:
        """Scale the accumulated gradient by the window's valid pixels, then step."""
        scale_accumulated_gradients(params, valid_pixels)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Two-stage protocol (docs/reviews/缓存分支审查响应_2026-10-16): stage A
    # trains the spectral adapter + head online (backbone frozen, no cache);
    # stage B freezes the adapter, materialises the frozen features once, and
    # continues training the head on the cache. `online` keeps today's
    # behaviour unchanged, so nothing here breaks the active path.
    # Review P1: the protocol owns the training mode. A two-stage record stays
    # in its own cell: the two-stage protocol forces stage "two" and every
    # other protocol stays online; a conflicting env override is a misconfig.
    protocol_id = writer.record.protocol_id
    protocol_stage = "two" if protocol_id == "frozen_adapter_two_stage" else "online"
    env_stage = os.environ.get("ANYSAT_T5_STAGE")
    if env_stage is not None and env_stage.lower() != protocol_stage:
        raise ValueError(
            f"ANYSAT_T5_STAGE={env_stage!r} does not match protocol "
            f"{protocol_id!r} (expected stage {protocol_stage!r}); dispatch the "
            "mode's own protocol instead"
        )
    stage = protocol_stage
    if stage not in {"online", "a", "b", "two"}:
        raise ValueError(f"resolved stage {stage!r} is not a known mode")
    stage_b_start = min(
        epochs, int(os.environ.get("ANYSAT_STAGE_B_START", str(max(1, epochs // 2))))
    )
    # review: the freeze point is part of the experimental condition and must
    # land in the record tags no matter which cache path the run takes
    writer.update(tags={"stage_b_start": str(stage_b_start)})
    if stage in {"b", "two"} and not freeze_backbone:
        # review: a full-finetune request must never claim stage B semantics
        raise ValueError(
            "ANYSAT_T5_STAGE=b/two requires a frozen backbone protocol; "
            "full-finetune runs must not be recorded as stage B"
        )

    checkpoints = TrainingCheckpoint(
        request,
        writer,
        model,
        optimizer,
        settings={
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "lr": lr,
            "freeze": freeze_backbone,
            "limit": limit,
            "eval_limit": eval_limit,
            "eval_split": EVAL_SPLIT,
            "stage": stage,
            "stage_b_start": stage_b_start,
            # Review P2: a resume may not change the training subset. The row
            # count is part of the identity, and the cache must cover all rows.
            "train_rows": len(train_ds),
            "cache_max_rows": cache_max_rows,
        },
    )
    emit(run_id, "train.resume", f"starting at epoch {checkpoints.start_epoch + 1}/{epochs}")
    optimizer_steps = checkpoints.optimizer_steps

    # Review #2e ordering: the checkpoint (with restored adapter weights) is in
    # place BEFORE any cache decision. Stage-B semantics after a resume: a
    # "two" run that already crossed its boundary continues with the adapter
    # frozen; a stage-b run always expects a frozen adapter (stage A outcome).
    crossed_boundary = stage == "two" and checkpoints.start_epoch >= stage_b_start
    if stage == "b" or crossed_boundary:
        # review #3: a full-finetune request must not get into the stage-B
        # head-only path; without a frozen backbone the stage is a misconfig
        if not freeze_backbone:
            raise ValueError(
                "ANYSAT_T5_STAGE=b/two requires a frozen backbone; "
                "full-finetune runs must not be claimed as stage B"
            )
        for parameter in model.adapter.parameters():
            parameter.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]

    # Review P2: an epoch's sample order must not depend on how much randomness
    # the model consumed, otherwise the cached path sees a different sample
    # sequence than the online path. Both loaders are reshuffled from an
    # explicit per-(seed, epoch) generator before every epoch.
    # Review P2 (third round): a smoke cap must bind every path. When the cache
    # is disabled or extraction fails, the fallback is this online loader; if it
    # kept the full split, the same run would train on more rows than the capped
    # cache path and the update count would jump.
    online_dataset = (
        Subset(train_ds, range(cache_max_rows))
        if cache_max_rows and cache_max_rows < len(train_ds)
        else train_ds
    )
    if online_dataset is not train_ds:
        emit(
            run_id,
            "train.cache",
            f"smoke cap: the online fallback trains on the first {cache_max_rows} rows",
        )
    loader = DataLoader(
        online_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=torch.Generator(),
    )
    feature_cache_used = False
    cache_directory: list[Path] = []

    def _cache_prelude() -> DataLoader | None:
        """Extract-or-load frozen dense features; None means the online path.

        Doc-promised fallback: every cache-side exception is caught here and
        recorded; the run continues on the online loader with warning tags.
        """
        from . import spectral_adapter as _adapter_module
        from .frozen_cache import (
            CachedSampleDataset,
            FrozenFeatureCache,
            populate,
            state_identity_hash,
        )

        if checkpoints.start_epoch >= epochs:
            # Review P2: finishing the budget and then re-running for evaluation
            # must not re-materialise ~428GB of features that are never read.
            emit(run_id, "train.cache", "training already complete; cache not built")
            return None

        max_rows = cache_max_rows or None
        try:
            # Review #3: the cache key binds the adapter+backbone state_dict
            # hash (the fixed feature extractor only), so different stage-A
            # seeds never share, and the head is excluded from the key.
            weights_fingerprint = state_identity_hash(model, prefixes=("adapter.", "backbone."))
            sources = [Path(__file__).resolve(), Path(_adapter_module.__file__).resolve()]
            identity_parts = {
                "weights_fingerprint": weights_fingerprint,
                "checkpoint_sha256": writer.record.model_checkpoint_sha256 or "",
                "manifest_sha256": writer.record.dataset.manifest_sha256,
                "bands": ",".join(BANDS),
                "patch_size_m": PATCH_SIZE_M,
                "time_steps": TIME_STEPS,
                "preprocessing_id": writer.record.input.preprocessing_id,
                "torch_version": torch.__version__,
                "cache_format": "npz-fp16-dense+int16-label-v1",
            }
            train_cache = FrozenFeatureCache(
                cache_root=Path(request.cache_root),
                identity_parts=identity_parts,
                worker_sources=sources,
                split="train",
            )
            cache_directory.append(train_cache.directory)
            materialised = populate(
                train_cache,
                train_ds,
                model,
                means,
                stds,
                run_id,
                emit,
                device,
                max_rows=max_rows,
            )
            if materialised <= 0:
                return None
            if not max_rows and materialised != len(train_ds):
                # Review P2: a short cache silently shrinks the training set and
                # the checkpoint identity cannot see it.
                raise _CacheIncomplete(
                    f"feature cache holds {materialised} rows but the training split "
                    f"has {len(train_ds)}; a resume must not change the training subset"
                )
            emit(
                run_id,
                "train.cache",
                "cache materialised: switching to the stage-B loader",
            )
            writer.update(
                tags={
                    "feature_cache": "disk fp16 (adapter+backbone materialised once)",
                    "feature_cache_dir": str(train_cache.directory),
                    "feature_cache_weights_fingerprint": weights_fingerprint,
                    "feature_cache_rows": str(materialised),
                    "feature_cache_training_rows_reduced": str(materialised < len(train_ds)),
                }
            )
            return DataLoader(
                CachedSampleDataset(train_cache, materialised),
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers,
                generator=torch.Generator(),
            )
        except _CacheIncomplete:
            raise
        except Exception as error:
            emit(
                run_id,
                "train.cache",
                f"cache unavailable ({type(error).__name__}): staying on the online loader",
                level="warning",
            )
            return None

    def _materialise_at_boundary() -> None:
        """Stage-two boundary: stage A finished, so freeze the adapter first.

        Entering stage B always freezes the adapter — the training algorithm
        stays identical whether the features come from the cache or the online
        path (review: resume consistency). Only then does the cache switch the
        loader; with the cache disabled stage B simply continues online with
        the adapter frozen.
        """
        nonlocal loader, feature_cache_used
        cached_loader = None
        if os.environ.get("ANYSAT_FROZEN_CACHE", "1") != "0":
            cached_loader = _cache_prelude()

        for parameter in model.adapter.parameters():
            parameter.requires_grad = False
        backbone_rg = any(p.requires_grad for p in model.backbone.parameters())
        if backbone_rg:
            emit(
                run_id,
                "train.cache",
                "stage-two boundary: gradients still present on the backbone after its freeze",
                level="warning",
            )
        if cached_loader is not None:
            loader = cached_loader
            feature_cache_used = True
        else:
            emit(
                run_id,
                "train.stage",
                "stage B continues online with the adapter frozen (cache disabled or unavailable)",
            )
            writer.update(tags={"stage_b_start": str(stage_b_start)})

    if (stage == "b" or crossed_boundary) and os.environ.get("ANYSAT_FROZEN_CACHE", "1") != "0":
        cached_loader = _cache_prelude()
        if cached_loader is not None:
            loader = cached_loader
            feature_cache_used = True
        else:
            emit(
                run_id,
                "train.cache",
                "cache unavailable: staying on the online loader",
                level="warning",
            )
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(checkpoints.start_epoch, epochs):
        if stage == "two" and epoch == stage_b_start and epoch > checkpoints.start_epoch:
            _materialise_at_boundary()
        seed_loader_for_epoch(loader, request.seed, epoch)
        model.train()
        steps, pending, window_valid = 0, 0, 0
        epoch_loss_sum, epoch_valid = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch in loader:
            if feature_cache_used:
                features = batch["dense"].to(device)
                labels = batch["label"].to(device)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
                ):
                    logits = model.head(features)
            else:
                images = (batch["images"].to(device) - means) / stds
                labels = batch["label"].to(device)
                dates = batch["dates"].to(device).float()
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
                ):
                    logits = model(images, dates)
            valid = int((labels != IGNORE_INDEX).sum())
            if valid == 0:
                continue
            loss_sum = criterion(logits.float(), labels)
            loss_sum.backward()
            window_valid += valid
            epoch_loss_sum += float(loss_sum.detach())
            epoch_valid += valid
            pending += 1
            if pending == grad_accum:
                _scale_and_step(window_valid)
                pending, window_valid = 0, 0
                steps += 1
                if limit and steps >= limit:
                    break
        if pending:  # flush a partial accumulation window at epoch end
            _scale_and_step(window_valid)
            steps += 1
        optimizer_steps += steps
        mean_loss = epoch_loss_sum / epoch_valid if epoch_valid else 0.0
        writer.add_curve("loss", "train", epoch, mean_loss)
        emit(
            run_id,
            "train.epoch",
            f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}",
            progress=(epoch + 1) / epochs,
        )
        checkpoints.save(epoch + 1, optimizer_steps, final=epoch + 1 == epochs)
    if checkpoints.start_epoch == epochs:
        checkpoints.save(epochs, optimizer_steps, final=True)
    wall = checkpoints.elapsed
    peak_mb = writer.record.training.peak_gpu_memory_mb or 0.0

    emit(run_id, "train.eval", f"evaluating on {len(eval_ds)} {EVAL_SPLIT} chips")
    collector = GroupedSegmentation(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    model.eval()
    with torch.inference_mode():
        loader = DataLoader(eval_ds, batch_size=1, shuffle=False, num_workers=2)
        for index, batch in enumerate(loader):
            images = (batch["images"].to(device) - means) / stds
            dates = batch["dates"].to(device).float()
            logits = model(images, dates)
            collector.update(
                [eval_ds.rows[index]["group_id"]],
                batch["label"].numpy(),
                logits.argmax(dim=1).cpu().numpy(),
            )
    metrics = multiclass_metrics(collector.global_matrix)
    writer.add_metrics(
        to_metric_records(metrics, primary_name=metric_name, direction="maximize", split=EVAL_SPLIT)
    )
    writer.add_confusion(collector.group_confusion(), NUM_CLASSES)
    writer.update_training(
        epochs_completed=epochs,
        optimizer_steps=optimizer_steps,
        wall_time_seconds=wall,
        peak_gpu_memory_mb=peak_mb,
    )
    summary = (
        f"{metric_name}={metrics['mean_iou']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"({epochs} epochs, adapter {adapter_params} + head {head_params}, "
        f"{'frozen' if freeze_backbone else 'fine-tuned'}, {wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]" if limit else " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    # Reclaim the (multi-hundred-GB) dense cache on success. A failed run keeps
    # it so a resume can continue stage B without rebuilding; set
    # ANYSAT_FROZEN_CACHE_KEEP=1 to keep it after success too.
    if (
        feature_cache_used
        and cache_directory
        and os.environ.get("ANYSAT_FROZEN_CACHE_KEEP", "0") != "1"
    ):
        import shutil

        shutil.rmtree(cache_directory[0], ignore_errors=True)
        emit(run_id, "train.cache", f"cache removed on completion: {cache_directory[0]}")
    return 0
