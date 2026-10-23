"""T3 (burn intensity) B0 reference for Prithvi-EO-2.0.

Follows the official ``configs/burnintensity.yaml``:

- Prithvi-EO-2.0 300M-TL, 3 frames, 6 HLS bands, location coordinates
- UPerNet decoder (512 channels), token indices 7/15/23/31
- CE with the official class weights, AdamW lr 1e-5 / wd 0.1, StepLR(3, 0.9)
- 50 epochs, batch 8, 16-mixed

The split comes from the project manifest (train/validation/test), so the only
differences from the B2/B1 matrix are the head and the official recipe. The
run record states that in its notes.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.groups import GroupedSegmentation
from rs_backbone_eval.evaluation.segmentation import multiclass_metrics
from rs_backbone_eval.worker_runtime import RunWriter

BANDS = ["BLUE", "GREEN", "RED", "NIR", "SWIR_1", "SWIR_2"]
NUM_CLASSES = 5
IGNORE_INDEX = -1
TIME_STEPS = 3
TOKEN_INDICES = [5, 11, 17, 23]  # 300M backbone; 7/15/23/31 is the 600M set
CLASS_WEIGHTS = [0.018, 0.14, 0.076, 0.11, 0.65]
OFFICIAL_EPOCHS = 50
OFFICIAL_BATCH_SIZE = 8
SELECT_SPLIT = "val"
EVAL_SPLIT = "test"


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


def _read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _names(rows: list[dict], split: str) -> list[str]:
    return [f"HLS_{row['sample_id']}.tiff" for row in rows if row["split"] == split]


def _prepare_root(
    data_root: Path,
    cache_root: Path,
    manifest_path: Path,
    split_lists: dict[str, list[str]],
    suffix: str,
) -> Path:
    """Build a TerraTorch-compatible data root from the manifest split."""
    source = data_root / "raw" / "burn_intensity" / "extracted"
    csv = data_root / "raw" / "burn_intensity" / "main" / "BS_files_raw.csv"
    key = hashlib.sha256(str(manifest_path).encode()).hexdigest()[:16] + f"-{suffix}"
    root = cache_root / "burn_intensity_b0" / key
    root.mkdir(parents=True, exist_ok=True)
    for step in ("pre", "during", "post"):
        link = root / step
        if not link.exists():
            link.symlink_to(source / step, target_is_directory=True)
    target_csv = root / "BS_files_raw.csv"
    if not target_csv.exists():
        shutil.copy2(csv, target_csv)
    for split, names in split_lists.items():
        (root / f"{split}.txt").write_text("\n".join(names), encoding="utf-8")
    return root


def _build_datamodule(data_root: Path, batch_size: int, num_workers: int):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from terratorch.datamodules import BurnIntensityNonGeoDataModule
    from terratorch.datasets.transforms import (
        FlattenTemporalIntoChannels,
        UnflattenTemporalFromChannels,
    )

    train_transform = [
        FlattenTemporalIntoChannels(),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2(),
        UnflattenTemporalFromChannels(n_timesteps=TIME_STEPS),
    ]
    eval_transform = [
        FlattenTemporalIntoChannels(),
        ToTensorV2(),
        UnflattenTemporalFromChannels(n_timesteps=TIME_STEPS),
    ]
    return BurnIntensityNonGeoDataModule(
        data_root=str(data_root),
        batch_size=batch_size,
        num_workers=num_workers,
        bands=BANDS,
        use_full_data=True,
        use_metadata=True,
        train_transform=train_transform,
        val_transform=eval_transform,
        test_transform=eval_transform,
    )


def _build_task(checkpoint: Path, *, epochs: int):
    from terratorch.tasks import SemanticSegmentationTask

    return SemanticSegmentationTask(
        model_args={
            "backbone": "prithvi_eo_v2_300_tl",
            "backbone_kwargs": {
                "pretrained": True,
                "ckpt_path": str(checkpoint),
                "bands": BANDS,
                "num_frames": TIME_STEPS,
                "coords_encoding": ["location"],
            },
            "decoder": "UperNetDecoder",
            "decoder_kwargs": {"channels": 512},
            "num_classes": NUM_CLASSES,
            "rescale": True,
            "head_kwargs": {"dropout": 0.1},
            "necks": [
                {"name": "SelectIndices", "indices": TOKEN_INDICES},
                {"name": "ReshapeTokensToImage", "effective_time_dim": TIME_STEPS},
            ],
        },
        model_factory="EncoderDecoderFactory",
        loss="ce",
        class_weights=CLASS_WEIGHTS,
        ignore_index=IGNORE_INDEX,
        lr=_float_env("PRITHVI_WORKER_LR", 1e-5),
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 0.1},
        scheduler="StepLR",
        scheduler_hparams={"step_size": 3, "gamma": 0.9},
    )


def _run_test_evaluation(task, datamodule, groups: dict[str, str], device: str):
    import torch

    task.eval()
    task.to(device)
    datamodule.setup("test")
    dataset = datamodule.test_dataset
    names = [sample["casename"] for sample in dataset.samples]
    collector = GroupedSegmentation(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    offset = 0
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            batch = {
                key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)
            }
            batch = datamodule.on_after_batch_transfer(batch)
            output = task.model(batch["image"], location_coords=batch.get("location_coords"))
            logits = output.output if hasattr(output, "output") else output
            predictions = logits.argmax(dim=1)
            size = batch["mask"].shape[0]
            batch_groups = [groups.get(name, name) for name in names[offset : offset + size]]
            collector.update(batch_groups, batch["mask"].cpu().numpy(), predictions.cpu().numpy())
            offset += size
    return multiclass_metrics(collector.global_matrix), collector.group_confusion()


def train_baseline_t3(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    checkpoint: Path,
) -> int:
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint

    from rs_backbone_eval.repro import random_init_requested

    if random_init_requested():
        message = (
            "B0 is the Prithvi reference reproduction and always loads the "
            "pretrained weights; RS_EVAL_INIT=random is not supported here"
        )
        emit(request.run_id, "train.init", message, level="error")
        writer.fail(message)
        return 1

    run_id = request.run_id
    data_root = Path(request.data_root)
    cache_root = Path(request.cache_root)
    run_dir = Path(request.runs_root) / run_id
    manifest_path = Path(request.manifest_path)

    epochs = _int_env("PRITHVI_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("PRITHVI_WORKER_BATCH_SIZE", OFFICIAL_BATCH_SIZE)
    num_workers = _int_env("PRITHVI_WORKER_NUM_WORKERS", 8)
    limit_batches = _int_env("PRITHVI_WORKER_LIMIT_BATCHES", 0) or None
    smoke = epochs != OFFICIAL_EPOCHS or limit_batches is not None

    rows = _read_manifest(manifest_path)
    groups = {row["sample_id"]: row.get("group_id", row["sample_id"]) for row in rows}
    train_names = _names(rows, "train")
    val_names = _names(rows, "validation")
    test_names = _names(rows, "test")
    emit(
        run_id,
        "train.prepare",
        (
            f"manifest splits: {len(train_names)} train / {len(val_names)} val / "
            f"{len(test_names)} test"
        ),
    )

    train_root = _prepare_root(
        data_root,
        cache_root,
        manifest_path,
        {"train": train_names, "val": val_names},
        "train",
    )
    test_root = _prepare_root(
        data_root,
        cache_root,
        manifest_path,
        {"train": train_names, "val": test_names},
        "test",
    )
    datamodule = _build_datamodule(train_root, batch_size, num_workers)
    test_module = _build_datamodule(test_root, batch_size, num_workers)

    task = _build_task(checkpoint, epochs=epochs)
    trainable = sum(p.numel() for p in task.parameters() if p.requires_grad)
    total = sum(p.numel() for p in task.parameters())
    writer.update_training(
        head_id="upernet_decoder", trainable_parameters=trainable, total_parameters=total
    )
    writer.update_input(preprocessing_id="official_burn_intensity_b0_224")
    writer.update(
        tags={
            "learning_rate": f"{_float_env('PRITHVI_WORKER_LR', 1e-5):g}",
            "loss": "ce",
            "class_weights": ",".join(str(value) for value in CLASS_WEIGHTS),
            "decoder": "UperNetDecoder(512)",
            "select_split": SELECT_SPLIT,
            "eval_split": EVAL_SPLIT,
        }
    )
    writer.update(
        notes=(
            "T3 paper-config reference (B0): Prithvi-EO-2.0 300M-TL, 3 frames, "
            "location coords, UPerNet 512 (indices 7/15/23/31), official class "
            "weights, AdamW 1e-5/0.1, StepLR(3,0.9), batch 8, 50 epochs. "
            "Splits come from the project manifest; head and recipe follow "
            "configs/burnintensity.yaml. Selection on validation loss."
        )
    )
    emit(
        run_id,
        "train.model",
        f"model ready: {trainable / 1e6:.1f}M trainable / {total / 1e6:.1f}M total",
    )

    class _Progress(L.pytorch.callbacks.Callback):
        def on_train_epoch_end(self, trainer, module) -> None:
            loss = trainer.callback_metrics.get("train/loss")
            value = float(loss) if loss is not None else float("nan")
            emit(
                run_id,
                "train.epoch",
                f"epoch {trainer.current_epoch + 1}/{trainer.max_epochs} loss={value:.4f}",
                progress=(trainer.current_epoch + 1) / trainer.max_epochs,
            )

        def on_validation_epoch_end(self, trainer, module) -> None:
            loss = trainer.callback_metrics.get("val/loss")
            if loss is None:
                return
            value = float(loss)
            writer.add_curve("loss", "validation", trainer.current_epoch, value)
            emit(run_id, "train.validation", f"validation loss {value:.4f}", level="debug")

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch:02d}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=epochs,
        limit_train_batches=limit_batches,
        limit_val_batches=limit_batches,
        check_val_every_n_epoch=2,
        log_every_n_steps=10,
        default_root_dir=str(run_dir),
        enable_progress_bar=False,
        callbacks=[_Progress(), checkpoint_callback],
        logger=False,
    )

    started = time.time()
    emit(run_id, "train.start", f"fitting for {epochs} epochs (smoke={smoke})")
    L.pytorch.seed_everything(request.seed, workers=True)
    torch.cuda.reset_peak_memory_stats()
    trainer.fit(task, datamodule)
    wall_time = time.time() - started
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)

    best = checkpoint_callback.best_model_path
    if best and Path(best).exists():
        state = torch.load(best, map_location="cpu")
        task.load_state_dict(state.get("state_dict", state))
        emit(run_id, "train.restore", f"restored best validation checkpoint {Path(best).name}")
        writer.update(
            tags={
                "selection": "best_validation_loss",
                "selected_checkpoint": Path(best).name,
                "best_val_loss": f"{float(checkpoint_callback.best_model_score or 0):.4f}",
            }
        )

    metrics, group_json = _run_test_evaluation(task, test_module, groups, "cuda")
    writer.add_metrics(
        to_metric_records(metrics, primary_name="mean_iou", direction="maximize", split=EVAL_SPLIT)
    )
    writer.add_confusion(group_json, NUM_CLASSES)
    writer.update_training(
        epochs_completed=trainer.current_epoch,
        wall_time_seconds=wall_time,
        peak_gpu_memory_mb=peak_memory,
    )
    if best and Path(best).exists():
        writer.add_artifact(
            "checkpoint", str(Path(best).relative_to(run_dir)), "best validation checkpoint"
        )
    summary = (
        f"mean_iou={metrics['mean_iou']:.4f} macro_f1={metrics['macro_f1']:.4f} "
        f"({epochs} epochs, best val loss {float(checkpoint_callback.best_model_score or 0):.4f})"
    )
    if smoke:
        summary += " [SMOKE: non-official epoch/batch limits]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
