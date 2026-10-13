"""T1 repo-config baseline training (B0) for Prithvi-EO-2.0 on Sen1Floods11.

Follows the official Prithvi-EO-2.0 ``configs/sen1floods11.yaml`` sample
configuration: Prithvi-EO-2.0 300M-TL encoder loaded from the external
checkpoint, UPerNet decoder (channels 256, multi-scale token indices
5/11/17/23), CrossEntropy ignoring -1, AdamW lr 5e-5 / weight decay 0.05 with
CosineAnnealingLR, Resize 224 plus horizontal/vertical flips, batch 16,
precision 16-mixed, 50 epochs (TerraTorch 1.2.x no longer accepts
``decoder_scale_modules``; everything else matches the sample YAML).

Differences from the sample YAML are recorded in the run notes. Evaluation
uses the project-wide confusion-matrix implementation, not TerraTorch's own
metric stack, so every model is scored by the same code.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
from rs_backbone_eval.evaluation.segmentation import binary_metrics
from rs_backbone_eval.worker_runtime import RunWriter

BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
DATASET_VERSION_DIR = Path("raw") / "sen1floods11" / "v1.1"
TOKEN_INDICES = [5, 11, 17, 23]
SPLIT_FILES = {
    "train": ("flood_train_data.csv", "flood_train_data.txt"),
    "validation": ("flood_valid_data.csv", "flood_valid_data.txt"),
    "test": ("flood_test_data.csv", "flood_test_data.txt"),
}
OFFICIAL_EPOCHS = 50
OFFICIAL_BATCH_SIZE = 16


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


def prepare_split_txts(data_root: Path) -> Path:
    """Convert official split CSVs to the .txt chip-ID lists TerraTorch reads.

    TerraTorch filters both S2Hand and LabelHand files by substring, so the
    txt lines must be bare chip IDs (e.g. ``Ghana_103272``). Idempotent.
    """
    splits_dir = data_root / DATASET_VERSION_DIR / "splits" / "flood_handlabeled"
    for csv_name, txt_name in SPLIT_FILES.values():
        csv_path = splits_dir / csv_name
        txt_path = splits_dir / txt_name
        if not csv_path.is_file():
            raise FileNotFoundError(f"missing official split CSV: {csv_path}")
        chips = [
            line.split(",", 1)[0].removesuffix("_S1Hand.tif").strip()
            for line in csv_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        payload = "\n".join(chips) + "\n"
        if not txt_path.exists() or txt_path.read_text(encoding="utf-8") != payload:
            txt_path.write_text(payload, encoding="utf-8")
    return data_root / DATASET_VERSION_DIR.parent


def build_datamodule(
    dataset_root: Path,
    *,
    batch_size: int,
    num_workers: int,
    protocol: str = "prithvi_repo_config",
):
    import torch
    from albumentations import CenterCrop, HorizontalFlip, RandomCrop, Resize, VerticalFlip
    from albumentations.pytorch import ToTensorV2
    from terratorch.datamodules import Sen1Floods11NonGeoDataModule
    from terratorch.datamodules.sen1floods11 import MEANS, STDS

    class NormalizedSen1Floods11DataModule(Sen1Floods11NonGeoDataModule):
        """Apply the band statistics the upstream module defines but never uses.

        TerraTorch 1.2.13 creates ``self.aug = K.Normalize(MEANS, STDS)`` in
        ``Sen1Floods11NonGeoDataModule`` but never applies it. The official
        sample config was written against versions where this normalization
        ran, so we restore it via the Lightning batch-transfer hook and use
        the same method during test evaluation.
        """

        val_test_batch_size: int = 16

        def on_after_batch_transfer(self, batch, dataloader_idx: int = 0):
            means = torch.as_tensor(
                [MEANS[b] for b in self.bands],
                dtype=batch["image"].dtype,
                device=batch["image"].device,
            ).view(-1, 1, 1)
            stds = torch.as_tensor(
                [STDS[b] for b in self.bands],
                dtype=batch["image"].dtype,
                device=batch["image"].device,
            ).view(-1, 1, 1)
            batch["image"] = (batch["image"] - means) / stds
            return batch

        def _dataloader_factory(self, split: str):
            """Use a smaller batch for full-resolution validation/testing."""
            from torch.utils.data import DataLoader

            dataset = self._valid_attribute(f"{split}_dataset", "dataset")
            batch = self._valid_attribute(f"{split}_batch_size", "batch_size")
            if split in ("val", "test"):
                batch = min(batch, self.val_test_batch_size)
            return DataLoader(
                dataset=dataset,
                batch_size=batch,
                shuffle=split == "train",
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                drop_last=split == "train" and self.drop_last,
            )

    if protocol == "prithvi_paper_protocol":
        # Paper recipe: 512 chips cropped to 448, then random 224 crops for
        # training; validation and testing run at the full 512 resolution.
        train_transform = [
            CenterCrop(448, 448),
            RandomCrop(224, 224),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ToTensorV2(),
        ]
        val_transform = [ToTensorV2()]
        test_transform = [ToTensorV2()]
        val_test_batch_size = 4
    else:
        resize = Resize(224, 224)
        train_transform = [resize, HorizontalFlip(p=0.5), VerticalFlip(p=0.5), ToTensorV2()]
        val_transform = [resize, ToTensorV2()]
        test_transform = [resize, ToTensorV2()]
        val_test_batch_size = batch_size

    module = NormalizedSen1Floods11DataModule(
        data_root=str(dataset_root),
        batch_size=batch_size,
        num_workers=num_workers,
        constant_scale=0.0001,
        no_data_replace=0,
        no_label_replace=-1,
        use_metadata=False,
        bands=BANDS,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
    )
    module.val_test_batch_size = val_test_batch_size
    return module


def build_task(checkpoint: Path, *, epochs: int, lr: float = 5.0e-5):
    from terratorch.tasks import SemanticSegmentationTask

    return SemanticSegmentationTask(
        model_args={
            "backbone": "prithvi_eo_v2_300_tl",
            "backbone_kwargs": {
                "pretrained": True,
                "ckpt_path": str(checkpoint),
                "bands": BANDS,
            },
            "decoder": "UperNetDecoder",
            "decoder_kwargs": {"channels": 256},
            "num_classes": 2,
            "rescale": True,
            "head_kwargs": {"dropout": 0.1},
            "necks": [
                {"name": "SelectIndices", "indices": TOKEN_INDICES},
                {"name": "ReshapeTokensToImage"},
            ],
        },
        model_factory="EncoderDecoderFactory",
        loss="ce",
        ignore_index=-1,
        lr=lr,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 0.05},
        scheduler="CosineAnnealingLR",
        scheduler_hparams={"T_max": epochs},
    )


def _param_counts(task):
    trainable = sum(p.numel() for p in task.parameters() if p.requires_grad)
    total = sum(p.numel() for p in task.parameters())
    return trainable, total


def _run_test_evaluation(task, datamodule, device: str) -> dict[str, float]:
    """Score the trained model on the test split with the shared metric layer."""
    import torch

    task.eval()
    task.to(device)
    datamodule.setup("test")
    matrix = ConfusionMatrix(num_classes=2)
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            batch = {key: value.to(device) for key, value in batch.items()}
            batch = datamodule.on_after_batch_transfer(batch)
            logits = task.model(batch["image"]).output
            predictions = logits.argmax(dim=1)
            matrix.update(batch["mask"].cpu().numpy(), predictions.cpu().numpy(), ignore_index=-1)
    metrics = binary_metrics(matrix, foreground=1)
    metrics["water_iou"] = metrics.pop("foreground_iou")
    return metrics


def train_baseline(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    checkpoint: Path,
) -> int:
    import lightning as L
    import torch
    from lightning.pytorch.callbacks import ModelCheckpoint

    run_id = request.run_id
    data_root = Path(request.data_root)
    run_dir = Path(request.runs_root) / run_id

    from rs_backbone_eval.registry import ProtocolSpec, load_yaml_spec

    protocol_id = load_yaml_spec(Path(request.protocol_config_path), ProtocolSpec).protocol_id

    epochs = _int_env("PRITHVI_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("PRITHVI_WORKER_BATCH_SIZE", OFFICIAL_BATCH_SIZE)
    num_workers = _int_env("PRITHVI_WORKER_NUM_WORKERS", 8)
    lr = _float_env("PRITHVI_WORKER_LR", 5.0e-5)
    limit_batches = _int_env("PRITHVI_WORKER_LIMIT_BATCHES", 0) or None
    smoke = epochs != OFFICIAL_EPOCHS or limit_batches is not None

    emit(run_id, "train.prepare", "converting split CSVs for TerraTorch datamodule")
    dataset_root = prepare_split_txts(data_root)
    datamodule = build_datamodule(
        dataset_root, batch_size=batch_size, num_workers=num_workers, protocol=protocol_id
    )
    emit(
        run_id,
        "train.prepare",
        f"datamodule ready (protocol {protocol_id}, batch {batch_size}, workers {num_workers})",
    )

    task = build_task(checkpoint, epochs=epochs, lr=lr)
    trainable, total = _param_counts(task)
    writer.update_training(
        head_id="upernet_decoder",
        trainable_parameters=trainable,
        total_parameters=total,
    )
    writer.update_input(
        preprocessing_id=f"official_sen1floods11_v1.1_{protocol_id}",
    )
    writer.update(tags={"learning_rate": f"{lr:g}"})
    if protocol_id == "prithvi_paper_protocol":
        protocol_note = (
            "T1 paper-protocol reproduction (B0): 512->448 crop then random 224 crops, "
            "full-resolution 512 validation/testing, restored band normalization. "
            "Differences from the paper: single trial and single seed instead of 10 "
            "hyperparameter trials and 10 seeds."
        )
    else:
        protocol_note = (
            "T1 repo-config baseline (B0). Official sample YAML settings; "
            "decoder_scale_modules is not accepted by TerraTorch 1.2.x."
        )
    writer.update(notes=protocol_note)
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

    emit(
        run_id,
        "train.done",
        f"fit finished in {wall_time / 60:.1f} min, peak GPU {peak_memory:.0f} MB",
    )

    metrics = _run_test_evaluation(task, datamodule, "cuda")
    records = to_metric_records(
        metrics, primary_name="water_iou", direction="maximize", split="test"
    )
    writer.add_metrics(records)
    writer.update_training(
        epochs_completed=trainer.current_epoch,
        wall_time_seconds=wall_time,
        peak_gpu_memory_mb=peak_memory,
    )
    best = checkpoint_callback.best_model_path
    if best and Path(best).exists():
        writer.add_artifact(
            "checkpoint", str(Path(best).relative_to(run_dir)), "best validation checkpoint"
        )
    summary = (
        f"water_iou={metrics['water_iou']:.4f} mean_iou={metrics['mean_iou']:.4f} "
        f"({epochs} epochs, best val loss {float(checkpoint_callback.best_model_score or 0):.4f})"
    )
    if smoke:
        summary += " [SMOKE: non-official epoch/batch limits]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
