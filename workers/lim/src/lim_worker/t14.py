"""T14 (SpaceNet 7) building-change training: frozen LIM siamese baseline.

Each sample is one AOI with two consecutive monthly Planet basemaps (t1, t2) and
a uint8 change mask (1 = building at t2 and not at t1, 255 = cloud/unusable in
t1 or t2, see ``t14_labels.py``). Cloud pixels are ignored by the loss and by
every metric; they are never a class.
The LIM MiT-Huge encoder is frozen and run live in fp16 on both dates (no
feature cache). For each of the four pyramid levels, ``|f2 - f1|`` feeds the
shared ``UPerNetHead`` (GroupNorm, fpn 256) with two classes. Only the head
trains. This is protocol ``t14_b0_wce``.

Training uses fixed 512 px crops of the native grid. Crop offsets and flips come
from one generator per (seed, epoch, index), and the same crop and flips apply
to both dates and the mask. Evaluation runs on whole basemaps, one at a time,
because tile sizes differ by a row or column (1024 x 1023 occurs).

Scores are binary, foreground = class 1: ``change_f1`` (primary), ``change_iou``
and ``mean_iou`` (mean of the class IoUs that occur).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from itertools import islice
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.data.manifest import require_no_group_leakage
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.confusion import GroupConfusion
from rs_backbone_eval.evaluation.segmentation import binary_metrics
from rs_backbone_eval.heads import UPerNetHead
from rs_backbone_eval.matrix import stopped_short_of_budget
from rs_backbone_eval.normalization import imagenet_rgb
from rs_backbone_eval.registry import ProtocolSpec, TaskSpec
from rs_backbone_eval.schedule import build_scheduler
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.worker_runtime import RunWriter

from .arms import safe_cross_entropy
from .mit import load_encoder, mit_huge

TASK_ID = "T14"
PROTOCOL_ID = "t14_b0_wce"
CHANNELS = (256, 512, 1024, 2048)
NUM_CLASSES = 2
FOREGROUND = 1
PRIMARY = "change_f1"
CROP = 512
# Change-class weight in the loss. sqrt(neg/pos) on train pixels: 1,072,802,369 / 945,745
# gives 1134 and its square root is 33.7. Full balancing (1134) is too extreme for CE.
CHANGE_WEIGHT = 34.0
IGNORE_INDEX = 255  # cloud/unusable pixels; must equal task_spec.ignore_index
EFFECTIVE_BATCH = 8  # same as T13 (t13.py); AGENTS.md batch rules apply
PREPROCESSING_ID = "t14_imagenet_rgb_siamese_abs_diff"


def load_rows(manifest: Path, split: str) -> list[dict]:
    """T14 rows of ``split``; rows of other tasks in a shared manifest are skipped."""
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_id") == TASK_ID and row.get("split") == split:
                validate_row(row)
                rows.append(row)
    return rows


def validate_row(row: dict) -> None:
    """Reject rows whose shape does not match the change-pair layout."""
    sample = row.get("sample_id", "?")
    if len(row.get("image_paths", [])) != 2:
        raise ValueError(f"{sample}: T14 needs two image paths (t1, t2)")
    if len(row.get("timestamps", [])) != 2:
        raise ValueError(f"{sample}: T14 needs two timestamps")
    window = row.get("window", [])
    if len(window) != 4:
        raise ValueError(f"{sample}: window must be [x, y, w, h]")
    if not row.get("label_path"):
        raise ValueError(f"{sample}: T14 row has no label_path")


def _window(row: dict) -> Window:
    x, y, w, h = (int(v) for v in row["window"])
    return Window(x, y, w, h)


def _read_rgb(path: Path, window: Window) -> np.ndarray:
    """``[3, H, W]`` uint8 (R, G, B). The alpha band is not read."""
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f"{path}: expected at least 3 bands, got {src.count}")
        return src.read(indexes=[1, 2, 3], window=window)


def read_pair(data_root: Path, row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(t1 [3,H,W] uint8, t2 [3,H,W] uint8, change [H,W] int64 in {0, 1})``."""
    window = _window(row)
    t1 = _read_rgb(data_root / row["image_paths"][0], window)
    t2 = _read_rgb(data_root / row["image_paths"][1], window)
    with rasterio.open(data_root / row["label_path"]) as src:
        mask = src.read(1, window=window)
    if t1.shape[1:] != mask.shape or t2.shape[1:] != mask.shape:
        raise ValueError(
            f"{row['sample_id']}: image {t1.shape[1:]} / {t2.shape[1:]} "
            f"and mask {mask.shape} differ"
        )
    if not np.isin(mask, (0, 1, IGNORE_INDEX)).all():
        raise ValueError(f"{row['sample_id']}: change mask values outside {{0, 1, 255}}")
    return t1, t2, mask.astype(np.int64)


def _flip(arrays: tuple[np.ndarray, ...], horizontal: bool, vertical: bool) -> tuple:
    """Flip every array (last two axes are H, W) with the same flips."""
    out = []
    for array in arrays:
        if horizontal:
            array = array[..., ::-1]
        if vertical:
            array = array[..., ::-1, :]
        out.append(array)
    return tuple(out)


def crop_and_flip(
    t1: np.ndarray, t2: np.ndarray, mask: np.ndarray, rng: np.random.Generator, crop: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One random ``crop`` window and h/v flips, drawn once and shared by all three."""
    height, width = mask.shape
    if height < crop or width < crop:
        raise ValueError(f"tile {height}x{width} is smaller than the {crop} px crop")
    top = int(rng.integers(height - crop + 1))
    left = int(rng.integers(width - crop + 1))
    horizontal, vertical = bool(rng.integers(2)), bool(rng.integers(2))
    cut = (
        t1[:, top : top + crop, left : left + crop],
        t2[:, top : top + crop, left : left + crop],
        mask[top : top + crop, left : left + crop],
    )
    return _flip(cut, horizontal, vertical)


class ChangePairs(Dataset):
    """Training pairs: fixed-size crops with flips, drawn per (seed, epoch, index)."""

    def __init__(self, rows: list[dict], data_root: Path, *, crop: int, seed: int = 0):
        self.rows = rows
        self.data_root = data_root
        self.crop = crop
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        # The loader must not use persistent workers: they keep the dataset copy
        # made at the first epoch, so the epoch would never reach them.
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        t1, t2, mask = read_pair(self.data_root, self.rows[index])
        rng = np.random.default_rng((self.seed, self.epoch, index))
        t1, t2, mask = crop_and_flip(t1, t2, mask, rng, self.crop)
        return {
            "t1": torch.from_numpy(np.ascontiguousarray(t1)),
            "t2": torch.from_numpy(np.ascontiguousarray(t2)),
            "label": torch.from_numpy(np.ascontiguousarray(mask)),
            "index": index,
        }


def change_features(feats1: list[torch.Tensor], feats2: list[torch.Tensor]) -> list[torch.Tensor]:
    """Per-level absolute feature difference ``|f2 - f1|``."""
    if len(feats1) != len(feats2):
        raise ValueError(f"pyramids differ in depth: {len(feats1)} vs {len(feats2)}")
    for a, b in zip(feats1, feats2, strict=True):
        if a.shape != b.shape:
            raise ValueError(f"pyramid levels differ: {tuple(a.shape)} vs {tuple(b.shape)}")
    return [(b - a).abs() for a, b in zip(feats1, feats2, strict=True)]


class ChangeHead(nn.Module):
    """Shared UPerNet over the change pyramid. The encoder is not a submodule."""

    def __init__(self, channels: tuple[int, ...] = CHANNELS, fpn_dim: int = 256):
        super().__init__()
        self.head = UPerNetHead(list(channels), num_classes=NUM_CLASSES, fpn_dim=fpn_dim)

    def forward(
        self,
        feats1: list[torch.Tensor],
        feats2: list[torch.Tensor],
        output_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        return self.head(change_features(feats1, feats2), output_size=output_size)


class Normalisation:
    """ImageNet RGB statistics on the 0-255 scale, as LIM was pretrained."""

    def __init__(self, device: str, dtype: torch.dtype):
        mean, std = imagenet_rgb()
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.device = device
        self.dtype = dtype


@torch.no_grad()
def encode(encoder: nn.Module, image_u8: torch.Tensor, norm: Normalisation) -> list[torch.Tensor]:
    """Frozen LIM pyramid for a uint8 RGB batch, computed in ``norm.dtype``."""
    x = (image_u8.to(norm.device).float() - norm.mean) / norm.std
    return [level.float() for level in encoder(x.to(norm.dtype))]


def change_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int | None,
    class_weight: float | None = None,
) -> torch.Tensor:
    """Cross-entropy. ``None`` means no pixel is ignored (every pixel counts).

    ``class_weight`` multiplies the change class (1); the background keeps weight 1.
    Without it the change pixels (about 0.09 % of train pixels) contribute too little
    gradient and the head predicts "no change" everywhere (run
    lim_mit_huge-T14-t14_b0-train-17, change_f1 0.0 at every validation epoch).
    """
    weight = None
    if class_weight is not None:
        weight = torch.tensor([1.0, float(class_weight)], device=logits.device, dtype=logits.dtype)
    if ignore_index is None:
        return F.cross_entropy(logits, target, weight=weight)
    if weight is None:
        return safe_cross_entropy(logits, target, ignore_index)
    if not (target != ignore_index).any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits, target, weight=weight, ignore_index=ignore_index)


def change_scores(matrix) -> dict[str, float]:
    """``change_f1`` and ``change_iou`` for class 1, plus the mean class IoU."""
    metrics = binary_metrics(matrix, foreground=FOREGROUND)
    return {
        PRIMARY: metrics["foreground_f1"],
        "change_iou": metrics["foreground_iou"],
        "mean_iou": metrics["mean_iou"],
    }


def score_rows(
    rows: list[dict],
    data_root: Path,
    predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    limit: int = 0,
) -> tuple[dict[str, float], dict]:
    """Whole-basemap scoring, one pair at a time. Confusion is kept per AOI."""
    collector = GroupConfusion(NUM_CLASSES)
    for row in rows[:limit] if limit else rows:
        t1, t2, mask = read_pair(data_root, row)
        logits = predict(torch.from_numpy(t1[None]), torch.from_numpy(t2[None]))
        prediction = logits.argmax(dim=1)[0].cpu().numpy()
        # Cloud pixels leave TP, FP and FN alike; they are not a third class.
        collector.update(row["group_id"], mask, prediction, ignore_index=IGNORE_INDEX)
    return change_scores(collector.combined()), collector.to_json()


def probe_pair(
    encoder: nn.Module,
    row: dict,
    data_root: Path,
    device: str,
    dtype: torch.dtype,
) -> dict:
    """One T14 row through the frozen encoder on both dates. Nothing is trained.

    Reuses the training path (``read_pair``, ``Normalisation``, ``encode`` and
    ``change_features``), so the probe cannot disagree with a run about
    normalisation, dtype or which two dates form a pair.
    """
    t1, t2, mask = read_pair(data_root, row)
    norm = Normalisation(device, dtype)
    started = time.time()
    feats1 = encode(encoder, torch.from_numpy(t1[None]), norm)
    feats2 = encode(encoder, torch.from_numpy(t2[None]), norm)
    change_features(feats1, feats2)  # raises when the two pyramids disagree
    if device == "cuda":
        torch.cuda.synchronize()
    latency = time.time() - started
    height, width = mask.shape
    levels = [
        {
            "channels": int(feat.shape[1]),
            "height": int(feat.shape[-2]),
            "width": int(feat.shape[-1]),
            "stride": [round(height / feat.shape[-2], 2), round(width / feat.shape[-1], 2)],
        }
        for feat in feats1
    ]
    return {
        "sample": row["sample_id"],
        "tile_px": [height, width],
        "dtype": str(dtype).replace("torch.", ""),
        "levels": levels,
        "channels_match_protocol": [lv["channels"] for lv in levels] == list(CHANNELS),
        "finite": all(bool(torch.isfinite(f).all()) for f in [*feats1, *feats2]),
        "latency_s": round(latency, 3),
    }


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def batch_plan() -> tuple[int, int]:
    """``(micro batch, gradient accumulation steps)``; their product is EFFECTIVE_BATCH.

    The default is one micro batch of EFFECTIVE_BATCH. A smaller micro batch
    needs ``LIM_WORKER_GRAD_ACCUM`` to keep the effective batch unchanged.
    """
    micro = _int_env("LIM_WORKER_MICRO_BATCH", EFFECTIVE_BATCH)
    accumulation = _int_env("LIM_WORKER_GRAD_ACCUM", EFFECTIVE_BATCH // max(micro, 1))
    if micro * accumulation != EFFECTIVE_BATCH:
        raise ValueError(
            f"T14 needs effective batch {EFFECTIVE_BATCH}; got {micro} x {accumulation}"
        )
    return micro, accumulation


def _missing_labels(rows: list[dict], data_root: Path) -> int:
    return sum(1 for row in rows if not (data_root / row["label_path"]).is_file())


def epoch_micro_batches(loader_len: int, limit: int) -> int:
    """Micro batches one epoch runs: every batch of the loader, or ``limit`` of them when set."""
    return min(loader_len, limit) if limit else loader_len


def optimizer_steps_per_epoch(micro_batches: int, accumulation: int) -> int:
    """Optimizer updates in one epoch: one per full accumulation group.

    A trailing partial group is dropped, so its micro batches run no forward pass.
    """
    return micro_batches // accumulation


def run_train_epoch(
    loader,
    device: str,
    predict: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler,
    accumulation: int,
    limit: int,
) -> tuple[float, int]:
    """One training epoch with gradient accumulation.

    Returns ``(mean loss over micro batches, optimizer steps taken)``. Only whole
    accumulation groups run: micro batches beyond ``steps * accumulation`` get no forward
    pass and no gradient. The step count is
    ``optimizer_steps_per_epoch(epoch_micro_batches(len(loader), limit), accumulation)``.
    """
    micro_batches = epoch_micro_batches(len(loader), limit)
    planned_steps = optimizer_steps_per_epoch(micro_batches, accumulation)
    used = planned_steps * accumulation

    total = 0.0
    batches = 0
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    for batch in islice(loader, used):
        t1 = batch["t1"].to(device, non_blocking=True)
        t2 = batch["t2"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        loss = loss_fn(predict(t1, t2), labels)
        (loss / accumulation).backward()
        total += float(loss.detach())
        batches += 1
        if batches % accumulation == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            steps += 1
    return total / max(batches, 1), steps


def train_t14(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    protocol: ProtocolSpec,
    task_spec: TaskSpec,
    checkpoint: Path,
) -> int:
    if protocol.protocol_id != PROTOCOL_ID:
        raise ValueError(f"T14 does not implement protocol {protocol.protocol_id}")
    if protocol.train_backbone:
        raise ValueError("T14 implements the frozen-encoder protocol only")
    run_id, seed = request.run_id, request.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(request.data_root)
    manifest = Path(request.manifest_path)

    epochs = int(protocol.epochs or 0)
    lr = float(protocol.learning_rate or 0)
    if not epochs or not lr:
        raise ValueError(f"{PROTOCOL_ID}: T14 protocols must declare epochs and learning_rate")
    if task_spec.ignore_index != IGNORE_INDEX:
        raise ValueError(
            f"T14 config ignore_index must be {IGNORE_INDEX}, got {task_spec.ignore_index}"
        )
    micro, accumulation = batch_plan()
    limit = _int_env("LIM_WORKER_LIMIT_BATCHES", 0)
    eval_images = _int_env("LIM_WORKER_EVAL_IMAGES", 0)
    workers = _int_env("LIM_WORKER_NUM_WORKERS", 6)

    train_rows = load_rows(manifest, "train")
    val_rows = load_rows(manifest, "validation")
    test_rows = load_rows(manifest, "test")
    require_no_group_leakage(manifest, task_spec.task_id)
    if not (train_rows and val_rows and test_rows):
        raise ValueError("T14 manifest needs train, validation and test pairs")
    missing = sum(_missing_labels(rows, data_root) for rows in (train_rows, val_rows, test_rows))
    if missing:
        raise FileNotFoundError(f"{missing} T14 change masks missing under {data_root}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    encoder = mit_huge(in_chans=3)
    load_stats = load_encoder(checkpoint, encoder)
    if load_stats["missing"] or load_stats["unexpected"]:
        raise RuntimeError(f"LIM encoder did not load cleanly: {load_stats}")
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    dtype = torch.float16 if device == "cuda" else torch.float32
    if device == "cuda":
        encoder = encoder.half()
    norm = Normalisation(device, dtype)

    head = ChangeHead().to(device)
    trainable = list(head.parameters())
    tags: dict[str, str] = {
        "protocol_id": PROTOCOL_ID,
        "arm": "siamese_abs_diff",
        "encoder": "MiT-Huge frozen, fp16 live forward on both dates (no feature cache)",
        "encoder_keys_loaded": str(load_stats["loaded"]),
        "encoder_normalization": "imagenet RGB (0-255 stats), bands 1-3 (alpha not read)",
        "change_features": "|f2 - f1| per pyramid level",
        "head": f"UPerNetHead(gn, fpn 256, {NUM_CLASSES} classes)",
        "label_mapping": "1 = building at t2 and not at t1; 0 = other; 255 = cloud (t1 or t2 UDM)",
        "ignore_index": str(IGNORE_INDEX),
        "metric_classes": "0..1; primary change_f1 on class 1",
        "class_weight": f"change={CHANGE_WEIGHT:g}, background=1",
        "selection": "best_validation_change_f1, ties keep the later epoch",
        "augmentation": f"{CROP}px random crop and h/v flips shared by t1, t2, mask",
        "learning_rate": f"{lr:g}",
        "lr_schedule": protocol.lr_schedule or "constant",
        "epochs_budget": str(epochs),
        "micro_batch": str(micro),
        "grad_accum": str(accumulation),
        "effective_batch": str(EFFECTIVE_BATCH),
        "trainable_parameters": str(sum(p.numel() for p in trainable)),
    }
    writer.update(tags=tags)
    writer.update_training(
        head_id="upernet_gn",
        trainable_parameters=sum(p.numel() for p in trainable),
        total_parameters=sum(p.numel() for p in head.parameters())
        + sum(p.numel() for p in encoder.parameters()),
    )
    writer.update_input(preprocessing_id=PREPROCESSING_ID)
    emit(
        run_id,
        "train.prepare",
        f"T14 {PROTOCOL_ID}: {len(train_rows)} train / {len(val_rows)} val / "
        f"{len(test_rows)} test pairs, {tags['trainable_parameters']} trainable",
    )

    torch.multiprocessing.set_sharing_strategy("file_system")
    dataset = ChangePairs(train_rows, data_root, crop=CROP, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=micro,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        drop_last=True,
        persistent_workers=False,
    )
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    steps_per_epoch = optimizer_steps_per_epoch(
        epoch_micro_batches(len(loader), limit), accumulation
    )
    if steps_per_epoch < 1:
        raise ValueError(
            f"T14 epoch has {len(loader)} micro batches, fewer than one accumulation "
            f"group of {accumulation}; no optimizer update would run"
        )
    scheduler = (
        build_scheduler(optimizer, epochs * steps_per_epoch, steps_per_epoch)
        if protocol.lr_schedule == "warmup1-cosine"
        else None
    )
    best = BestCheckpoint()
    val_every = validation_every()
    optimizer_steps = 0
    started = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def predict(t1_u8: torch.Tensor, t2_u8: torch.Tensor) -> torch.Tensor:
        f1 = encode(encoder, t1_u8, norm)
        f2 = encode(encoder, t2_u8, norm)
        return head(f1, f2, output_size=tuple(t1_u8.shape[-2:]))

    def evaluate(rows: list[dict]) -> tuple[dict[str, float], dict]:
        head.eval()
        with torch.inference_mode():
            return score_rows(rows, data_root, predict, limit=eval_images)

    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        head.train()
        mean_loss, epoch_steps = run_train_epoch(
            loader,
            device,
            predict,
            lambda logits, labels: change_loss(logits, labels, IGNORE_INDEX, CHANGE_WEIGHT),
            optimizer,
            scheduler,
            accumulation,
            limit,
        )
        optimizer_steps += epoch_steps
        writer.add_curve("loss", "train", epoch, mean_loss)
        emit(
            run_id,
            "train.epoch",
            f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}",
            progress=(epoch + 1) / epochs,
        )
        if (epoch + 1) % val_every == 0 or epoch + 1 == epochs:
            val_scores, _ = evaluate(val_rows)
            writer.add_curve(PRIMARY, "validation", epoch, val_scores[PRIMARY])
            best.offer(val_scores[PRIMARY], epoch + 1, head, ties_go_later=True)
    train_seconds = time.time() - started

    best.restore(head)
    best.save(writer.run_dir / "best_checkpoint.pt")
    writer.add_artifact("checkpoint", "best_checkpoint.pt", "validation-selected head")

    metrics, groups = evaluate(test_rows)
    writer.add_metrics(
        to_metric_records(metrics, primary_name=PRIMARY, direction="maximize", split="test")
    )
    writer.add_confusion(groups, NUM_CLASSES)
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    writer.update_training(
        epochs_completed=epochs,
        optimizer_steps=optimizer_steps,
        wall_time_seconds=time.time() - started,
        peak_gpu_memory_mb=peak,
    )
    writer.update(
        tags={
            "selection": f"best_validation_{PRIMARY}",
            "selected_epoch": str(best.epoch),
            "validation_selection_score": f"{best.score:.4f}",
            "optimizer_steps": str(optimizer_steps),
            "train_seconds": f"{train_seconds:.1f}",
            "test_aois": str(len({r["group_id"] for r in test_rows})),
        }
    )
    summary = (
        f"{PRIMARY}={metrics[PRIMARY]:.4f} change_iou={metrics['change_iou']:.4f} "
        f"mean_iou={metrics['mean_iou']:.4f} ({epochs} epochs, best epoch {best.epoch})"
    )
    if limit or eval_images:
        summary += " [SMOKE]"
    elif stopped_short_of_budget(task_spec.task_id, epochs_completed=epochs):
        summary += " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
