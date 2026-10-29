"""T10 (FAIR1M) for the LIM worker: Rotated Faster R-CNN over MiT-Huge features.

LIM's home turf again: high-resolution aerial RGB, and the only backbone here
with a native /4../32 pyramid whose channels match a detection FPN without an
adapter. The detector is the shared one (``rs_backbone_eval.detection``), so the
only LIM-specific code is the encoder wrapper and the tile dataset.

Protocol numbers, all recorded in run tags:

* effective batch 2 = micro 1 x accumulation 2 (the detector runs one image per
  forward pass; gradient accumulation does not change activation memory);
* objects visible under 0.7 of their area are ignored, not background;
* evaluation is per whole scene: tile detections are shifted into image
  coordinates and merged with rotated NMS before COCO AP is computed;
* the budget counts optimizer steps, like T9, because a 12-epoch detection run is
  hours long and the epoch count alone says nothing about how much it saw.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.data.isaid_io import read_rgb
from rs_backbone_eval.data.tiling import infer_stride, pad_tile, valid_rect
from rs_backbone_eval.detection.ap import Predictions, Targets, evaluate
from rs_backbone_eval.detection.classes import FAIR1M_ID_TO_CLASS, class_id
from rs_backbone_eval.detection.detector import GroundTruth, RotatedFasterRCNN
from rs_backbone_eval.detection.geometry import Instance, poly2obb, visible_fraction
from rs_backbone_eval.detection.nms import Detection, clip_polygon, merge_tiles
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.matrix import stopped_short_of_budget
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.worker_runtime import RunWriter

from .mit import MixMIMEncoder, load_encoder, mit_huge

# The detector consumes one image per forward pass, so the protocol's effective
# batch is expressed as micro 1 x accumulation 2 and enforced here (plan §7.2).
EFFECTIVE_BATCH = 2
MICRO_BATCH = 1
DEFAULT_KEEP_RATIO = 0.7


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


def load_rows(manifest_path: Path, split: str) -> list[dict]:
    rows: list[dict] = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row["split"] == split:
                    rows.append(row)
    return rows


def tile_geometry(rows: list[dict]) -> tuple[int, int]:
    """``(crop, stride)`` from the manifest, so a run matches its build."""
    crop = int(rows[0]["window"][2])
    groups: dict[str, set[int]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], set()).add(int(row["window"][0]))
    stride = infer_stride(groups)
    return crop, (stride if stride else crop // 2)


class AnnotationCache:
    """Per-image objects, read once per run."""

    def __init__(self, data_root: Path, keep_ratio: float) -> None:
        self.data_root = data_root
        self.keep_ratio = keep_ratio
        self._cache: dict[str, dict] = {}

    def get(self, relative: str) -> dict:
        if relative not in self._cache:
            self._cache[relative] = json.loads(
                (self.data_root / relative).read_text(encoding="utf-8")
            )
        return self._cache[relative]

    def instances(self, relative: str) -> list[Instance]:
        payload = self.get(relative)
        return [
            Instance(
                class_name=item["class_name"],
                points=tuple((float(x), float(y)) for x, y in item["points"]),
            )
            for item in payload["instances"]
        ]

    def split_tile(
        self, relative: str, window: list[int]
    ) -> tuple[list[tuple[np.ndarray, int]], list[np.ndarray]]:
        """Kept and ignored instance polygons for one tile window."""
        payload = self.get(relative)
        x, y, width, height = (int(value) for value in window)
        kept: list[tuple[np.ndarray, int]] = []
        ignored: list[np.ndarray] = []
        for item in payload["instances"]:
            polygon = np.asarray(item["points"], dtype=np.float64)
            overlapping = (
                polygon[:, 0].max() > x
                and polygon[:, 0].min() < x + width
                and polygon[:, 1].max() > y
                and polygon[:, 1].min() < y + height
            )
            if not overlapping:
                continue
            fraction = visible_fraction(polygon, x, y, x + width, y + height)
            local = polygon - np.array([x, y], dtype=np.float64)
            if fraction >= self.keep_ratio:
                kept.append((local, int(item["class_id"])))
            else:
                ignored.append(local)
        return kept, ignored


class Fair1mTiles(Dataset):
    """One training tile: image crop, kept rotated boxes, ignored boxes."""

    def __init__(
        self,
        rows: list[dict],
        data_root: Path,
        crop: int,
        mean: np.ndarray,
        std: np.ndarray,
        annotations: AnnotationCache,
        *,
        augment: bool = True,
    ) -> None:
        self.rows = rows
        self.data_root = data_root
        self.crop = crop
        self.mean = mean.reshape(1, 1, 3)
        self.std = std.reshape(1, 1, 3)
        self.annotations = annotations
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        window = [int(value) for value in row["window"]]
        image = read_rgb(self.data_root / row["image_paths"][0], tuple(window)).astype(np.float32)
        kept, ignored = self.annotations.split_tile(row["annotation_path"], window)

        # `read_rgb` pads out-of-bounds reads, so the array is always crop-sized;
        # the real block comes from the annotation's image size, which is exact
        # and already in memory. Cross-check it against the manifest's
        # valid_fraction so a stale annotation cannot silently shift the region.
        payload = self.annotations.get(row["annotation_path"])
        real_w = min(self.crop, int(payload["width"]) - window[0])
        real_h = min(self.crop, int(payload["height"]) - window[1])
        expected = real_w * real_h / float(self.crop * self.crop)
        if abs(expected - float(row["valid_fraction"])) > 1e-3:
            raise ValueError(
                f"{row['sample_id']}: annotation size {payload['width']}x{payload['height']} "
                f"gives valid_fraction {expected:.4f}, manifest says {row['valid_fraction']}"
            )
        flip_x = self.augment and np.random.random() < 0.5
        flip_y = self.augment and np.random.random() < 0.5
        if flip_x:
            image = image[:, ::-1].copy()
        if flip_y:
            image = image[::-1].copy()
        image = pad_tile((image - self.mean) / self.std, self.crop, fill=0.0)
        # where the real pixels ended up after padding and flips; the detector
        # uses it to keep padded anchors out of the negative sample
        region = valid_rect(self.crop, real_w, real_h, flip_x=flip_x, flip_y=flip_y)

        def transform(polygon: np.ndarray) -> np.ndarray:
            values = polygon.copy()
            if flip_x:
                values[:, 0] = self.crop - values[:, 0]
            if flip_y:
                values[:, 1] = self.crop - values[:, 1]
            return values

        boxes = np.array([poly2obb(transform(polygon)) for polygon, _ in kept], dtype=np.float32)
        labels = np.array([label for _, label in kept], dtype=np.int64)
        ignored_boxes = np.array(
            [poly2obb(transform(polygon)) for polygon in ignored], dtype=np.float32
        )
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
            "boxes": torch.from_numpy(boxes).reshape(-1, 5),
            "labels": torch.from_numpy(labels),
            "ignored_boxes": torch.from_numpy(ignored_boxes).reshape(-1, 5),
            "valid": torch.tensor(region, dtype=torch.long),
            "group": row["group_id"],
        }


def _norm_statistics(
    rows: list[dict], data_root: Path, sample: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(rows), size=min(sample, len(rows)), replace=False)
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    count = 0
    for index in picks:
        row = rows[int(index)]
        window = tuple(int(value) for value in row["window"])
        image = read_rgb(data_root / row["image_paths"][0], window).astype(np.float64)
        if image.size == 0:
            continue
        flat = image.reshape(-1, image.shape[-1])
        total += flat.sum(axis=0)
        total_sq += (flat**2).sum(axis=0)
        count += image.shape[0] * image.shape[1]
    if not count:
        raise ValueError("no training pixels to compute normalisation statistics")
    mean = total / count
    variance = np.maximum(total_sq / count - mean**2, 1e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


class LimDetector(nn.Module):
    """MiT-Huge encoder (frozen or not) + the shared rotated detector."""

    def __init__(self, encoder: MixMIMEncoder, num_classes: int, fpn_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.detector = RotatedFasterRCNN(
            encoder, list(encoder.embed_dims), num_classes=num_classes, fpn_dim=fpn_dim
        )


def train_t10(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    checkpoint: Path,
    freeze_backbone: bool,
    num_classes: int,
    task_id: str,
    metric_name: str = "map",
) -> int:
    run_id = request.run_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accumulation = _int_env("LIM_WORKER_GRAD_ACCUM", EFFECTIVE_BATCH)
    if MICRO_BATCH * accumulation != EFFECTIVE_BATCH:
        emit(
            run_id,
            "train.batch",
            f"effective batch {MICRO_BATCH}x{accumulation} != {EFFECTIVE_BATCH}",
            level="error",
        )
        writer.fail(
            f"T10 requires an effective batch of {EFFECTIVE_BATCH} (micro 1 x accum 2); "
            f"got {MICRO_BATCH} x {accumulation}"
        )
        return 1
    epochs = _int_env("LIM_WORKER_EPOCHS", 12)
    lr = _float_env("LIM_WORKER_LR", 1e-4)
    weight_decay = _float_env("LIM_WORKER_WEIGHT_DECAY", 0.05)
    warmup = _int_env("LIM_WORKER_WARMUP_STEPS", 500)
    workers = _int_env("LIM_WORKER_NUM_WORKERS", 4)
    limit = _int_env("LIM_WORKER_LIMIT_BATCHES", 0) or None
    total_steps = _int_env("LIM_WORKER_TOTAL_STEPS", 0)
    eval_limit = _int_env("LIM_WORKER_EVAL_IMAGES", 0)
    norm_samples = _int_env("LIM_WORKER_NORM_SAMPLES", 64)
    fpn_dim = _int_env("LIM_WORKER_FPN_DIM", 256)
    keep_ratio = _float_env("LIM_WORKER_KEEP_RATIO", DEFAULT_KEEP_RATIO)

    data_root = Path(request.data_root)
    manifest = Path(request.manifest_path)
    train_rows = load_rows(manifest, "train")
    val_rows = load_rows(manifest, "validation")
    test_rows = load_rows(manifest, "test")
    if not train_rows or not test_rows:
        emit(run_id, "train.manifest", "manifest has no train/test tiles", level="error")
        writer.fail("manifest has no train/test tiles; run rs-eval build-manifest --task T10")
        return 1
    if not freeze_backbone and not val_rows:
        writer.fail(
            "rotated_finetune needs a validation split: run rs-eval holdout-manifest "
            "--fraction 0.2 --seed 17 --split validation"
        )
        return 1
    crop, stride = tile_geometry(train_rows)
    annotations = AnnotationCache(data_root, keep_ratio)
    mean, std = _norm_statistics(train_rows, data_root, norm_samples, request.seed)
    means = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    stds = torch.tensor(std, device=device).view(1, 3, 1, 1)

    encoder = mit_huge(in_chans=3, drop_path_rate=0.1 if not freeze_backbone else 0.0)
    if random_init_requested():
        load_stats = {"loaded": 0, "missing": 0, "unexpected": 0, "dropped": 0}
    else:
        load_stats = load_encoder(checkpoint, encoder)
        if load_stats["missing"] or load_stats["unexpected"]:
            emit(run_id, "train.checkpoint", f"encoder load mismatch: {load_stats}", level="error")
            writer.fail(f"LIM encoder did not load cleanly: {load_stats}")
            return 1
    model = LimDetector(encoder, num_classes, fpn_dim=fpn_dim).to(device)
    detector = model.detector
    if freeze_backbone:
        for parameter in detector.backbone.parameters():
            parameter.requires_grad = False

    head_parameters = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    writer.update_training(
        head_id="rotated_frcnn_gn",
        trainable_parameters=head_parameters,
        total_parameters=sum(p.numel() for p in model.parameters()),
    )
    writer.update_input(preprocessing_id="fair1m_rgb_train_stats")
    writer.update(
        tags={
            "head_id": "rotated_frcnn_gn",
            "head_parameters": str(head_parameters),
            "norm_layer": "GroupNorm",
            "backbone_frozen": str(freeze_backbone),
            "encoder_params": str(sum(p.numel() for p in encoder.parameters())),
            "encoder_load": str(load_stats),
            "normalization": (
                f"train-tile mean {mean.round(4).tolist()} std {std.round(4).tolist()}"
            ),
            "normalization_samples": str(norm_samples),
            "learning_rate": f"{lr:g}",
            "weight_decay": f"{weight_decay:g}",
            "warmup_steps": str(warmup),
            "micro_batch": str(MICRO_BATCH),
            "grad_accum": str(accumulation),
            "effective_batch": str(EFFECTIVE_BATCH),
            "train_crop_px": str(crop),
            "tile_stride_px": str(stride),
            "test_protocol": (
                "per-tile inference, shifted to image coordinates, rotated NMS, COCO AP"
            ),
            "box_convention": "le90 (w = long edge, theta in (-pi/2, pi/2])",
            "keep_ratio": f"{keep_ratio:g}",
            "ignore_policy": "objects below keep_ratio are ignored in RPN and RoI sampling",
            "anchor_sizes": "32,64,128,256",
            "anchor_ratios": "0.5,1,2",
            "ap_convention": "COCO 101-point, IoU 0.5:0.05:0.95, area 32^2/96^2 px^2",
            "device": device,
            "gsd_source": "declared",
            "gsd_m": f"{_float_env('RS_EVAL_FAIR1M_GSD', 0.5):g}",
            "detector": "Rotated Faster R-CNN (horizontal proposals, rotated regression)",
        }
    )
    emit(
        run_id,
        "train.prepare",
        f"LIM T10 ({'frozen' if freeze_backbone else 'fine-tuned'}): "
        f"{len(train_rows)} train tiles, {len(val_rows)} validation, {len(test_rows)} test, "
        f"crop {crop} stride {stride}, "
        f"{head_parameters} trainable detector params, {num_classes} classes",
    )

    train_ds = Fair1mTiles(train_rows, data_root, crop, mean, std, annotations)
    train_loader = DataLoader(
        train_ds,
        batch_size=MICRO_BATCH,
        shuffle=True,
        num_workers=workers,
        drop_last=False,
        persistent_workers=workers > 0,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    eval_stats = {"seconds": 0.0, "images": 0, "tiles": 0}

    def score(rows: list[dict]) -> dict[str, object]:
        started_eval = time.time()
        groups: dict[str, list[dict]] = {}
        for row in rows:
            groups.setdefault(row["group_id"], []).append(row)
        keys = sorted(groups)
        if eval_limit:
            keys = keys[:eval_limit]
        model.eval()
        predictions: dict[str, Predictions] = {}
        targets: dict[str, Targets] = {}
        with torch.inference_mode():
            for key in keys:
                group = sorted(groups[key], key=lambda row: (row["window"][1], row["window"][0]))
                tile_outputs: list[tuple[int, int, list[Detection]]] = []
                for row in group:
                    window = [int(value) for value in row["window"]]
                    # real pixels of this tile: the padded band is never a target
                    # or a background example in training, so a box centred there
                    # is dropped rather than counted as a false positive
                    payload = annotations.get(row["annotation_path"])
                    valid_w = min(crop, int(payload["width"]) - window[0])
                    valid_h = min(crop, int(payload["height"]) - window[1])
                    pixels = read_rgb(data_root / row["image_paths"][0], tuple(window)).astype(
                        np.float32
                    )
                    tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
                    tensor = (tensor[None].to(device) - means) / stds
                    output = detector.detections(detector.pyramid(tensor), (crop, crop))[0]
                    detections = []
                    for polygon, value, label in zip(
                        output["polygons"].cpu().numpy(),
                        output["scores"].cpu().tolist(),
                        output["labels"].cpu().tolist(),
                        strict=True,
                    ):
                        clipped = clip_polygon(polygon, valid_w, valid_h)
                        if clipped is None:
                            continue
                        detections.append(
                            Detection(polygon=clipped, score=float(value), class_id=int(label))
                        )
                    tile_outputs.append((window[0], window[1], detections))
                    eval_stats["tiles"] += 1
                merged = merge_tiles(tile_outputs, iou_threshold=0.5)
                scene = Predictions()
                for detection in merged:
                    scene.append(detection.polygon, detection.class_id, detection.score)
                predictions[key] = scene

                target = Targets()
                for instance in annotations.instances(group[0]["annotation_path"]):
                    target.append(instance.polygon, class_id(instance.class_name))
                targets[key] = target
        eval_stats["seconds"] += time.time() - started_eval
        eval_stats["images"] += len(keys)
        metrics = evaluate(predictions, targets, num_classes)
        return metrics

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
        total_loss, steps = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            labels = batch["labels"][0]
            boxes = batch["boxes"][0]
            ignored = batch["ignored_boxes"][0]
            valid = batch["valid"][0].tolist()
            pixels = batch["image"].to(device)
            target = GroundTruth(
                boxes=boxes.to(device),
                labels=labels.to(device),
                ignored_boxes=ignored.to(device),
                valid_region=(
                    float(valid[0]),
                    float(valid[1]),
                    float(valid[0] + valid[2]),
                    float(valid[1] + valid[3]),
                ),
            )
            losses = detector.losses(detector.pyramid(pixels), [target])
            (losses["loss"] / accumulation).backward()
            total_loss += float(losses["loss"].detach())
            steps += 1
            train_tiles += 1
            if steps % accumulation == 0:
                if warmup and optimizer_steps < warmup:
                    rate = (optimizer_steps + 1) / max(warmup, 1)
                    for group in optimizer.param_groups:
                        group["lr"] = lr * rate
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if total_steps and optimizer_steps >= total_steps:
                    budget_hit = True
                    break
            if limit and steps >= limit:
                break
        if steps % accumulation:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        epoch_seconds = time.time() - epoch_started
        train_seconds += epoch_seconds
        if not budget_hit:
            completed_epochs = epoch + 1
        mean_loss = total_loss / max(steps, 1)
        writer.add_curve("loss", "train", epoch, mean_loss)
        emit(
            run_id,
            "train.epoch",
            f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f} steps={optimizer_steps}",
            progress=(epoch + 1) / epochs,
        )
        if budget_hit:
            break
        if select and ((epoch + 1) % val_every == 0 or epoch + 1 == epochs):
            validation = score(val_rows)
            writer.add_curve("map", "validation", epoch, validation["map"])
            best.offer(validation["map"], epoch + 1, model)
    wall = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0

    if select:
        best.restore(model)
        best.save(writer.run_dir / "best_checkpoint.pt")
        writer.add_artifact("checkpoint", "best_checkpoint.pt", "validation-selected weights")
        writer.update(
            tags={
                "selection": "best_validation_map",
                "selected_epoch": str(best.epoch),
                "validation_map": f"{best.score:.4f}",
            }
        )
    emit(run_id, "train.eval", "evaluating the test split (per-tile inference + rotated NMS)")
    metrics = score(test_rows)
    flat_metrics: dict[str, float] = {
        "map": float(metrics["map"]),
        "map_50": float(metrics["map_50"]),
        "map_75": float(metrics["map_75"]),
        "recall_small": float(metrics["recall_small"]),
        "recall_medium": float(metrics["recall_medium"]),
        "recall_large": float(metrics["recall_large"]),
    }
    # `class_id` is also the module-level helper the nested score() calls; using
    # it as a loop variable here made it a local of train_t10 and shadowed the
    # import for every nested function (NameError at the first evaluation).
    for klass, value in dict(metrics["per_class_ap50"]).items():
        flat_metrics[f"ap50/{FAIR1M_ID_TO_CLASS[int(klass)]}"] = float(value)
    writer.add_metrics(
        to_metric_records(
            flat_metrics, primary_name=metric_name, direction="maximize", split="test"
        )
    )
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
            "classes_with_targets": str(metrics["classes_with_targets"]),
        }
    )
    summary = (
        f"map={metrics['map']:.4f} map_50={metrics['map_50']:.4f} map_75={metrics['map_75']:.4f} "
        f"({completed_epochs} epochs / {optimizer_steps} steps, "
        f"{head_parameters} trainable detector params, "
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
