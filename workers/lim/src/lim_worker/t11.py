"""LIM on PASTIS-HD single / median variants for T11 (registry runs).

Two declared variants without a temporal head (the 20-bin + LTAE2d track is
implemented in ``lim_worker.t11_ltae``):

* ``lim_s2_single`` - one acquisition per patch, the frame whose date is
  closest to the year-mid anchor 2019-07-15 (PASTIS, R=G=B = B04/B03/B02);
* ``lim_s2_median`` - a per-pixel temporal median over every frame of the
  patch (R/G/B only), the parameter-free yearly composite.

Both run the frozen MiT-Huge encoder plus the shared
``rs_backbone_eval.heads.UPerNetHead`` (fpn 256) exactly like the LIM T9 lane.
Inputs come from the contract manifest's ``aux_paths.s2_path`` (the AEF grid
stays the row payload: ``image_paths[0]``); RGB bands are scaled DN/10000*255
and the effective input is z-scored by the train-fold tile-balanced statistics
(the ImageNet affine pass cancels under z-scoring, see ``t11_common.norm_tags``
-- NORM_DECISION a808302 is NOT effective here, declared T11 exception,
review 2026-11-04). Runs on CUDA when available and records device/host in
the tags (review 2026-11-04). Budget mirrors the AEF track: 8 epochs on the
full Fold 1-3, AdamW lr 1e-3, effective batch 8, per-seed epoch shuffles
(seed*1000+epoch).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lim_worker.mit import load_encoder, mit_huge
else:
    from .mit import load_encoder, mit_huge

from lim_worker.t11_common import norm_tags, training_device
from rs_backbone_eval.contracts import ManifestSample, MetricRecord, WorkerRequest
from rs_backbone_eval.data.manifest import audit_task_manifest
from rs_backbone_eval.heads import UPerNetHead, count_parameters
from rs_backbone_eval.registry import ProtocolSpec, TaskSpec
from rs_backbone_eval.worker_runtime import RunWriter

VOID_CLASS = 19
FALLBACK_IGNORE = 19
BANDS_RGB = (2, 1, 0)  # B04=red, B03=green, B02=blue
SCALE = 255.0 / 10000.0
IMG_MEAN = (123.675, 116.28, 103.53)
IMG_STD = (58.395, 57.12, 57.375)
YEAR_DATE_ANCHOR = 20190715
EPOCHS = 8
BATCH = 8
STATS_PATCHES = 80
TRAIN_MIN = 1400


class LimT11Failure(Exception):
    """Gates convert to a failed run with a readable reason."""


def _rows(request: WorkerRequest) -> list[ManifestSample]:
    return [
        ManifestSample.model_validate(json.loads(line))
        for line in Path(request.manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _decode_row(
    row: ManifestSample, request: WorkerRequest, ignore: int, variant: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (3,H,W) LIM-RGB stack and the labels for one patch."""
    aux = row.aux_paths or {}
    s2_rel = aux.get("s2_path")
    if not s2_rel:
        raise LimT11Failure(f"sample {row.sample_id}: aux_paths has no s2_path")
    data_root = Path(request.data_root)
    sidecar = (data_root / row.image_paths[0]).with_suffix(".json")
    if not sidecar.is_file():
        raise LimT11Failure(f"sample {row.sample_id}: AEF sidecar {sidecar.name} missing")
    with sidecar.open("r", encoding="utf-8") as fh:
        meta = json.loads(fh.read())
    dates = meta.get("dates_s2") or []
    stack = np.load(data_root / s2_rel, mmap_mode="r")
    if stack.shape[0] != len(dates):
        raise LimT11Failure(
            f"sample {row.sample_id}: S2 stack T={stack.shape[0]} vs sidecar dates {len(dates)}"
        )
    rgb = np.asarray(stack[:, list(BANDS_RGB)], dtype=np.float32) * SCALE
    if variant == "single":
        anchor = int(aux.get("frame_date_anchor") or YEAR_DATE_ANCHOR)
        index = int(np.argmin(np.abs(np.asarray(dates, dtype=np.int64) - anchor)))
        frame = rgb[index]
    elif variant == "median":
        frame = np.median(rgb, axis=0)
    else:
        raise LimT11Failure(f"unknown LIM T11 variant {variant!r}")
    mean = np.asarray(IMG_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMG_STD, dtype=np.float32)[:, None, None]
    features = (frame - mean) / std
    labels = np.load(data_root / row.label_path)[0].astype(np.int64)
    if labels.shape != features.shape[1:]:
        raise LimT11Failure(f"sample {row.sample_id}: label grid {labels.shape} != feature grid")
    labels = np.where(labels == VOID_CLASS, ignore, labels)
    return features, labels


def _tile_balanced_stats(
    rows: list[ManifestSample], request: WorkerRequest, ignore: int, variant: str
) -> tuple[np.ndarray, np.ndarray]:
    """Tile-balanced statistics over the train fold, mirror of the AEF track."""
    split = sorted(
        (row for row in rows if row.split == "train"), key=lambda row: int(row.sample_id)
    )
    by_tile: dict[str, list[ManifestSample]] = {}
    for row in split:
        by_tile.setdefault(row.aux_paths.get("tile", ""), []).append(row)
    chosen: list[ManifestSample] = []
    index = 0
    while len(chosen) < min(STATS_PATCHES, len(split)):
        advanced = False
        for tile in sorted(by_tile):
            if index < len(by_tile[tile]):
                chosen.append(by_tile[tile][index])
                advanced = True
        index += 1
        if not advanced:
            break
    total = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    seen = 0
    for row in chosen[:STATS_PATCHES]:
        features, _ = _decode_row(row, request, ignore, variant)
        total += features.sum(axis=(1, 2))
        squares += (features**2).sum(axis=(1, 2))
        seen += features.shape[1] * features.shape[2]
    mean = total / max(seen, 1)
    std = np.sqrt(np.maximum(squares / max(seen, 1) - mean**2, 0.0)) + 1e-3
    return mean.astype(np.float32), std.astype(np.float32)


def train_t11(
    request: WorkerRequest,
    writer: RunWriter,
    event_log,
    *,
    protocol: ProtocolSpec,
    task_spec: TaskSpec,
    checkpoint: Path,
) -> int:
    from rs_backbone_eval.repro import seed_everything

    seed_everything(request.seed)
    variant = {
        "lim_s2_single": "single",
        "lim_s2_median": "median",
    }[protocol.protocol_id]
    ignore = task_spec.ignore_index if task_spec.ignore_index is not None else FALLBACK_IGNORE

    rows = _rows(request)
    manifest_path = Path(request.manifest_path)
    audit = audit_task_manifest(manifest_path, task_spec)
    if not audit.valid:
        raise LimT11Failure(f"manifest audit failed: {audit.all_errors()[:5]}")
    event_log(
        request.run_id,
        "manifest.ok",
        f"{audit.sample_count} rows, splits {dict(audit.split_counts)}",
    )

    groups = {
        split: sorted(
            (row for row in rows if row.split == split), key=lambda row: int(row.sample_id)
        )
        for split in ("train", "validation", "test")
    }
    if len(groups["train"]) < TRAIN_MIN:
        raise LimT11Failure(
            f"train fold has {len(groups['train'])} rows; the protocol needs >= {TRAIN_MIN}"
        )

    encoder = mit_huge(in_chans=3, drop_path_rate=0.0)
    load_encoder(checkpoint, encoder)
    device, provenance = training_device()
    encoder.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    head = UPerNetHead(list(encoder.embed_dims), num_classes=task_spec.num_outputs)
    head.to(device)
    started = time.time()
    writer.update_training(
        head_id=f"lim_s2_{variant}_upernet_fpn256",
        trainable_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
        total_parameters=count_parameters(head),
    )
    writer.update_input(preprocessing_id=f"lim_s2_{variant}_tilestats")
    mean, std = _tile_balanced_stats(groups["train"], request, ignore, variant)
    tags = dict(provenance)
    tags.update(norm_tags(mean, std))
    tags["test_manifest_samples"] = str(len(groups["test"]))
    writer.update(tags=tags)
    mean_t = torch.from_numpy(mean)[:, None, None]
    std_t = torch.from_numpy(std)[:, None, None]
    optimiser = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)

    total_steps = 0
    for epoch in range(EPOCHS):
        order = torch.randperm(
            len(groups["train"]),
            generator=torch.Generator().manual_seed(1_000 * request.seed + epoch),
        ).tolist()
        head.train()
        total = 0.0
        steps = 0
        for start in range(0, len(group_rows := groups["train"]), BATCH):
            chunk = [group_rows[index] for index in order[start : start + BATCH]]
            pairs = [_decode_row(row, request, ignore, variant) for row in chunk]
            stacked = torch.stack(
                [(torch.from_numpy(values) - mean_t) / std_t for values, _ in pairs]
            ).to(device)
            labels = torch.stack([torch.from_numpy(target) for _, target in pairs]).to(device)
            with torch.no_grad():
                pyramid = encoder(stacked)
            loss = F.cross_entropy(
                head(pyramid, output_size=stacked.shape[-2:]), labels, ignore_index=ignore
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            steps += 1
        total_steps += steps
        event_log(
            request.run_id,
            "train.epoch",
            f"epoch {epoch}: loss {total / max(steps, 1):.4f}",
            progress=(epoch + 1) / EPOCHS,
        )

    for split in ("validation", "test"):
        evaluator = _IoUSink(task_spec.num_outputs, ignore)
        head.eval()
        evaluated = 0
        with torch.no_grad():
            for start in range(0, len(groups[split]), BATCH):
                chunk = groups[split][start : start + BATCH]
                evaluated += len(chunk)
                pairs = [_decode_row(row, request, ignore, variant) for row in chunk]
                stacked = torch.stack(
                    [(torch.from_numpy(values) - mean_t) / std_t for values, _ in pairs]
                ).to(device)
                labels = torch.stack([torch.from_numpy(target) for _, target in pairs]).to(device)
                logits = head(encoder(stacked), output_size=stacked.shape[-2:])
                evaluator.add(labels, logits.argmax(dim=1))
        if split == "test":
            if evaluated != len(groups["test"]):
                raise LimT11Failure(
                    f"test fold evaluated on {evaluated} of {len(groups['test'])} rows; "
                    "an eval cap silently shrank the holdout (review 2026-11-04)"
                )
            writer.update(tags={"test_samples_evaluated": str(evaluated)})
        value = evaluator.value()
        writer.add_metrics(
            [
                MetricRecord(
                    name="mean_iou",
                    direction=task_spec.primary_metric.direction,
                    value=round(value["miou"], 6),
                    split=split,
                    is_primary=split == "test",
                ),
                MetricRecord(
                    name="mean_iou_foreground",
                    direction=task_spec.primary_metric.direction,
                    value=round(value["foreground"], 6),
                    split=split,
                    is_primary=False,
                ),
            ]
        )
    test = next(metric for metric in writer.record.metrics if metric.is_primary)
    foreground = next(
        metric.value
        for metric in writer.record.metrics
        if metric.name == "mean_iou_foreground" and metric.split == "test"
    )
    writer.update_training(
        epochs_completed=EPOCHS,
        optimizer_steps=total_steps,
        wall_time_seconds=round(time.time() - started, 1),
    )
    writer.complete(
        notes=(
            f"LIM MiT-Huge frozen; variant {variant}; head UPerNetHead(fpn 256); "
            "value decoding DN/10000*255; final input z-scored by train-fold tile "
            "statistics (ImageNet affine pass cancels; NORM_DECISION a808302 not "
            "effective for T11, declared exception); "
            f"single = year-mid frame, median = temporal median over all frames; "
            f"host {tags['host']}; labels 2019 pending LPIS-year confirmation"
        )
    )
    event_log(
        request.run_id,
        "train.done",
        f"variant {variant}: test mIoU {test.value:.4f}, foreground {foreground:.4f}",
        progress=1.0,
    )
    return 0


class _IoUSink:
    """Confusion-matrix sink for the LIM T11 variants (aef-tracked mirror)."""

    def __init__(self, num_classes: int, ignore: int) -> None:
        self.num_classes = num_classes
        self.ignore = ignore
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def add(self, targets: torch.Tensor, predicted: torch.Tensor) -> None:
        mask = targets != self.ignore
        flat = (targets[mask] * self.num_classes + predicted[mask]).cpu().numpy()
        self.confusion += np.bincount(flat, minlength=self.num_classes * self.num_classes).reshape(
            self.num_classes, self.num_classes
        )

    def value(self) -> dict[str, float]:
        tp = np.diag(self.confusion)
        fp = self.confusion.sum(axis=0) - tp
        fn = self.confusion.sum(axis=1) - tp
        present = self.confusion.sum(axis=1) > 0
        iou = np.where(present, tp / np.maximum(tp + fp + fn, 1), np.nan)
        foreground = [float(iou[index]) for index in range(1, self.ignore) if present[index]]
        return {
            "miou": float(np.nanmean(iou)),
            "foreground": float(np.mean(foreground)) if foreground else float("nan"),
        }
