"""LIM on PASTIS-HD with a 20-bin LTAE2d temporal head (T11, registry runs).

Declared protocol ``lim_s2_20bin`` (track BB, same family as lim_s2_single /
lim_s2_median but with a learned temporal encoder):

* per patch, 20 frames are picked with a deterministic linspace over the
  sorted date list (no randomness, identical train/val/test selection);
* each frame runs the frozen MiT-Huge encoder (no_grad); the four pyramid
  levels of the T frames are stacked per level and folded into a learned
  yearly embedding by a vendored official LTAE2d
  (``lim_worker.vendor.ltae.ltae.LTAE2d``, utae-paps @ 987874e, MIT) --
  one LTAE2d per level, n_head 16 / d_k 8 / dropout 0.2 / d_model 256,
  positional encoding over day-of-year with period 366;
* the shared ``rs_backbone_eval.heads.UPerNetHead`` (fpn 256) consumes the
  four per-level embeddings, exactly like the single/median variants.

Values are decoded from ``aux_paths.s2_path`` exactly like the other LIM
variants: bands B04/B03/B02, DN/10000*255; the effective input is z-scored by
the train-fold tile-balanced statistics (the ImageNet affine pass cancels
under z-scoring -- NORM_DECISION a808302 is NOT effective for T11, declared
exception, review 2026-11-04; see ``t11_common.norm_tags``). Budget mirrors
the AEF track: 8 epochs on the full Fold 1-3, AdamW lr 1e-3, effective batch
8 (batch 8 fits 40 GB).

The formal three seeds run in the Colab A100 40GB lane (2026-11-04
leadership decision): the whole cell stays on one GPU class and the host is
recorded from ``torch.cuda.get_device_name`` at runtime (review 2026-11-04).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from lim_worker.mit import load_encoder, mit_huge
else:
    from .mit import load_encoder, mit_huge

import lim_worker.t11_common as t11_common
from lim_worker.t11_common import norm_tags, training_device
from lim_worker.vendor.ltae.ltae import LTAE2d
from rs_backbone_eval.contracts import ManifestSample, MetricRecord, WorkerRequest
from rs_backbone_eval.data.manifest import audit_task_manifest
from rs_backbone_eval.heads import UPerNetHead, count_parameters
from rs_backbone_eval.registry import ProtocolSpec, TaskSpec
from rs_backbone_eval.worker_runtime import RunWriter, emit_event

VOID_CLASS = 19
FALLBACK_IGNORE = 19
BANDS_RGB = (2, 1, 0)  # B04=red, B03=green, B02=blue
SCALE = 255.0 / 10000.0
IMG_MEAN = (123.675, 116.28, 103.53)
IMG_STD = (58.395, 57.12, 57.375)
EPOCHS = 8
BATCH = 8
BINS = 20
N_HEAD = 16
D_K = 8
DROPOUT = 0.2
D_MODEL = 256
T_PE = 366
STATS_PATCHES = 24
TRAIN_MIN = 1400


class LimT11LtaeFailure(Exception):
    """Gates convert to a failed run with a readable reason."""


class LIMT11LTAE(nn.Module):
    """MiT-Huge frozen per-frame encoder + per-level LTAE2d + shared head."""

    def __init__(self, in_channels, num_classes: int, fpn_dim: int = 256):
        super().__init__()
        self.levels = nn.ModuleList(
            [
                LTAE2d(
                    in_channels=channels,
                    n_head=N_HEAD,
                    d_k=D_K,
                    mlp=[D_MODEL, D_MODEL],
                    dropout=DROPOUT,
                    d_model=D_MODEL,
                    T=T_PE,
                )
                for channels in in_channels
            ]
        )
        self.head = UPerNetHead(
            [D_MODEL] * len(in_channels), num_classes=num_classes, fpn_dim=fpn_dim
        )

    def forward(
        self, pyramid_frames: list[torch.Tensor], doy: torch.Tensor, output_size: tuple[int, int]
    ) -> torch.Tensor:
        embeddings = [
            module(level, doy) for module, level in zip(self.levels, pyramid_frames, strict=True)
        ]
        return self.head(embeddings, output_size=output_size)


def _rows(request: WorkerRequest) -> list[ManifestSample]:
    return [
        ManifestSample.model_validate(json.loads(line))
        for line in Path(request.manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _bin_indices(frame_count: int) -> list[int]:
    """Deterministic linspace bin selection over the sorted date list."""
    if frame_count <= BINS:
        return list(range(frame_count))
    picked = {round(index * (frame_count - 1) / (BINS - 1)) for index in range(BINS)}
    return sorted(picked)


def _doy(date_int: int) -> int:
    text = str(date_int)
    return date(int(text[:4]), int(text[4:6]), int(text[6:8])).timetuple().tm_yday


def _decode_stack(
    row: ManifestSample, data_root: Path, ignore: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the (bins,3,H,W) z-scored bin stack, the labels and the doys."""
    aux = row.aux_paths or {}
    s2_rel = aux.get("s2_path")
    if not s2_rel:
        raise LimT11LtaeFailure(f"sample {row.sample_id}: aux_paths has no s2_path")
    sidecar = (data_root / row.image_paths[0]).with_suffix(".json")
    if not sidecar.is_file():
        raise LimT11LtaeFailure(f"sample {row.sample_id}: sidecar {sidecar.name} missing")
    with sidecar.open("r", encoding="utf-8") as handle:
        meta = json.loads(handle.read())
    dates = meta.get("dates_s2") or []
    stack = np.load(data_root / s2_rel, mmap_mode="r")
    if stack.shape[0] != len(dates):
        raise LimT11LtaeFailure(
            f"sample {row.sample_id}: S2 stack T={stack.shape[0]} vs sidecar dates {len(dates)}"
        )
    indices = _bin_indices(stack.shape[0])
    rgb = np.asarray(stack[indices][:, list(BANDS_RGB)], dtype=np.float32) * SCALE
    mean = np.asarray(IMG_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMG_STD, dtype=np.float32)[:, None, None]
    features = (rgb - mean) / std
    labels = np.load(data_root / row.label_path)[0].astype(np.int64)
    if labels.shape != features.shape[-2:]:
        raise LimT11LtaeFailure(
            f"sample {row.sample_id}: label grid {labels.shape} != feature grid"
        )
    labels = np.where(labels == VOID_CLASS, ignore, labels)
    doys = np.asarray([_doy(int(dates[index])) for index in indices], dtype=np.int64)
    return features, labels, doys


def _tile_balanced_stats(
    rows: list[ManifestSample], data_root: Path, ignore: int
) -> tuple[np.ndarray, np.ndarray]:
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
        features, _, _ = _decode_stack(row, data_root, ignore)
        total += features.sum(axis=(0, 2, 3))
        squares += (features**2).sum(axis=(0, 2, 3))
        seen += features.shape[0] * features.shape[2] * features.shape[3]
    mean = total / max(seen, 1)
    std = np.sqrt(np.maximum(squares / max(seen, 1) - mean**2, 0.0)) + 1e-3
    return mean.astype(np.float32), std.astype(np.float32)


def train_t11_ltae(
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
    ignore = task_spec.ignore_index if task_spec.ignore_index is not None else FALLBACK_IGNORE

    rows = _rows(request)
    manifest_path = Path(request.manifest_path)
    audit = audit_task_manifest(manifest_path, task_spec)
    if not audit.valid:
        raise LimT11LtaeFailure(f"manifest audit failed: {audit.all_errors()[:5]}")
    emit_event(
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
        raise LimT11LtaeFailure(
            f"train fold has {len(groups['train'])} rows; the protocol needs >= {TRAIN_MIN}"
        )

    data_root = Path(request.data_root)
    encoder = mit_huge(in_chans=3, drop_path_rate=0.0)
    load_encoder(checkpoint, encoder)
    device, provenance = training_device()
    encoder.to(device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    channels = list(encoder.embed_dims)
    model = LIMT11LTAE(channels, task_spec.num_outputs)
    model.to(device)
    started = time.time()
    writer.update_training(
        head_id="lim_s2_20bin_ltae2d_upernet_fpn256",
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="lim_s2_20bin_tilestats")
    mean, std = _tile_balanced_stats(groups["train"], data_root, ignore)
    tags = dict(provenance)
    tags.update(norm_tags(mean, std))
    tags["test_manifest_samples"] = str(len(groups["test"]))
    tags["test_expected_samples"] = str(t11_common.EXPECTED_TEST_SAMPLES)
    writer.update(tags=tags)
    mean_t = torch.from_numpy(mean)[:, None, None]
    std_t = torch.from_numpy(std)[:, None, None]
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Diagnostic knobs (never on the formal path): the Colab/A100 formal cell
    # keeps the declared 8-epoch / batch-8 budget, while a quick one-seed
    # sanity run may cap epochs and use a smaller batch to share a busy card.
    # Both overrides are recorded in the tags and the notes carry
    # REDUCED-BUDGET so is_reduced_run() can never mistake this for formal.
    epochs = int(os.environ.get("LIM_T11_EPOCHS", EPOCHS))
    batch_size = int(os.environ.get("LIM_T11_BATCH", BATCH))
    if epochs != EPOCHS or batch_size != BATCH:
        writer.update(tags={"epochs_override": str(epochs), "batch_override": str(batch_size)})

    total_steps = 0
    for epoch in range(epochs):
        order = torch.randperm(
            len(groups["train"]),
            generator=torch.Generator().manual_seed(1_000 * request.seed + epoch),
        ).tolist()
        model.train()
        total = 0.0
        steps = 0
        optimiser.zero_grad()
        for start in range(0, len(groups["train"]), batch_size):
            chunk = [groups["train"][index] for index in order[start : start + batch_size]]
            decoded = [_decode_stack(row, data_root, ignore) for row in chunk]
            stacked = torch.stack(
                [(torch.from_numpy(values) - mean_t) / std_t for values, _, _ in decoded]
            ).to(device)
            labels = torch.stack([torch.from_numpy(target) for _, target, _ in decoded]).to(device)
            doy = torch.stack([torch.from_numpy(positions) for _, _, positions in decoded]).to(
                device
            )
            with torch.no_grad():
                per_frame = encoder(stacked.flatten(0, 1))
                pyramid = [
                    level.view(stacked.shape[0], stacked.shape[1], *level.shape[1:])
                    for level in per_frame
                ]
            loss = F.cross_entropy(
                model(pyramid, doy, output_size=stacked.shape[-2:]), labels, ignore_index=ignore
            )
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            total += float(loss.detach())
            steps += 1
        total_steps += steps
        emit_event(
            request.run_id,
            "train.epoch",
            f"epoch {epoch}: loss {total / max(steps, 1):.4f}",
            progress=(epoch + 1) / EPOCHS,
        )

    for split in ("validation", "test"):
        confusion = np.zeros((task_spec.num_outputs, task_spec.num_outputs), dtype=np.int64)
        model.eval()
        evaluated = 0
        with torch.no_grad():
            for start in range(0, len(groups[split]), batch_size):
                chunk = groups[split][start : start + batch_size]
                evaluated += len(chunk)
                decoded = [_decode_stack(row, data_root, ignore) for row in chunk]
                stacked = torch.stack(
                    [(torch.from_numpy(values) - mean_t) / std_t for values, _, _ in decoded]
                ).to(device)
                labels = torch.stack([torch.from_numpy(target) for _, target, _ in decoded]).to(
                    device
                )
                doy = torch.stack([torch.from_numpy(positions) for _, _, positions in decoded]).to(
                    device
                )
                per_frame = encoder(stacked.flatten(0, 1))
                pyramid = [
                    level.view(stacked.shape[0], stacked.shape[1], *level.shape[1:])
                    for level in per_frame
                ]
                predicted = model(pyramid, doy, output_size=stacked.shape[-2:]).argmax(dim=1)
                mask = labels != ignore
                flat = (labels[mask] * task_spec.num_outputs + predicted[mask]).cpu().numpy()
                confusion += np.bincount(flat, minlength=task_spec.num_outputs**2).reshape(
                    task_spec.num_outputs, task_spec.num_outputs
                )
        if split == "test":
            expected = t11_common.EXPECTED_TEST_SAMPLES
            if evaluated != expected:
                raise LimT11LtaeFailure(
                    f"test fold evaluated on {evaluated} of the official {expected} "
                    "rows; a shrunken manifest/cap must not self-attest "
                    "(review 2026-11-04 round 2, #2)"
                )
            writer.update(tags={"test_samples_evaluated": str(evaluated)})
        tp = np.diag(confusion)
        fp = confusion.sum(axis=0) - tp
        fn = confusion.sum(axis=1) - tp
        present = confusion.sum(axis=1) > 0
        iou = np.where(present, tp / np.maximum(tp + fp + fn, 1), np.nan)
        foreground = [float(iou[index]) for index in range(1, ignore) if present[index]]
        writer.add_metrics(
            [
                MetricRecord(
                    name="mean_iou",
                    direction=task_spec.primary_metric.direction,
                    value=round(float(np.nanmean(iou)), 6),
                    split=split,
                    is_primary=split == "test",
                ),
                MetricRecord(
                    name="mean_iou_foreground",
                    direction=task_spec.primary_metric.direction,
                    value=round(float(np.mean(foreground)), 6) if foreground else float("nan"),
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
        epochs_completed=epochs,
        optimizer_steps=total_steps,
        wall_time_seconds=round(time.time() - started, 1),
    )
    writer.complete(
        notes=(
            ("REDUCED-BUDGET diagnostic " if epochs != EPOCHS else "")
            + "LIM MiT-Huge frozen per-frame; 20-bin deterministic linspace binning "
            "(bin indices identical train/val/test); vendored LTAE2d "
            f"per level (n_head {N_HEAD}, d_k {D_K}, dropout {DROPOUT}, "
            f"d_model {D_MODEL}, T {T_PE} day-of-year, vendored utae-paps@987874e); "
            "head UPerNetHead(fpn 256); value decoding DN/10000*255; final input "
            "z-scored by train-fold tile statistics (ImageNet affine "
            "pass cancels; NORM_DECISION a808302 not effective for T11, declared "
            f"exception); formal host {tags['host']} (whole cell)"
        )
    )
    emit_event(
        request.run_id,
        "train.done",
        f"lim_s2_20bin: test mIoU {test.value:.4f}, foreground {foreground:.4f}",
        progress=1.0,
    )
    return 0
