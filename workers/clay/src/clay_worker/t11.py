"""Clay v1.5 on PASTIS-HD (T11, single variant).

Clay's ``sentinel-2-l2a`` entry takes exactly the ten PASTIS-HD bands in the
documented order (blue..swir22 == B2..B12), raw reflectance with the sensor
mean/std from Clay's own metadata, plus gsd=10 m and the band wavelengths.
Segmentation reuses the T9 recipe: ``SegmentEncoder`` emits one token grid at
stride 8 and the shared ``UPerNetHead`` receives four levels produced by
parameter-free average pooling (recorded in the tags).

Only the single variant is registered (operator decision 2026-11-04: the
per-pixel temporal median composes a pseudo-scene backbones never saw; new
backbones are single-only). Row inputs come from ``aux_paths.s2_path`` exactly
like the LIM/AnySat adapters. Budget mirrors the T11 AEF probe: 8 epochs,
AdamW lr 1e-3, effective batch 8 (2 x 4 accumulation), frozen backbone,
coverage hard-anchored on 496/482.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rs_backbone_eval.contracts import ManifestSample, MetricRecord, WorkerRequest
from rs_backbone_eval.data.manifest import audit_task_manifest
from rs_backbone_eval.heads import UPerNetHead, count_parameters
from rs_backbone_eval.matrix import OFFICIAL_TEST_SAMPLES, OFFICIAL_VALIDATION_SAMPLES

from .clay_probe import add_upstream_to_path, load_sensor_metadata, prepare_encoder
from .t1 import CLAY_LARGE

VOID_CLASS = 19
FALLBACK_IGNORE = 19
YEAR_DATE_ANCHOR = 20190715
EPOCHS = 8
BATCH = 2
ACCUM = 4
STATS_PATCHES = 24
TRAIN_MIN = 1400
SENSOR = "sentinel-2-l2a"
GSD_M = 10.0
BAND_ORDER = [
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "swir16",
    "swir22",
]


class ClayT11Failure(Exception):
    """Gates convert to a failed run with a readable reason."""


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def training_device() -> tuple[str, dict[str, str]]:
    required = os.environ.get("RS_EVAL_REQUIRE_DEVICE", "")
    if required == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RS_EVAL_REQUIRE_DEVICE=cuda but torch.cuda.is_available() is False")
    if torch.cuda.is_available():
        tags = {"device": "cuda", "host": torch.cuda.get_device_name(0).lower().replace(" ", "-")}
    else:
        tags = {"device": "cpu", "host": "cpu"}
    tags["torch"] = torch.__version__
    return tags["device"], tags


def _rows(request: WorkerRequest) -> list[ManifestSample]:
    return [
        ManifestSample.model_validate(json.loads(line))
        for line in Path(request.manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _doy(date_int: int) -> int:
    text = str(date_int)
    return date(int(text[:4]), int(text[4:6]), int(text[6:8])).timetuple().tm_yday


def _decode_single(row: ManifestSample, data_root: Path, ignore: int):
    """Year-mid frame (raw DN) + labels + day-of-year."""
    aux = row.aux_paths or {}
    s2_rel = aux.get("s2_path")
    if not s2_rel:
        raise ClayT11Failure(f"sample {row.sample_id}: aux_paths has no s2_path")
    sidecar = (data_root / row.image_paths[0]).with_suffix(".json")
    if not sidecar.is_file():
        raise ClayT11Failure(f"sample {row.sample_id}: sidecar {sidecar.name} missing")
    with sidecar.open("r", encoding="utf-8") as handle:
        dates = json.loads(handle.read()).get("dates_s2") or []
    stack = np.load(data_root / s2_rel, mmap_mode="r")
    if stack.shape[0] != len(dates):
        raise ClayT11Failure(
            f"sample {row.sample_id}: S2 stack T={stack.shape[0]} vs sidecar {len(dates)}"
        )
    anchor = int(aux.get("frame_date_anchor") or YEAR_DATE_ANCHOR)
    index = int(np.argmin(np.abs(np.asarray(dates, dtype=np.int64) - anchor)))
    pixels = np.asarray(stack[index], dtype=np.float32)  # raw DN (10,H,W)
    labels = np.load(data_root / row.label_path)[0].astype(np.int64)
    if labels.shape != pixels.shape[-2:]:
        raise ClayT11Failure(f"sample {row.sample_id}: label grid {labels.shape} mismatch")
    return pixels, np.where(labels == VOID_CLASS, ignore, labels), _doy(int(dates[index]))


def _waves(metadata: dict) -> torch.Tensor:
    wavelengths = metadata["bands"]["wavelength"]
    return torch.tensor([wavelengths[band] for band in BAND_ORDER], dtype=torch.float32)


def _mean_std(metadata: dict) -> tuple[torch.Tensor, torch.Tensor]:
    means = metadata["bands"]["mean"]
    stds = metadata["bands"]["std"]
    return (
        torch.tensor([means[band] for band in BAND_ORDER], dtype=torch.float32).view(10, 1, 1),
        torch.tensor([stds[band] for band in BAND_ORDER], dtype=torch.float32).view(10, 1, 1),
    )


class ClaySingleSegmentation(nn.Module):
    """Clay SegmentEncoder (frozen) + shared UPerNet head at stride 8 -> 4 levels."""

    def __init__(self, encoder, num_classes: int, waves: torch.Tensor, fpn_dim: int = 256):
        super().__init__()
        self.encoder = encoder
        self.head = UPerNetHead([CLAY_LARGE["dim"]] * 4, num_classes=num_classes, fpn_dim=fpn_dim)
        self.register_buffer("waves", waves, persistent=False)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = pixels.shape
        datacube = {
            "pixels": pixels,
            "time": torch.zeros(batch, 4, device=pixels.device),
            "latlon": torch.zeros(batch, 4, device=pixels.device),
            "gsd": torch.tensor(GSD_M, device=pixels.device),
            "waves": self.waves,
            "platform": SENSOR,
        }
        with torch.no_grad():
            tokens = self.encoder(datacube)
        patch = CLAY_LARGE["patch_size"]
        features = tokens.transpose(1, 2).reshape(
            batch, CLAY_LARGE["dim"], height // patch, width // patch
        )
        levels = [features]
        for _ in range(3):
            levels.append(F.avg_pool2d(levels[-1], 2))
        return self.head(levels, output_size=(height, width))


def _build_encoder(checkpoint: Path):
    from claymodel.finetune.segment.factory import SegmentEncoder  # upstream checkout

    return SegmentEncoder(**CLAY_LARGE, ckpt_path=str(checkpoint))


def train_t11_single(
    request: WorkerRequest,
    writer,
    emit,
    *,
    protocol,
    task_spec,
    checkpoint: Path,
) -> int:
    from rs_backbone_eval.repro import seed_everything

    seed_everything(request.seed)
    if protocol.protocol_id != "clay_s2_single":
        raise ClayT11Failure(f"clay T11 runs clay_s2_single only, got {protocol.protocol_id}")
    ignore = task_spec.ignore_index if task_spec.ignore_index is not None else FALLBACK_IGNORE
    data_root = Path(request.data_root)

    rows = _rows(request)
    audit = audit_task_manifest(Path(request.manifest_path), task_spec)
    if not audit.valid:
        raise ClayT11Failure(f"manifest audit failed: {audit.all_errors()[:5]}")
    emit(
        request.run_id,
        "manifest.ok",
        f"{audit.sample_count} rows, splits {dict(audit.split_counts)}",
    )

    groups = {
        split: sorted((r for r in rows if r.split == split), key=lambda r: int(r.sample_id))
        for split in ("train", "validation", "test")
    }
    if len(groups["train"]) < TRAIN_MIN:
        raise ClayT11Failure(
            f"train fold has {len(groups['train'])} rows; the protocol needs >= {TRAIN_MIN}"
        )

    source_root = add_upstream_to_path()
    metadata = load_sensor_metadata(source_root, SENSOR)
    device, provenance = training_device()
    encoder = _build_encoder(checkpoint)
    prepare_encoder(encoder, freeze_backbone=True)
    encoder.to(device)
    model = ClaySingleSegmentation(encoder, task_spec.num_outputs, _waves(metadata)).to(device)
    mean, std = _mean_std(metadata)
    mean, std = mean.to(device), std.to(device)

    batch = _int_env("CLAY_T11_BATCH", BATCH)
    accum = _int_env("CLAY_T11_ACCUM", ACCUM)
    epochs = _int_env("CLAY_T11_EPOCHS", EPOCHS)

    started = time.time()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writer.update_training(
        head_id="clay_s2_single_upernet_fpn256",
        trainable_parameters=trainable,
        total_parameters=count_parameters(model),
    )
    writer.update_input(preprocessing_id="clay_s2_single_claystats")
    tags = dict(provenance)
    tags.update(
        {
            "norm_source": "clay_sensor_metadata",
            "true_input": (
                "aux s2_path stack single year-mid frame (10,128,128) raw DN, "
                "Clay sentinel-2-l2a mean/std, gsd 10 m, 10 band wavelengths"
            ),
            "clay_levels": "stride8 token grid avg-pooled to 4 levels (parameter-free)",
            "test_manifest_samples": str(len(groups["test"])),
            "test_expected_samples": str(OFFICIAL_TEST_SAMPLES["T11"]),
            "validation_manifest_samples": str(len(groups["validation"])),
            "validation_expected_samples": str(OFFICIAL_VALIDATION_SAMPLES["T11"]),
        }
    )
    if epochs != EPOCHS or batch != BATCH or accum != ACCUM:
        tags["epochs_override"] = str(epochs)
        tags["batch_override"] = f"{batch}x{accum}"
    writer.update(tags=tags)

    optimiser = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4
    )
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
        for start in range(0, len(groups["train"]), batch):
            chunk = [groups["train"][index] for index in order[start : start + batch]]
            decoded = [_decode_single(row, data_root, ignore) for row in chunk]
            pixels = torch.stack(
                [(torch.from_numpy(pix) - mean) / std for pix, _, _ in decoded]
            ).to(device)
            labels = torch.stack([torch.from_numpy(t) for _, t, _ in decoded]).to(device)
            loss = F.cross_entropy(model(pixels), labels, ignore_index=ignore) / accum
            loss.backward()
            if (steps + 1) % accum == 0:
                optimiser.step()
                optimiser.zero_grad()
            total += float(loss.detach()) * accum
            steps += 1
        if steps % accum:
            optimiser.step()
            optimiser.zero_grad()
        total_steps += steps
        emit(
            request.run_id,
            "train.epoch",
            f"epoch {epoch}: loss {total / max(steps, 1):.4f}",
            progress=(epoch + 1) / epochs,
        )

    for split in ("validation", "test"):
        confusion = np.zeros((task_spec.num_outputs, task_spec.num_outputs), dtype=np.int64)
        model.eval()
        evaluated = 0
        with torch.no_grad():
            for start in range(0, len(groups[split]), batch):
                chunk = groups[split][start : start + batch]
                evaluated += len(chunk)
                decoded = [_decode_single(row, data_root, ignore) for row in chunk]
                pixels = torch.stack(
                    [(torch.from_numpy(pix) - mean) / std for pix, _, _ in decoded]
                ).to(device)
                labels = torch.stack([torch.from_numpy(t) for _, t, _ in decoded]).to(device)
                predicted = model(pixels).argmax(dim=1)
                mask = labels != ignore
                flat = (labels[mask] * task_spec.num_outputs + predicted[mask]).cpu().numpy()
                confusion += np.bincount(flat, minlength=task_spec.num_outputs**2).reshape(
                    task_spec.num_outputs, task_spec.num_outputs
                )
        expected = (
            OFFICIAL_TEST_SAMPLES["T11"] if split == "test" else OFFICIAL_VALIDATION_SAMPLES["T11"]
        )
        if evaluated != expected:
            raise ClayT11Failure(
                f"{split} fold evaluated on {evaluated} of the official {expected} rows"
            )
        writer.update(tags={f"{split}_samples_evaluated": str(evaluated)})
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
    writer.update_training(
        epochs_completed=epochs,
        optimizer_steps=total_steps,
        wall_time_seconds=round(time.time() - started, 1),
    )
    writer.complete(
        notes=(
            ("REDUCED-BUDGET diagnostic " if epochs != EPOCHS or batch != BATCH else "")
            + "Clay v1.5 SegmentEncoder frozen; single year-mid frame; "
            "sentinel-2-l2a metadata (mean/std, gsd 10 m, 10 wavelengths); "
            "UPerNet(fpn 256) over avg-pooled stride-8 token levels; "
            f"host {tags['host']}; labels 2019 pending LPIS-year confirmation"
        )
    )
    emit(request.run_id, "train.done", f"clay single: test mIoU {test.value:.4f}", progress=1.0)
    return 0
