"""AnySat on PASTIS-HD (T11): frozen S2 entrance + shared spatial head.

AnySat's native Sentinel-2 projector *is* a learned temporal attention over
per-pixel time series (LTAE2d-style, T=367 positional encoding) and it expects
exactly the ten PASTIS-HD bands in this order, so no spectral adapter is
needed: the manifest's ``aux_paths.s2_path`` stack is fed straight in.

Three protocols mirror the LIM variants on the same rows, so the BB track can
compare backbones on identical inputs:

* ``anysat_s2_single`` - the frame closest to the 2019-07-15 year-mid anchor;
* ``anysat_s2_20bin``  - 20 deterministic linspace bins plus days-since-
  2018-09-01 positional inputs (AnySat aggregates them with its own temporal
  attention; the median variant was withdrawn, so only these two are declared).

Values: DN/10000 (reflectance), per-band mean/std from the train fold
(tile-balanced sample), frozen backbone, ``UPerNetHead`` (fpn 256, the same
decoder LIM and Clay use; the dense grid is pooled to stride 8 as a
single-scale pyramid, as Clay's T9 path does). Budget still
mirrors the T11 AEF probe: 8 epochs, AdamW lr 1e-3, effective batch 8
(micro batch x gradient accumulation env-overridable), per-seed shuffles,
coverage hard-anchored on the official 496 test / 482 validation rows.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from rs_backbone_eval.contracts import ManifestSample, MetricRecord, WorkerRequest
from rs_backbone_eval.data.manifest import audit_task_manifest
from rs_backbone_eval.heads import UPerNetHead, count_parameters
from rs_backbone_eval.matrix import OFFICIAL_TEST_SAMPLES, OFFICIAL_VALIDATION_SAMPLES
from rs_backbone_eval.registry import ProtocolSpec, TaskSpec
from rs_backbone_eval.timeutil import PASTIS_REFERENCE_DATE, days_since_reference

from .t2 import build_backbone

VOID_CLASS = 19
FALLBACK_IGNORE = 19
SCALE = 1.0 / 10000.0  # PASTIS DN int16 -> reflectance
YEAR_DATE_ANCHOR = 20190715
BINS = 20
EPOCHS = 8
BATCH = 2
ACCUM = 4
PATCH_M = 80  # AnySat token spacing: 80 m -> 8 px -> 16x16 tokens on 128 px
STATS_PATCHES = 24
TRAIN_MIN = 1400


class AnySatT11Failure(Exception):
    """Gates convert to a failed run with a readable reason."""


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def training_device() -> tuple[str, dict[str, str]]:
    """cuda-if-available; RS_EVAL_REQUIRE_DEVICE=cuda turns silent CPU into an error."""
    required = os.environ.get("RS_EVAL_REQUIRE_DEVICE", "")
    if required == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("RS_EVAL_REQUIRE_DEVICE=cuda but torch.cuda.is_available() is False")
    if torch.cuda.is_available():
        tags = {"device": "cuda", "host": torch.cuda.get_device_name(0).lower().replace(" ", "-")}
    else:
        tags = {"device": "cpu", "host": "cpu"}
    tags["torch"] = torch.__version__
    return tags["device"], tags


def _dense_grid(dense: torch.Tensor) -> torch.Tensor:
    """(B,H,W,C) full-resolution dense tokens -> (B,C,H/8,W/8) stride-8 grid.

    AnySat decodes to input resolution while Clay decodes to stride 8; pooling
    the dense grid into one coarse pyramid level feeds the same
    ``UPerNetHead`` in both workers, so the BB comparison varies only the
    backbone (review 2026-11-04).
    """
    features = dense.permute(0, 3, 1, 2).contiguous()
    size = (max(1, features.shape[-2] // 8), max(1, features.shape[-1] // 8))
    return F.adaptive_avg_pool2d(features, size)


def _rows(request: WorkerRequest) -> list[ManifestSample]:
    return [
        ManifestSample.model_validate(json.loads(line))
        for line in Path(request.manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _bin_indices(frame_count: int) -> list[int]:
    if frame_count <= BINS:
        return list(range(frame_count))
    return sorted({round(index * (frame_count - 1) / (BINS - 1)) for index in range(BINS)})


def _date_position(date_int: int) -> int:
    """Days since 2018-09-01: AnySat's PASTIS loader (``prepare_dates``) convention.

    Day-of-year repeats across the 13-month series (2018-09-24 / 2019-09-24)
    and jumps at New Year, so LTAE's sinusoidal encoding aliased frames a year
    apart (review 2026-11-04).
    """
    return days_since_reference(int(date_int))


def _decode(row: ManifestSample, data_root: Path, ignore: int, variant: str):
    """Return (frames,3-agnostic stack) -> (n,10,H,W), labels, date positions."""
    aux = row.aux_paths or {}
    s2_rel = aux.get("s2_path")
    if not s2_rel:
        raise AnySatT11Failure(f"sample {row.sample_id}: aux_paths has no s2_path")
    sidecar = (data_root / row.image_paths[0]).with_suffix(".json")
    if not sidecar.is_file():
        raise AnySatT11Failure(f"sample {row.sample_id}: sidecar {sidecar.name} missing")
    with sidecar.open("r", encoding="utf-8") as handle:
        meta = json.loads(handle.read())
    dates = meta.get("dates_s2") or []
    stack = np.load(data_root / s2_rel, mmap_mode="r")
    if stack.shape[0] != len(dates):
        raise AnySatT11Failure(
            f"sample {row.sample_id}: S2 stack T={stack.shape[0]} vs sidecar {len(dates)}"
        )
    values = np.asarray(stack, dtype=np.float32) * SCALE
    if variant == "single":
        anchor = int(aux.get("frame_date_anchor") or YEAR_DATE_ANCHOR)
        index = int(np.argmin(np.abs(np.asarray(dates, dtype=np.int64) - anchor)))
        frames = values[index : index + 1]
        positions = [_date_position(int(dates[index]))]
    elif variant == "20bin":
        indices = _bin_indices(values.shape[0])
        frames = values[indices]
        positions = [_date_position(int(dates[index])) for index in indices]
    else:
        raise AnySatT11Failure(f"unknown AnySat T11 variant {variant!r}")
    labels = np.load(data_root / row.label_path)[0].astype(np.int64)
    if labels.shape != frames.shape[-2:]:
        raise AnySatT11Failure(f"sample {row.sample_id}: label grid {labels.shape} mismatch")
    labels = np.where(labels == VOID_CLASS, ignore, labels)
    return frames, labels, positions


def _tile_sample(rows: list[ManifestSample], limit: int) -> list[ManifestSample]:
    split = sorted((row for row in rows if row.split == "train"), key=lambda r: int(r.sample_id))
    by_tile: dict[str, list[ManifestSample]] = {}
    for row in split:
        by_tile.setdefault(row.aux_paths.get("tile", ""), []).append(row)
    chosen: list[ManifestSample] = []
    index = 0
    while len(chosen) < min(limit, len(split)):
        advanced = False
        for tile in sorted(by_tile):
            if index < len(by_tile[tile]):
                chosen.append(by_tile[tile][index])
                advanced = True
        index += 1
        if not advanced:
            break
    return chosen[:limit]


NORM_CACHE_VERSION = 1


def _normalization(
    rows, data_root: Path, cache_dir: Path, ignore: int, variant: str, manifest_digest: str
):
    """Per-band train-fold mean/std over the variant frames (cached).

    The cache key covers everything that changes the statistics: the variant,
    the manifest digest (which rows are sampled), the tile-balanced patch count
    and the decoder version. The previous name-only key silently reused stats
    after a manifest or recipe change, across seeds and runs (review 2026-11-04).
    ``POSITIONS`` is part of the key because the decode position convention
    changed from day-of-year to days-since-2018-09-01.
    """
    key = f"{variant}-{manifest_digest[:16]}-p{STATS_PATCHES}-v{NORM_CACHE_VERSION}"
    cache = cache_dir / f"anysat_t11_norm_{key}.json"
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return np.asarray(payload["mean"], np.float32), np.asarray(payload["std"], np.float32)
    total = np.zeros(10, dtype=np.float64)
    squares = np.zeros(10, dtype=np.float64)
    seen = 0
    for row in _tile_sample(rows, STATS_PATCHES):
        frames, _, _ = _decode(row, data_root, ignore, variant)
        total += frames.sum(axis=(0, 2, 3))
        squares += (frames**2).sum(axis=(0, 2, 3))
        seen += frames.shape[0] * frames.shape[2] * frames.shape[3]
    mean = total / max(seen, 1)
    std = np.sqrt(np.maximum(squares / max(seen, 1) - mean**2, 0.0)) + 1e-6
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "mean": list(map(float, mean)),
                "std": list(map(float, std)),
                "patches": STATS_PATCHES,
                "variant": variant,
                "manifest_sha256": manifest_digest,
                "cache_version": NORM_CACHE_VERSION,
                "positions": f"days since {PASTIS_REFERENCE_DATE}",
            }
        ),
        encoding="utf-8",
    )
    os.replace(temporary, cache)
    return mean.astype(np.float32), std.astype(np.float32)


def train_t11(
    request: WorkerRequest,
    writer,
    emit,
    *,
    protocol: ProtocolSpec,
    task_spec: TaskSpec,
    checkpoint: Path,
) -> int:
    from rs_backbone_eval.repro import seed_everything

    seed_everything(request.seed)
    variant = {
        "anysat_s2_single": "single",
        "anysat_s2_20bin": "20bin",
    }[protocol.protocol_id]
    ignore = task_spec.ignore_index if task_spec.ignore_index is not None else FALLBACK_IGNORE
    data_root = Path(request.data_root)
    cache_dir = Path(request.cache_root)

    rows = _rows(request)
    audit = audit_task_manifest(Path(request.manifest_path), task_spec)
    if not audit.valid:
        raise AnySatT11Failure(f"manifest audit failed: {audit.all_errors()[:5]}")
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
        raise AnySatT11Failure(
            f"train fold has {len(groups['train'])} rows; the protocol needs >= {TRAIN_MIN}"
        )

    device, provenance = training_device()
    backbone = build_backbone(checkpoint)
    backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    head = UPerNetHead([2 * 768], task_spec.num_outputs, fpn_dim=256)
    head.to(device)

    from rs_backbone_eval.data.manifest import manifest_sha256

    manifest_digest = manifest_sha256(Path(request.manifest_path))
    mean, std = _normalization(
        groups["train"], data_root, cache_dir, ignore, variant, manifest_digest
    )
    mean_t = torch.from_numpy(mean)[:, None, None].to(device)
    std_t = torch.from_numpy(std)[:, None, None].to(device)

    batch = _int_env("ANYSAT_T11_BATCH", BATCH)
    accum = _int_env("ANYSAT_T11_ACCUM", ACCUM)
    epochs = _int_env("ANYSAT_T11_EPOCHS", EPOCHS)
    patch_m = _int_env("ANYSAT_T11_PATCH_M", PATCH_M)

    started = time.time()
    writer.update_training(
        head_id=f"anysat_s2_{variant}_upernet_fpn256",
        trainable_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
        total_parameters=count_parameters(head),
    )
    writer.update_input(preprocessing_id=f"anysat_s2_{variant}_tilestats")
    tags = dict(provenance)
    tags.update(
        {
            "norm_source": "train_fold_tile_stats",
            "norm_mean": ",".join(f"{v:.4f}" for v in mean),
            "norm_std": ",".join(f"{v:.4f}" for v in std),
            "true_input": (
                "aux s2_path stack (T,10,128,128) int16 -> reflectance DN/10000, "
                "PASTIS band order == AnySat S2 entrance, z-scored per band"
            ),
            "anysat_patch_m": str(patch_m),
            "temporal_positions": f"days since {PASTIS_REFERENCE_DATE}",
            "head": "UPerNetHead(fpn 256), stride-8 token grid",
            "anysat_batch": f"{batch}x{accum}",
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

    optimiser = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    total_steps = 0
    for epoch in range(epochs):
        order = torch.randperm(
            len(groups["train"]),
            generator=torch.Generator().manual_seed(1_000 * request.seed + epoch),
        ).tolist()
        head.train()
        total = 0.0
        steps = 0
        optimiser.zero_grad()
        for start in range(0, len(groups["train"]), batch):
            chunk = [groups["train"][index] for index in order[start : start + batch]]
            decoded = [_decode(row, data_root, ignore, variant) for row in chunk]
            s2 = torch.stack(
                [(torch.from_numpy(frames) - mean_t) / std_t for frames, _, _ in decoded]
            ).to(device)
            labels = torch.stack([torch.from_numpy(t) for _, t, _ in decoded]).to(device)
            position = torch.tensor([p for _, _, p in decoded], dtype=torch.float32, device=device)
            with torch.no_grad():
                dense = backbone(
                    {"s2": s2, "s2_dates": position},
                    patch_size=patch_m,
                    output="dense",
                    output_modality="s2",
                )
            logits = F.interpolate(
                head([_dense_grid(dense)]),
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = F.cross_entropy(logits, labels, ignore_index=ignore) / accum
            loss.backward()
            if (steps + 1) % accum == 0:
                optimiser.step()
                optimiser.zero_grad()
                total_steps += 1
            total += float(loss.detach()) * accum
            steps += 1
        if steps % accum:
            optimiser.step()
            optimiser.zero_grad()
            total_steps += 1
        emit(
            request.run_id,
            "train.epoch",
            f"epoch {epoch}: loss {total / max(steps, 1):.4f}",
            progress=(epoch + 1) / epochs,
        )

    for split in ("validation", "test"):
        confusion = np.zeros((task_spec.num_outputs, task_spec.num_outputs), dtype=np.int64)
        head.eval()
        evaluated = 0
        with torch.no_grad():
            for start in range(0, len(groups[split]), batch):
                chunk = groups[split][start : start + batch]
                evaluated += len(chunk)
                decoded = [_decode(row, data_root, ignore, variant) for row in chunk]
                s2 = torch.stack(
                    [(torch.from_numpy(f) - mean_t) / std_t for f, _, _ in decoded]
                ).to(device)
                labels = torch.stack([torch.from_numpy(t) for _, t, _ in decoded]).to(device)
                position = torch.tensor(
                    [p for _, _, p in decoded], dtype=torch.float32, device=device
                )
                dense = backbone(
                    {"s2": s2, "s2_dates": position},
                    patch_size=patch_m,
                    output="dense",
                    output_modality="s2",
                )
                predicted = F.interpolate(
                    head([_dense_grid(dense)]),
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).argmax(dim=1)
                mask = labels != ignore
                flat = (labels[mask] * task_spec.num_outputs + predicted[mask]).cpu().numpy()
                confusion += np.bincount(flat, minlength=task_spec.num_outputs**2).reshape(
                    task_spec.num_outputs, task_spec.num_outputs
                )
        expected = (
            OFFICIAL_TEST_SAMPLES["T11"] if split == "test" else OFFICIAL_VALIDATION_SAMPLES["T11"]
        )
        if evaluated != expected:
            raise AnySatT11Failure(
                f"{split} fold evaluated on {evaluated} of the official {expected} rows; "
                "a shrunken manifest must not self-attest"
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
            + f"AnySat base frozen S2 entrance; variant {variant}; "
            f"token spacing {patch_m} m; UPerNetHead(1536->fpn256) on the stride-8 "
            "grid; reflectance DN/10000 "
            "z-scored per band on the train fold; "
            f"host {tags['host']}; labels 2019 pending LPIS-year confirmation"
        )
    )
    emit(
        request.run_id,
        "train.done",
        f"anysat {variant}: test mIoU {test.value:.4f}",
        progress=1.0,
    )
    return 0
