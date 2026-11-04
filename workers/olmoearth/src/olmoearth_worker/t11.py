"""OlmoEarth v1.2 on PASTIS-HD (T11, single variant).

PASTIS-HD carries ten S2 L2A bands; OlmoEarth's SENTINEL2_L2A entry expects
twelve grouped into three band-sets:

* set 0 ``[B02,B03,B04,B08]``   -- fully covered by PASTIS;
* set 1 ``[B05,B06,B07,B8A,B11,B12]`` -- fully covered;
* set 2 ``[B01,B09]``           -- absent, masked MISSING (no zero-fill lie).

So unlike T1, this is a *clean* adapter: two sets ONLINE, one MISSING, with
zeros only under a masked set. The frozen encoder returns one token grid per
date; the single variant passes the year-mid frame with its real acquisition
date and the shared ``UPerNetHead`` (fpn 256) decodes the grid, upsampled to
labels; this is the same decoder LIM, AnySat and Clay use on T11.

Only the single variant is registered (operator decision 2026-11-04: the
per-pixel temporal median composes a pseudo-scene backbones never saw).
Budget mirrors the T11 AEF probe: 8 epochs, AdamW lr 1e-3, effective batch 8
(4 x 2 accumulation), frozen encoder, coverage anchored on 496/482.
"""

from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from rs_backbone_eval.contracts import ManifestSample, MetricRecord, WorkerRequest
from rs_backbone_eval.data.manifest import audit_task_manifest, manifest_sha256
from rs_backbone_eval.heads import UPerNetHead, count_parameters
from rs_backbone_eval.matrix import OFFICIAL_TEST_SAMPLES, OFFICIAL_VALIDATION_SAMPLES

from .bands import OLMOEARTH_S2L2A_BANDS, bandset_coverage, to_olmoearth_s2l2a
from .t1_frozen import ENCODER_INPUT, PATCH_SIZE, build_encoder

VOID_CLASS = 19
FALLBACK_IGNORE = 19
YEAR_DATE_ANCHOR = 20190715
EPOCHS = 8
BATCH = 4
ACCUM = 2
TRAIN_MIN = 1400
PASTIS_GSD_M = 10
TRAIN_PATCH_PX = 128  # PASTIS-HD S2/annotation grid, 10 m
# The adapter upsamples the 128 px patch to 224 for the encoder. The encoder's
# spatial positional encoding uses input_res, so passing the native 10 m would
# claim 10 m pixels on an interpolated 5.7 m grid; pass the effective GSD
# (review 2026-11-04 round 8).
ENCODER_INPUT_RES_M = PASTIS_GSD_M * TRAIN_PATCH_PX / ENCODER_INPUT
PASTIS_BANDS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")


class OlmoEarthT11Failure(Exception):
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


def _decode_single(row: ManifestSample, data_root: Path, ignore: int):
    aux = row.aux_paths or {}
    s2_rel = aux.get("s2_path")
    if not s2_rel:
        raise OlmoEarthT11Failure(f"sample {row.sample_id}: aux_paths has no s2_path")
    sidecar = (data_root / row.image_paths[0]).with_suffix(".json")
    if not sidecar.is_file():
        raise OlmoEarthT11Failure(f"sample {row.sample_id}: sidecar {sidecar.name} missing")
    with sidecar.open("r", encoding="utf-8") as handle:
        dates = json.loads(handle.read()).get("dates_s2") or []
    stack = np.load(data_root / s2_rel, mmap_mode="r")
    if stack.shape[0] != len(dates):
        raise OlmoEarthT11Failure(
            f"sample {row.sample_id}: S2 stack T={stack.shape[0]} vs sidecar {len(dates)}"
        )
    anchor = int(aux.get("frame_date_anchor") or YEAR_DATE_ANCHOR)
    index = int(np.argmin(np.abs(np.asarray(dates, dtype=np.int64) - anchor)))
    pixels = np.asarray(stack[index], dtype=np.float32) * 0.0001  # reflectance
    labels = np.load(data_root / row.label_path)[0].astype(np.int64)
    if labels.shape != pixels.shape[-2:]:
        raise OlmoEarthT11Failure(f"sample {row.sample_id}: label grid {labels.shape} mismatch")
    stamp = str(dates[index])
    parsed = datetime.date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))
    return (
        pixels,
        np.where(labels == VOID_CLASS, ignore, labels),
        [parsed.day, parsed.month - 1, parsed.year],
    )


def _tile_sample(rows, limit: int) -> list[ManifestSample]:
    split = sorted((r for r in rows if r.split == "train"), key=lambda r: int(r.sample_id))
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


def _normalization(rows, data_root: Path, cache_dir: Path, ignore: int, manifest_digest: str):
    """Train-fold tile statistics, keyed by the data/recipe identity.

    A name-only key silently reused statistics after a manifest or recipe
    change, across seeds and runs (review 2026-11-04 round 8, same defect as
    AnySat round 7).
    """
    key = f"single-{manifest_digest[:16]}-p24-v{NORM_CACHE_VERSION}"
    cache = cache_dir / f"olmoearth_t11_norm_{key}.json"
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return np.asarray(payload["mean"], np.float32), np.asarray(payload["std"], np.float32)
    total = np.zeros(10, dtype=np.float64)
    squares = np.zeros(10, dtype=np.float64)
    seen = 0
    for row in _tile_sample(rows, 24):
        pixels, _, _ = _decode_single(row, data_root, ignore)
        total += pixels.sum(axis=(1, 2))
        squares += (pixels**2).sum(axis=(1, 2))
        seen += pixels.shape[1] * pixels.shape[2]
    mean = total / max(seen, 1)
    std = np.sqrt(np.maximum(squares / max(seen, 1) - mean**2, 0.0)) + 1e-6
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            {
                "mean": list(map(float, mean)),
                "std": list(map(float, std)),
                "patches": 24,
                "manifest_sha256": manifest_digest,
                "cache_version": NORM_CACHE_VERSION,
                "positions": "single year-mid frame",
            }
        ),
        encoding="utf-8",
    )
    os.replace(temporary, cache)
    return mean.astype(np.float32), std.astype(np.float32)


def _batch_input(
    decoded, mean: np.ndarray, std: np.ndarray, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chunks -> (12-band 224 tiles, band-set mask, timestamps).

    Standardisation happens in numpy, so it cannot mix a CPU tensor with the
    CUDA statistics tensor (review 2026-11-04 round 8).
    """
    batch = len(decoded)
    expanded = torch.zeros(batch, ENCODER_INPUT, ENCODER_INPUT, 1, len(OLMOEARTH_S2L2A_BANDS))
    for index, (pixels, _, _) in enumerate(decoded):
        standardized = torch.from_numpy((pixels - mean[:, None, None]) / std[:, None, None])
        resized = F.interpolate(
            standardized[None],
            size=(ENCODER_INPUT, ENCODER_INPUT),
            mode="bilinear",
            align_corners=False,
        )[0]
        expanded[index] = (
            torch.from_numpy(to_olmoearth_s2l2a(resized.numpy(), PASTIS_BANDS, fill=0.0))
            .permute(1, 2, 0)
            .unsqueeze(2)
        )
    # set0 and set1 are fully covered by PASTIS; set2 [B01,B09] is absent.
    # MaskValue: ONLINE_ENCODER=0, TARGET_ENCODER_ONLY=1, DECODER=2, MISSING=3 —
    # a 1 here would send the empty set to the target encoder, not drop it.
    mask = torch.tensor([0, 0, 3], dtype=torch.long).view(1, 1, 1, 1, 3)
    mask = mask.expand(batch, ENCODER_INPUT, ENCODER_INPUT, 1, 3)
    stamps = torch.tensor([d for _, _, d in decoded], dtype=torch.long).unsqueeze(1)
    return expanded.to(device), mask.to(device), stamps.to(device)


def _encode(encoder, expanded, mask, stamps) -> torch.Tensor:
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample

    sample = MaskedOlmoEarthSample(
        timestamps=stamps, sentinel2_l2a=expanded, sentinel2_l2a_mask=mask
    )
    tokens = encoder(sample, patch_size=PATCH_SIZE, input_res=ENCODER_INPUT_RES_M)[
        "tokens_and_masks"
    ]
    dense = tokens.sentinel2_l2a  # [B, P_H, P_W, T, BS, D]
    return dense[:, :, :, 0, 0, :].permute(0, 3, 1, 2).contiguous()


def train_t11_single(
    request: WorkerRequest,
    writer,
    emit,
    *,
    protocol,
    task_spec,
    checkpoint_dir: Path,
) -> int:
    from rs_backbone_eval.repro import seed_everything

    seed_everything(request.seed)
    if protocol.protocol_id != "olmoearth_s2_single":
        raise OlmoEarthT11Failure(
            f"olmoearth T11 runs olmoearth_s2_single only, got {protocol.protocol_id}"
        )
    ignore = task_spec.ignore_index if task_spec.ignore_index is not None else FALLBACK_IGNORE
    # task_spec.bands names the AEF embedding channels; the S2 payload this
    # adapter reads is the aux s2_path raster, whose ten bands are PASTIS_BANDS.
    coverage = bandset_coverage(PASTIS_BANDS)
    # set2 [B01,B09] is the only gap; the adapter masks it MISSING, so this run
    # is a clean mapping rather than a constant-fill adapter (see frozen_backbone_protocol).
    if coverage != (True, True, False):
        raise OlmoEarthT11Failure(f"unexpected S2 band-set coverage: {coverage}")
    data_root = Path(request.data_root)
    cache_dir = Path(request.cache_root)

    rows = _rows(request)
    audit = audit_task_manifest(Path(request.manifest_path), task_spec)
    if not audit.valid:
        raise OlmoEarthT11Failure(f"manifest audit failed: {audit.all_errors()[:5]}")
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
        raise OlmoEarthT11Failure(
            f"train fold has {len(groups['train'])} rows; the protocol needs >= {TRAIN_MIN}"
        )

    device, provenance = training_device()
    encoder = build_encoder(checkpoint_dir, load_weights=True).to(device).eval()
    mean, std = _normalization(
        groups["train"],
        data_root,
        cache_dir,
        ignore,
        manifest_sha256(Path(request.manifest_path)),
    )

    # embed dim from one real forward: Tiny 192, Base 768 (same as t1_frozen)
    probe_pixels, _, probe_stamp = _decode_single(groups["train"][0], data_root, ignore)
    probe = _batch_input([(probe_pixels, None, probe_stamp)], mean, std, device)
    with torch.inference_mode():
        embed_dim = int(_encode(encoder, *probe).shape[1])
    head = UPerNetHead([embed_dim], task_spec.num_outputs, fpn_dim=256).to(device)

    batch = _int_env("OLMOEARTH_T11_BATCH", BATCH)
    accum = _int_env("OLMOEARTH_T11_ACCUM", ACCUM)
    epochs = _int_env("OLMOEARTH_T11_EPOCHS", EPOCHS)

    started = time.time()
    writer.update_training(
        head_id="olmoearth_s2_single_upernet_fpn256",
        trainable_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
        total_parameters=count_parameters(head),
    )
    writer.update_input(preprocessing_id="olmoearth_s2_single_tilestats")
    tags = dict(provenance)
    tags.update(
        {
            "norm_source": "train_fold_tile_stats",
            "true_input": (
                "aux s2_path single year-mid frame (10,128,128) reflectance, 12-band "
                "OlmoEarth S2 L2A adapter (sets 0/1 online, set2 [B01,B09] MISSING)"
            ),
            "adapter_bandsets": "set0 online, set1 online, set2 missing",
            "embed_dim": str(embed_dim),
            "olmoearth_batch": f"{batch}x{accum}",
            "encoder_input_px": str(ENCODER_INPUT),
            "encoder_input_res_m": f"{ENCODER_INPUT_RES_M:.3f}",
            "test_manifest_samples": str(len(groups["test"])),
            "test_expected_samples": str(OFFICIAL_TEST_SAMPLES["T11"]),
            "validation_manifest_samples": str(len(groups["validation"])),
            "validation_expected_samples": str(OFFICIAL_VALIDATION_SAMPLES["T11"]),
        }
    )
    hf_endpoint = os.environ.get("RS_EVAL_HF_ENDPOINT", "")
    if hf_endpoint:
        tags["hf_endpoint"] = hf_endpoint
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
        total, steps = 0.0, 0
        optimiser.zero_grad()
        for start in range(0, len(groups["train"]), batch):
            chunk = [groups["train"][index] for index in order[start : start + batch]]
            decoded = [_decode_single(row, data_root, ignore) for row in chunk]
            expanded, mask, stamps = _batch_input(decoded, mean, std, device)
            labels = torch.stack([torch.from_numpy(t) for _, t, _ in decoded]).to(device)
            with torch.no_grad():
                features = _encode(encoder, expanded, mask, stamps)
            logits = F.interpolate(
                head([features]), size=labels.shape[-2:], mode="bilinear", align_corners=False
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
                decoded = [_decode_single(row, data_root, ignore) for row in chunk]
                expanded, mask, stamps = _batch_input(decoded, mean, std, device)
                labels = torch.stack([torch.from_numpy(t) for _, t, _ in decoded]).to(device)
                features = _encode(encoder, expanded, mask, stamps)
                logits = F.interpolate(
                    head([features]), size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                predicted = logits.argmax(dim=1)
                keep = labels != ignore
                flat = (labels[keep] * task_spec.num_outputs + predicted[keep]).cpu().numpy()
                confusion += np.bincount(flat, minlength=task_spec.num_outputs**2).reshape(
                    task_spec.num_outputs, task_spec.num_outputs
                )
        expected = (
            OFFICIAL_TEST_SAMPLES["T11"] if split == "test" else OFFICIAL_VALIDATION_SAMPLES["T11"]
        )
        if evaluated != expected:
            raise OlmoEarthT11Failure(
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
            + "OlmoEarth v1.2 encoder frozen; single year-mid frame; 12-band S2 L2A "
            "adapter (sets 0/1 online, set2 [B01,B09] MISSING); UPerNetHead(fpn 256); "
            f"224 px from 128 px at an effective {ENCODER_INPUT_RES_M:.2f} m; "
            f"host {tags['host']}; labels 2019 pending LPIS-year confirmation"
        )
    )
    emit(
        request.run_id,
        "train.done",
        f"olmoearth single: test mIoU {test.value:.4f}",
        progress=1.0,
    )
    return 0
