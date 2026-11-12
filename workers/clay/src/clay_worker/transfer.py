"""PASTIS-distilled RGB+NIR student transferred to FBP/T8 (diagnostic).

Frozen Clay encoder plus one readout, and a **new** head trained on the local
GF-2 labels. No teacher is read here: this is the "train a new task head on the
target domain" migration from the handoff (P2), not distillation.

Three readouts share everything else:

* ``clay_rgbn_t8_raw``   - the 1024-dim stride-8 tokens, no projection;
* ``clay_rgbn_t8_a64``   - the PASTIS-distilled AEF64 projector, frozen;
* ``clay_rgbn_t8_oe128`` - the PASTIS-distilled OlmoEarth128 projector, frozen.

Caveats recorded in every run:

* GF-2 is 4 m and normalised with **its own** train-scene mean/std, while the
  projector was distilled on 10 m S2 tokens normalised with Clay's S2 stats;
  the readout therefore sees a shifted feature distribution, and the raw-F arm
  is the control for that shift;
* the chip grid is the shared T8 protocol (224 px train draws, 64 px/scene
  fixed validation, full test grid);
* 8 epochs is a reduced budget against the 50-epoch T8 track, so these runs
  never enter the formal matrix.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.evaluation.groups import GroupedSegmentation
from rs_backbone_eval.evaluation.segmentation import multiclass_metrics
from rs_backbone_eval.gf2_chips import (
    TRAIN_CHIPS_PER_SCENE,
    VALIDATION_CHIPS_PER_SCENE,
    VALIDATION_DRAW_SEED,
    Gf2Chips,
    channel_statistics,
    epoch_seed,
    load_split,
)
from rs_backbone_eval.worker_runtime import RunWriter

from .clay_probe import add_upstream_to_path, prepare_encoder
from .t1 import CLAY_LARGE

BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD"]
WAVELENGTHS_UM = {"BLUE": 0.48, "GREEN": 0.56, "RED": 0.66, "NIR_BROAD": 0.83}
NUM_CLASSES = 25
IGNORE_INDEX = -1
GSD_M = 4.0
OFFICIAL_EPOCHS = 50
PATCH = 8
DEVICE = os.environ.get("CLAY_WORKER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# protocol_id -> (projector target, dims, source task) or raw 1024.
# The source task is explicit: T11D and T11P projectors now live side by side
# in the same (target, seed) directory and must not be selected by count
# (review 2026-11-12, P1).
TRANSFER_PROTOCOLS = {
    "clay_rgbn_t8_raw": (None, 1024, None),
    "clay_rgbn_t8_a64": ("aef64", 64, "T11D"),
    "clay_rgbn_t8_oe128": ("oe128", 128, "T11D"),
}


class TransferFailure(Exception):
    """A gate turns into a failed run with a readable reason."""


def scene_root() -> Path:
    return Path(os.environ.get("RS_EVAL_FBP_ROOT", "/geodata/Five-Billion-Pixels")).resolve()


def _projector_source_task(candidate: Path, payload: dict, runs_root: Path) -> str | None:
    """Task that produced a projector, from the payload or its source run.

    Older projector payloads do not carry ``task_id``; their ``source_run_id``
    points at the distill run, whose ``run.json`` does. This keeps the
    T11D/T11P coexistence safe without rewriting cached artifacts.
    """
    if payload.get("task_id"):
        return str(payload["task_id"])
    source_run = payload.get("source_run_id")
    if not source_run:
        return None
    record_path = runs_root / str(source_run) / "run.json"
    if not record_path.is_file():
        return None
    try:
        import json

        return str(json.loads(record_path.read_text(encoding="utf-8")).get("task_id") or "") or None
    except (OSError, ValueError):
        return None


def find_projector(
    target: str,
    seed: int,
    *,
    source_task: str,
    runs_root: Path,
    root: Path | None = None,
) -> tuple[Path, dict]:
    """The projector produced by ``source_task`` for (target, seed).

    Multiple experiments may have written a projector for the same target and
    seed (T11D four-band and T11P six-band). Selecting "the only file" is no
    longer valid; the source task is part of the contract (review 2026-11-12,
    P1).
    """
    from .rgbn import cache_root

    directory = (root or cache_root()) / "projector" / target / f"seed{seed}"
    candidates = sorted(directory.glob("*.pt")) if directory.is_dir() else []
    matching: list[tuple[Path, dict]] = []
    for candidate in candidates:
        payload = torch.load(candidate, map_location="cpu")
        if payload.get("target") != target:
            continue
        if _projector_source_task(candidate, payload, runs_root) == source_task:
            matching.append((candidate, payload))
    if len(matching) != 1:
        names = ", ".join(str(path.name) for path in candidates) or "none"
        raise TransferFailure(
            f"expected exactly one {source_task} projector for {target} seed {seed} "
            f"under {directory}; found {len(matching)} matching of [{names}]"
        )
    return matching[0]


def build_head(in_channels: int, hidden: int = 128) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 1),
        nn.GroupNorm(8, hidden),
        nn.GELU(),
        nn.Conv2d(hidden, NUM_CLASSES, 1),
    )


def build_projector(in_channels: int, dims: int, hidden: int = 256) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 1),
        nn.GELU(),
        nn.Conv2d(hidden, dims, 1),
    )


def _tokens(encoder, pixels, waves) -> Any:

    batch, _, height, width = pixels.shape
    datacube = {
        "pixels": pixels,
        "time": torch.zeros(batch, 4, device=pixels.device),
        "latlon": torch.zeros(batch, 4, device=pixels.device),
        "gsd": torch.tensor(GSD_M, device=pixels.device),
        "waves": waves,
        "platform": "sentinel-2-l2a",
    }
    with torch.no_grad():
        tokens = encoder(datacube)
    return tokens.transpose(1, 2).reshape(batch, CLAY_LARGE["dim"], height // PATCH, width // PATCH)


def train_t8_transfer(
    request: WorkerRequest,
    writer: RunWriter,
    emit,
    *,
    protocol,
    checkpoint: Path,
) -> int:
    add_upstream_to_path()
    from claymodel.finetune.segment.factory import SegmentEncoder

    target, dims, source_task = TRANSFER_PROTOCOLS[protocol.protocol_id]
    manifest = Path(request.manifest_path)
    train_ds = Gf2Chips(scene_root(), load_split(manifest, "train"))
    val_ds = Gf2Chips(scene_root(), load_split(manifest, "validation"))
    val_ds.set_chips(val_ds.draw(VALIDATION_CHIPS_PER_SCENE, VALIDATION_DRAW_SEED))
    test_ds = Gf2Chips(scene_root(), load_split(manifest, "test"))
    test_ds.set_chips(test_ds.full_grid())

    norm_chips = train_ds.draw(4, VALIDATION_DRAW_SEED)[: 4 * 20]
    means_array, stds_array = channel_statistics(train_ds, norm_chips)
    means = torch.tensor(means_array, device=DEVICE).view(1, -1, 1, 1)
    stds = torch.tensor(stds_array, device=DEVICE).view(1, -1, 1, 1)
    waves = torch.tensor(
        [WAVELENGTHS_UM[band] for band in BANDS], dtype=torch.float32, device=DEVICE
    )

    encoder = SegmentEncoder(**CLAY_LARGE, ckpt_path=str(checkpoint)).to(DEVICE)
    prepare_encoder(encoder, freeze_backbone=True)
    projector = None
    projector_path = None
    if target is not None:
        projector_path, payload = find_projector(
            target,
            request.seed,
            source_task=source_task,
            runs_root=Path(request.runs_root),
        )
        projector = build_projector(1024, dims).to(DEVICE)
        projector.load_state_dict(payload["state_dict"])
        for parameter in projector.parameters():
            parameter.requires_grad = False
        projector.eval()
        writer.update(
            tags={
                "transfer_projector": str(projector_path),
                "transfer_projector_source_run": str(payload.get("source_run_id", "")),
                "transfer_projector_source_task": source_task or "-",
                "transfer_projector_fingerprint": str(payload.get("feature_fingerprint", "")),
                "transfer_projector_sha256": _sha256(projector_path),
            }
        )
    head = build_head(dims).to(DEVICE)
    epochs = protocol.epochs or 8
    batch_size = 2
    optimiser = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-4)
    writer.update_training(
        head_id=f"t8_transfer_head_{dims}",
        trainable_parameters=sum(p.numel() for p in head.parameters()),
        total_parameters=sum(p.numel() for p in head.parameters()),
    )
    writer.update_input(preprocessing_id="gf2_train_stats_four_band")
    writer.update(
        tags={
            "device": DEVICE,
            "host": (
                torch.cuda.get_device_name(0).lower().replace(" ", "-")
                if DEVICE == "cuda"
                else "cpu"
            ),
            "torch": torch.__version__,
            "transfer_readout": "raw-1024" if target is None else f"pastis-{target}",
            "transfer_target_domain": "FBP/T8 GF-2 4 m, 25 classes",
            "transfer_normalization": "GF-2 train-scene mean/std (not Clay S2 stats)",
            "transfer_teacher_at_inference": "no",
            "transfer_epochs": str(epochs),
            "transfer_batch": str(batch_size),
            "transfer_lr": "0.003",
            "label_mapping": "0 -> ignore, 1..24 kept",
            "chip_protocol": (
                "224px; train 32/scene/epoch redrawn, validation 64/scene fixed, test full grid"
            ),
        }
    )

    def forward_readout(pixels):
        tokens = _tokens(encoder, pixels, waves)
        return tokens if projector is None else projector(tokens)

    def score(dataset: Gf2Chips) -> tuple[dict, dict[str, list[list[int]]]]:
        collector = GroupedSegmentation(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
        head.eval()
        with torch.inference_mode():
            for batch in DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2):
                pixels = (batch["image"].to(DEVICE) - means) / stds
                logits = head(forward_readout(pixels))
                logits = torch.nn.functional.interpolate(
                    logits, size=pixels.shape[-2:], mode="bilinear", align_corners=False
                )
                collector.update(
                    list(batch["group"]), batch["label"].numpy(), logits.argmax(1).cpu().numpy()
                )
        return multiclass_metrics(collector.global_matrix), collector.group_confusion()

    started = time.time()
    for epoch in range(epochs):
        train_ds.set_chips(train_ds.draw(TRAIN_CHIPS_PER_SCENE, epoch_seed(request.seed, epoch)))
        head.train()
        optimizer_loss, steps = 0.0, 0
        for batch in DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2):
            pixels = (batch["image"].to(DEVICE) - means) / stds
            labels = batch["label"].to(DEVICE)
            if int((labels != IGNORE_INDEX).sum()) == 0:
                continue
            logits = head(forward_readout(pixels))
            logits = torch.nn.functional.interpolate(
                logits, size=pixels.shape[-2:], mode="bilinear", align_corners=False
            )
            loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            optimizer_loss += float(loss.detach())
            steps += 1
        writer.add_curve("loss", "train", epoch + 1, optimizer_loss / max(steps, 1))
        emit(
            request.run_id,
            "transfer.epoch",
            f"epoch {epoch + 1}/{epochs} loss={optimizer_loss / max(steps, 1):.4f}",
            progress=(epoch + 1) / epochs,
        )

    validation, _ = score(val_ds)
    writer.add_curve("mean_iou", "validation", epochs, validation["mean_iou"])
    head_file = _save_head(
        writer,
        head,
        dims=dims,
        protocol=protocol.protocol_id,
        seed=request.seed,
        epochs=epochs,
        means=means_array,
        stds=stds_array,
        target=target,
        projector_path=projector_path,
        checkpoint=checkpoint,
    )
    emit(request.run_id, "transfer.head", f"wrote {head_file.name}")
    test, groups = score(test_ds)
    from rs_backbone_eval.evaluation import to_metric_records

    writer.add_metrics(
        to_metric_records(test, primary_name="mean_iou", direction="maximize", split="test")
    )
    writer.add_metrics(
        to_metric_records(
            validation,
            primary_name="mean_iou",
            direction="maximize",
            split="validation",
            primary=False,
        )
    )
    writer.add_confusion(groups, NUM_CLASSES)
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=round(time.time() - started, 1)
    )
    summary = (
        f"T8 transfer {target or 'raw-1024'}: test mIoU {test['mean_iou']:.4f}, "
        f"{epochs} epochs, frozen Clay + new head"
    )
    summary += " [REDUCED-BUDGET]" if epochs < OFFICIAL_EPOCHS else ""
    writer.complete(summary)
    emit(request.run_id, "transfer.done", summary)
    return 0


def _save_head(
    writer: RunWriter,
    head: nn.Module,
    *,
    dims: int,
    protocol: str,
    seed: int,
    epochs: int,
    means,
    stds,
    target: str | None,
    projector_path: Path | None,
    checkpoint: Path,
) -> Path:
    """Persist the trained target head so the run can be re-scored or inferred.

    Only the head is trained here; the encoder and projector stay frozen. The
    payload therefore records the readout chain it was trained on, copies the
    projector into the run directory (the cache can be pruned), and stores the
    normalisation statistics verbatim, because they come from a seeded chip
    draw rather than from the label manifest (review 2026-11-12, P1).
    """
    import io
    import shutil

    projector_record = None
    if projector_path is not None:
        local = writer.run_dir / "projector.pt"
        shutil.copy2(projector_path, local)
        projector_record = {
            "source": str(projector_path),
            "sha256": _sha256(projector_path),
            "local": local.name,
        }
        writer.add_artifact("checkpoint", local.name, "frozen PASTIS projector used by the head")
    payload = {
        "state_dict": head.state_dict(),
        "in_channels": dims,
        "num_classes": NUM_CLASSES,
        "bands": list(BANDS),
        "wavelengths_um": [WAVELENGTHS_UM[band] for band in BANDS],
        "gsd_m": GSD_M,
        "patch": PATCH,
        "chip_protocol": {
            "train_per_scene_per_epoch": TRAIN_CHIPS_PER_SCENE,
            "validation_per_scene": VALIDATION_CHIPS_PER_SCENE,
            "validation_draw_seed": VALIDATION_DRAW_SEED,
            "test": "full_grid",
        },
        "normalization": {
            "mean": [float(value) for value in means],
            "std": [float(value) for value in stds],
            "source": "gf2_train_stats_four_band",
        },
        "readout": "raw-1024" if target is None else f"pastis-{target}",
        "projector": projector_record,
        "encoder_checkpoint": str(checkpoint),
        "encoder_checkpoint_sha256": writer.record.model_checkpoint_sha256,
        "seed": seed,
        "protocol": protocol,
        "epochs": epochs,
        "official_epochs": OFFICIAL_EPOCHS,
        "reduced_budget": epochs < OFFICIAL_EPOCHS,
        "channel_policy": "GF-2 per-channel train statistics; label 0 -> ignore, 1..24 kept",
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path = writer.run_dir / "head.pt"
    temporary = path.with_suffix(".pt.tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)
    writer.add_artifact("checkpoint", path.name, "trained FBP/T8 readout head")
    return path


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
