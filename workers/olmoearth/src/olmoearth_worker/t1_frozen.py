"""OlmoEarth v1.2 Tiny worker: stage A (integration check) only.

This worker currently implements **probe** only: it validates the manifest and
checkpoint, builds the Tiny encoder, expands a real six-band chip through the
band adapter and runs one encoder forward, recording the token shape, parameter
counts and cost. Training is **not** implemented here yet; dispatching ``train``
fails with an explicit message.

Why no training yet: with the project's six bands **no OlmoEarth band-set is
covered completely**, so T1 needs an input adapter and must be reported as
``frozen_adapter`` once head training is wired. See
``docs/OlmoEarth接入进展.md`` for the probed contract and the remaining work.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.data.manifest import audit_task_manifest
from rs_backbone_eval.heads import count_parameters
from rs_backbone_eval.registry import TaskSpec, load_yaml_spec
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.splits import evaluation_split
from rs_backbone_eval.worker_runtime import RunWriter

from .bands import OLMOEARTH_S2L2A_BANDS, bandset_usable, to_olmoearth_s2l2a

PROJECT_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
# Positions of the task bands inside the 13-band Sen1Floods11 L1C chip raster,
# matching workers/prithvi/src/prithvi_worker/t1_frozen.py:CHIP_BAND_INDICES so
# every model reads the same channels.
CHIP_BAND_INDICES = {"BLUE": 1, "GREEN": 2, "RED": 3, "NIR_NARROW": 8, "SWIR_1": 11, "SWIR_2": 12}
EMBED_DIM = 192
ENCODER_INPUT = 224
PATCH_SIZE = 16
GRID = ENCODER_INPUT // PATCH_SIZE
EVAL_SPLIT = evaluation_split("test")
BATCH_SIZE = 4


def emit(run_id: str, event: str, message: str, **kwargs) -> None:
    from rs_backbone_eval.worker_runtime import emit_event

    emit_event(run_id, event, message, **kwargs)


def _fail(writer: RunWriter, run_id: str, event: str, message: str) -> int:
    emit(run_id, event, message, level="error")
    writer.fail(message)
    return 1


def build_encoder(checkpoint_dir: Path, *, load_weights: bool):
    """Build the Tiny model and expose its frozen ``encoder`` sub-module."""
    from olmoearth_pretrain.model_loader import load_model_from_path

    model = load_model_from_path(checkpoint_dir, load_weights=load_weights)
    encoder = model.encoder
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


class T1ChipDataset(Dataset):
    """Six-band flood chips, labels, group ids and timestamps from one split."""

    def __init__(self, manifest_path: Path, data_root: Path, split: str):
        self.data_root = data_root
        self.rows: list[dict] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["split"] == split:
                    self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        import rasterio

        row = self.rows[index]
        # The raster is a 13-band stack; the manifest names the task bands, so
        # select them by position before the adapter ever sees the chip.
        bands = list(row.get("bands", PROJECT_BANDS))
        try:
            indices = [CHIP_BAND_INDICES[band] for band in bands]
        except KeyError as error:
            raise ValueError(f"unknown task band {error} in {row['sample_id']}") from error
        with rasterio.open(self.data_root / row["image_paths"][0]) as source:
            image = source.read().astype(np.float32)[indices] * 0.0001
        with rasterio.open(self.data_root / row["label_path"]) as source:
            label = source.read(1).astype(np.int64)
        return {
            "image": image,
            "label": label,
            "group_id": row["group_id"],
            # A LongTensor [T, 3] collates to [B, T, 3]; a list of strings would
            # be transposed by the default collate (one entry per date).
            "timestamps": torch.tensor(_timestamps(row["timestamps"]), dtype=torch.long),
        }


def _timestamps(stamps: list[str]) -> list[list[int]]:
    """``[day, month-1, year]`` per date: the layout OlmoEarth expects."""
    out = []
    for stamp in stamps:
        date = datetime.date.fromisoformat(str(stamp)[:10])
        out.append([date.day, date.month - 1, date.year])
    return out


def band_statistics(
    dataset: T1ChipDataset, *, limit: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-band mean/std **after** the constant scale, from a sample of rows.

    Matches the other T1 workers' ``train_stats`` recipe so the six-band input is
    comparable; the adapter then fills the missing bands with 0, i.e. the mean.
    """
    collected: list[torch.Tensor] = []
    for index in range(min(limit, len(dataset))):
        item = dataset[index]
        collected.append(item["image"].reshape(len(PROJECT_BANDS), -1))
    if not collected:
        return torch.zeros(len(PROJECT_BANDS)), torch.ones(len(PROJECT_BANDS))
    pixels = torch.cat(collected, dim=1)
    return pixels.mean(dim=1), pixels.std(dim=1).clamp_min(1e-6)


def encode_dataset(
    dataset: T1ChipDataset,
    encoder,
    means: torch.Tensor,
    stds: torch.Tensor,
    device: str,
    emit: Callable[..., None],
    run_id: str,
) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
    """Expand six-band chips through the adapter and return dense token maps.

    ``features`` has shape ``(B, embed_dim, grid, grid)`` ready for the shared
    spatial head; labels keep their original resolution for the full-label loss,
    and group ids are carried through for per-group confusion matrices.
    """
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

    online = [
        MaskValue.ONLINE_ENCODER.value if ok else MaskValue.MISSING.value
        for ok in bandset_usable(PROJECT_BANDS)
    ]
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    groups: list[str] = []
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    for batch in loader:
        chips = batch["image"].to(device)
        chips = torch.nn.functional.interpolate(
            chips, size=(ENCODER_INPUT, ENCODER_INPUT), mode="bilinear", align_corners=False
        )
        # Standardise the six task bands *before* the adapter: the filled bands
        # then land on 0 (the standardised mean) and the statistics stay [1, 6, 1, 1].
        chips = (chips - means) / stds
        width = chips.shape[0]
        expanded = torch.empty(
            width, ENCODER_INPUT, ENCODER_INPUT, 1, len(OLMOEARTH_S2L2A_BANDS), device=device
        )
        for index in range(width):
            expanded[index] = (
                torch.from_numpy(
                    to_olmoearth_s2l2a(chips[index].cpu().numpy(), PROJECT_BANDS, fill=0.0)
                )
                .to(device)
                .permute(1, 2, 0)
                .unsqueeze(2)
            )
        mask = (
            torch.tensor(online, dtype=torch.long, device=device)
            .view(1, 1, 1, 1, len(online))
            .expand(width, ENCODER_INPUT, ENCODER_INPUT, 1, len(online))
        )
        sample = MaskedOlmoEarthSample(
            timestamps=batch["timestamps"].to(device),  # [B, T, 3] from the collate
            sentinel2_l2a=expanded,
            sentinel2_l2a_mask=mask,
        )
        with torch.inference_mode():
            tokens = encoder(sample, patch_size=PATCH_SIZE)["tokens_and_masks"]
            dense = tokens.sentinel2_l2a  # [B, P_H, P_W, T, BS, D]
            features.append(dense[:, :, :, 0, 0, :].permute(0, 3, 1, 2).contiguous().cpu())
        labels.append(batch["label"])
        groups.extend(batch["group_id"])
    emit(run_id, "probe.encode", f"encoded {len(dataset)} chips into {GRID}x{GRID} frozen tokens")
    return torch.cat(features), labels, groups


def record_checkpoint_hash(writer: RunWriter, checkpoint_dir: Path) -> None:
    """Record the SHA-256 of the downloaded weights."""
    digest = hashlib.sha256()
    with (checkpoint_dir / "weights.pth").open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    writer.update(model_checkpoint_sha256=digest.hexdigest())


def probe(request: WorkerRequest, writer: RunWriter, event_log: Callable[..., None]) -> int:
    """Stage A: validate the manifest/checkpoint and run one real forward pass."""
    run_id = request.run_id
    event_log(run_id, "probe.start", "validating manifest, checkpoint and encoder")

    task_spec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    audit = audit_task_manifest(Path(request.manifest_path), task_spec)
    if audit.error_count - len(audit.leakage):
        return _fail(
            writer, run_id, "probe.manifest", f"manifest audit failed: {audit.all_errors()[:2]}"
        )
    event_log(
        run_id,
        "probe.manifest",
        f"manifest OK: {audit.sample_count} samples, splits {dict(audit.split_counts)}",
        progress=0.2,
    )

    checkpoint = os.environ.get("OLMOEARTH_CHECKPOINT")
    if not checkpoint:
        return _fail(writer, run_id, "probe.checkpoint", "OLMOEARTH_CHECKPOINT is not set")
    checkpoint_dir = Path(checkpoint)
    if not (checkpoint_dir / "weights.pth").is_file():
        return _fail(
            writer,
            run_id,
            "probe.checkpoint",
            f"{checkpoint_dir} does not contain weights.pth (download via the catalog entry)",
        )
    record_checkpoint_hash(writer, checkpoint_dir)
    event_log(run_id, "probe.checkpoint", f"checkpoint OK: {checkpoint_dir}", progress=0.5)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = build_encoder(checkpoint_dir, load_weights=not random_init_requested()).to(device)
    encoder_params = count_parameters(encoder)
    event_log(
        run_id,
        "probe.encoder",
        f"encoder {encoder_params} params ({EMBED_DIM}-d tokens, {GRID}x{GRID} grid "
        f"from {ENCODER_INPUT}px at patch {PATCH_SIZE})",
        progress=0.7,
    )

    eval_ds = T1ChipDataset(Path(request.manifest_path), Path(request.data_root), EVAL_SPLIT)
    if not eval_ds:
        return _fail(writer, run_id, "probe.data", f"no rows in split {EVAL_SPLIT}")
    train_ds = T1ChipDataset(Path(request.manifest_path), Path(request.data_root), "train")
    if not train_ds:
        return _fail(writer, run_id, "probe.data", "no rows in split train")
    mean, std = band_statistics(train_ds)
    means = mean.to(device).view(1, -1, 1, 1)
    stds = std.to(device).view(1, -1, 1, 1)
    event_log(
        run_id,
        "probe.norm",
        f"train band stats over {len(train_ds)} rows: mean "
        f"{[round(v, 4) for v in mean.tolist()]} std {[round(v, 4) for v in std.tolist()]}",
    )
    features, labels, groups = encode_dataset(
        eval_ds, encoder, means, stds, device, event_log, run_id
    )
    event_log(
        run_id,
        "probe.forward",
        f"token map {tuple(features.shape)} over {len(labels)} chips, {len(groups)} groups",
        progress=1.0,
    )
    writer.update(
        tags={
            "encoder_parameters": str(encoder_params),
            "token_dim": str(EMBED_DIM),
            "token_grid": f"{GRID}x{GRID}",
            "eval_split": EVAL_SPLIT,
            "band_adapter": "constant_fill_12band",
            "device": device,
        }
    )
    writer.complete(
        f"probe passed: OlmoEarth v1.2 Tiny encoder {encoder_params / 1e6:.1f}M params; "
        f"{len(groups)} {EVAL_SPLIT} chips encoded to {tuple(features.shape)}"
    )
    return 0


def train_action(request: WorkerRequest, writer: RunWriter, event_log: Callable[..., None]) -> int:
    """Not implemented: stage B (head training) is next, see docs/OlmoEarth接入进展.md."""
    return _fail(
        writer,
        request.run_id,
        "train.unimplemented",
        "olmoearth worker implements probe only; head training is the next stage "
        "(band adapter must be reported as frozen_adapter)",
    )
