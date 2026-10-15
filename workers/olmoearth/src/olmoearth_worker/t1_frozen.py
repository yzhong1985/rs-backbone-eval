"""OlmoEarth v1.2 Tiny worker: T1 (Sen1Floods11) frozen head training.

Two paths share one frozen Tiny encoder and the six-band constant-fill input
adapter:

- ``probe``: stage A integration check (manifest + checkpoint + one forward).
- ``train``: stage B, the frozen-probe recipe shared with Prithvi and Clay —
  features are extracted once, the shared ``LiteSpatialHead`` trains 50 epochs
  on the full 512 label grid with AdamW lr 1e-3, and evaluation uses the
  grouped confusion matrix. Because no OlmoEarth band-set is fully covered by
  the task's six bands, runs are tagged ``track=frozen_adapter`` (the input
  adapter is required), not a pure frozen probe.

See ``docs/models/OlmoEarth接入进展.md`` for the probed upstream contract.
"""

from __future__ import annotations

import datetime
import hashlib
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
from rs_backbone_eval.data.manifest import audit_task_manifest
from rs_backbone_eval.heads import LiteSpatialHead, count_parameters
from rs_backbone_eval.registry import ProtocolSpec, TaskSpec, load_yaml_spec
from rs_backbone_eval.repro import random_init_requested, seed_everything
from rs_backbone_eval.splits import evaluation_split
from rs_backbone_eval.worker_runtime import RunWriter

from .bands import (
    OLMOEARTH_S2L2A_BANDS,
    bandset_usable,
    bandset_usable_after_drop,
    to_olmoearth_s2l2a,
)

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
EVAL_BATCH = 16
LABEL_SIZE = 512
OFFICIAL_EPOCHS = 50


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
        # __getitem__ returns NumPy arrays; torch.cat needs tensors, and
        # silently relying on the DataLoader's collate would lose this path.
        collected.append(torch.as_tensor(item["image"]).reshape(len(PROJECT_BANDS), -1))
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
    *,
    perturbation: str | None = None,
) -> tuple[torch.Tensor, list[torch.Tensor], list[str]]:
    """Expand six-band chips through the adapter and return dense token maps.

    ``features`` has shape ``(B, embed_dim, grid, grid)`` ready for the shared
    spatial head; labels keep their original resolution for the full-label loss,
    and group ids are carried through for per-group confusion matrices.

    An optional evaluation ``perturbation`` applies to the standardized chip:
    ``band_drop:`` replaces the named bands with the mean fill and masks any
    band-set left without one real band, ``resolution:`` downsample/restores.
    """
    from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue

    from rs_backbone_eval.perturb import (
        apply_band_drop,
        apply_resolution,
        band_indices_to_drop,
        resolve,
    )

    spec = resolve(perturbation) if perturbation else {"kind": "none"}
    if spec["kind"] not in {"none", "band_drop", "resolution"}:
        raise ValueError(f"unsupported perturbation for the OlmoEarth probe: {perturbation}")
    drop = band_indices_to_drop(PROJECT_BANDS, spec["bands"]) if spec["kind"] == "band_drop" else []
    # Sentinel-2 masks are band-set level (3 sets): a set stays ONLINE while at
    # least one of its bands is real, otherwise the whole set is MISSING.
    usable = (
        bandset_usable_after_drop(PROJECT_BANDS, drop) if drop else bandset_usable(PROJECT_BANDS)
    )
    online = [MaskValue.ONLINE_ENCODER.value if ok else MaskValue.MISSING.value for ok in usable]
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
        if spec["kind"] == "band_drop":
            chips = apply_band_drop(chips, drop)
        elif spec["kind"] == "resolution":
            chips = apply_resolution(chips, int(spec["factor"]))
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
    return torch.cat(features), torch.cat(labels), groups


class OlmoEarthFrozenProbe(nn.Module):
    """Frozen Tiny encoder plus the shared ``LiteSpatialHead``; features are cached."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.head = LiteSpatialHead(EMBED_DIM, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


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
    if task_spec.task_id != "T1":
        return _fail(
            writer,
            run_id,
            "probe.task",
            f"the OlmoEarth worker implements T1 only, not {task_spec.task_id} "
            "(T5 needs its own read-out decision; see docs/models/OlmoEarth接入进展.md)",
        )
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


def _evaluate_features(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    dataset: T1ChipDataset,
    *,
    collect_groups: bool = False,
) -> tuple[dict[str, float], dict[str, list[list[int]]]]:
    """Decode cached features and score them on the full 512 label grid.

    ``collect_groups`` adds the per-tile (group) confusion used for bootstrap
    intervals; perturbation runs skip groups and use the plain matrix, like the
    other workers.
    """
    from rs_backbone_eval.evaluation.confusion import ConfusionMatrix
    from rs_backbone_eval.evaluation.groups import GroupedSegmentation
    from rs_backbone_eval.evaluation.segmentation import binary_metrics

    device = next(model.parameters()).device
    collector: ConfusionMatrix | GroupedSegmentation
    if collect_groups:
        collector = GroupedSegmentation(num_classes=2, ignore_index=-1)
    else:
        collector = ConfusionMatrix(num_classes=2)
    model.eval()
    with torch.inference_mode():
        for start in range(0, features.shape[0], EVAL_BATCH):
            logits = model(features[start : start + EVAL_BATCH].to(device))
            upsampled = torch.nn.functional.interpolate(
                logits,
                size=(LABEL_SIZE, LABEL_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            targets = labels[start : start + EVAL_BATCH].numpy()
            prediction = upsampled.argmax(dim=1).cpu().numpy()
            if collect_groups:
                collector.update(  # type: ignore[union-attr]
                    [row["group_id"] for row in dataset.rows[start : start + EVAL_BATCH]],
                    targets,
                    prediction,
                )
            else:
                collector.update(targets, prediction, ignore_index=-1)
    matrix = collector.global_matrix if collect_groups else collector
    metrics = binary_metrics(matrix, foreground=1)
    metrics["water_iou"] = metrics.pop("foreground_iou")
    if not collect_groups:  # plain ConfusionMatrix carries no per-group data
        return metrics, None
    return metrics, collector.group_confusion()


def frozen_backbone_protocol(protocol_config_path: Path) -> tuple[bool, str]:
    """May this stage-B recipe run under the requested protocol?

    Two guards:
    - the recipe trains only a head over a frozen backbone, so a protocol that
      also trains the backbone (full_finetune) is rejected;
    - the six-band constant-fill adapter means every run must be *recorded as*
      ``frozen_adapter``: reports group by ``run.protocol_id``, which a
      ``tags.track`` label cannot override, so ``frozen_probe`` is rejected as
      well — a frozen_probe record would present an adapter run as a pure
      frozen result.
    """
    protocol = load_yaml_spec(protocol_config_path, ProtocolSpec)
    if protocol.protocol_id != "frozen_adapter":
        return (
            False,
            f"protocol {protocol.protocol_id} cannot run here: the six-band constant-fill "
            "adapter means every run must be recorded as frozen_adapter "
            "(reports group by protocol_id, not by tags)",
        )
    if protocol.train_backbone or not protocol.train_head:
        return (
            False,
            f"protocol {protocol.protocol_id} requires "
            f"train_backbone={protocol.train_backbone} train_head={protocol.train_head}; "
            "the worker implements only frozen-backbone head training",
        )
    return True, ""


def train_action(request: WorkerRequest, writer: RunWriter, event_log: Callable[..., None]) -> int:
    """Stage B: frozen-encoder/head training on the six-band adapter input.

    Same recipe as the other T1 frozen probes: frozen encoder features are
    extracted once, the shared ``LiteSpatialHead`` trains for 50 epochs with
    AdamW lr 1e-3 on the full label grid, and evaluation uses the grouped
    confusion matrix. The constant-fill band adapter is **not** trainable, so
    the track is ``frozen_adapter``, not a pure frozen probe.
    """
    from rs_backbone_eval.evaluation import to_metric_records
    from rs_backbone_eval.perturb import describe, perturbations_from_env, resolve

    run_id = request.run_id
    # Set the recorded seed before anything else, so head init, shuffling and
    # dropout reproduce from the run record even when called directly.
    seed_everything(request.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = _int_env("OLMOEARTH_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("OLMOEARTH_WORKER_BATCH_SIZE", BATCH_SIZE)
    lr = _float_env("OLMOEARTH_WORKER_LR", 1e-3)
    limit = _int_env("OLMOEARTH_WORKER_LIMIT_BATCHES", 0) or None
    stats_rows = _int_env("OLMOEARTH_WORKER_STATS_ROWS", 64)

    event_log(run_id, "train.start", f"frozen head training ({epochs} epochs, lr {lr:g})")
    task_spec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    if task_spec.task_id != "T1":
        return _fail(
            writer,
            run_id,
            "train.task",
            f"the OlmoEarth worker implements T1 only, not {task_spec.task_id} "
            "(T5 needs its own read-out decision; see docs/models/OlmoEarth接入进展.md)",
        )
    protocol_ok, protocol_message = frozen_backbone_protocol(Path(request.protocol_config_path))
    if not protocol_ok:
        # A finetune protocol must not silently run this recipe, or the record
        # would be kept under the wrong regime and poison the comparison.
        return _fail(writer, run_id, "train.protocol", protocol_message)
    audit = audit_task_manifest(Path(request.manifest_path), task_spec)
    if audit.error_count - len(audit.leakage):
        return _fail(
            writer, run_id, "train.manifest", f"manifest audit failed: {audit.all_errors()[:2]}"
        )
    checkpoint = os.environ.get("OLMOEARTH_CHECKPOINT")
    if not checkpoint or not (Path(checkpoint) / "weights.pth").is_file():
        return _fail(
            writer,
            run_id,
            "train.checkpoint",
            f"{checkpoint or 'OLMOEARTH_CHECKPOINT'} does not contain weights.pth",
        )
    checkpoint_dir = Path(checkpoint)
    record_checkpoint_hash(writer, checkpoint_dir)

    data_root = Path(request.data_root)
    manifest = Path(request.manifest_path)
    train_ds = T1ChipDataset(manifest, data_root, "train")
    eval_ds = T1ChipDataset(manifest, data_root, EVAL_SPLIT)
    if not train_ds or not eval_ds:
        return _fail(writer, run_id, "train.data", "train or test split is empty")

    encoder = build_encoder(checkpoint_dir, load_weights=not random_init_requested()).to(device)
    encoder_params = count_parameters(encoder)
    model = OlmoEarthFrozenProbe(encoder).to(device)
    head_params = count_parameters(model.head)
    writer.update_training(
        head_id="lite_spatial_head",
        trainable_parameters=head_params,
        total_parameters=encoder_params + head_params,
    )
    writer.update_input(preprocessing_id="train_stats_6band+constant_fill_adapter")
    writer.update(
        tags={
            "track": "frozen_adapter",
            "head_parameters": str(head_params),
            "encoder_parameters": str(encoder_params),
            "backbone_frozen": "True",
            "band_adapter": "constant_fill_12band",
            "adapter_note": "missing OlmoEarth bands filled with the standardised mean; "
            "band-sets with no real band masked MISSING",
            "encoder_input_px": str(ENCODER_INPUT),
            "feature_grid": f"{GRID}x{GRID}",
            "feature_stride_px": f"{PATCH_SIZE} (at 224 encoding, evaluated on the 512 grid)",
            "learning_rate": f"{lr:g}",
            "normalization_rows": str(stats_rows),
            "device": device,
        }
    )
    event_log(
        run_id,
        "train.model",
        f"frozen Tiny encoder {encoder_params} + head {head_params} params "
        f"({GRID}x{GRID} features at stride {PATCH_SIZE})",
        progress=0.1,
    )

    mean, std = band_statistics(train_ds, limit=stats_rows)
    means = mean.to(device).view(1, -1, 1, 1)
    stds = std.to(device).view(1, -1, 1, 1)

    started = time.time()
    extract_started = time.time()
    train_features, train_labels, _ = encode_dataset(
        train_ds, encoder, means, stds, device, event_log, run_id
    )
    eval_features, eval_labels, _ = encode_dataset(
        eval_ds, encoder, means, stds, device, event_log, run_id
    )
    phases = {"extract": time.time() - extract_started}
    event_log(
        run_id,
        "train.cache",
        f"frozen features ready: train {tuple(train_features.shape)}, "
        f"test {tuple(eval_features.shape)}",
        progress=0.4,
    )

    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    label_size = train_labels.shape[-1]  # full-label loss on the original grid
    train_started = time.time()
    optimizer_steps = 0
    for epoch in range(epochs):
        permutation = torch.randperm(train_features.shape[0], device=device)
        total_loss, steps = 0.0, 0
        for start in range(0, permutation.numel(), batch_size):
            index = permutation[start : start + batch_size]
            logits = torch.nn.functional.interpolate(
                model(train_features[index]),
                size=(label_size, label_size),
                mode="bilinear",
                align_corners=False,
            )
            loss = criterion(logits, train_labels[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            steps += 1
            if limit and steps >= limit:
                break
        optimizer_steps += steps
        mean_loss = total_loss / max(steps, 1)
        writer.add_curve("loss", "train", epoch, mean_loss)
        event_log(
            run_id,
            "train.epoch",
            f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}",
            progress=0.4 + 0.5 * (epoch + 1) / epochs,
        )
    phases["train"] = time.time() - train_started
    wall = time.time() - started

    event_log(run_id, "train.eval", f"evaluating on {len(eval_ds)} {EVAL_SPLIT} chips")
    eval_started = time.time()
    metrics, group_json = _evaluate_features(
        model, eval_features, eval_labels, eval_ds, collect_groups=True
    )
    phases["eval"] = time.time() - eval_started
    writer.add_metrics(
        to_metric_records(metrics, primary_name="water_iou", direction="maximize", split=EVAL_SPLIT)
    )
    writer.add_confusion(group_json, 2)
    for name in perturbations_from_env():
        if resolve(name)["kind"] == "none":
            continue
        perturbed_features, _, _ = encode_dataset(
            eval_ds, encoder, means, stds, device, event_log, run_id, perturbation=name
        )
        perturbed, _ = _evaluate_features(model, perturbed_features, eval_labels, eval_ds)
        writer.add_metrics(
            to_metric_records(
                perturbed,
                primary_name="water_iou",
                direction="maximize",
                split=describe(name),
                primary=False,
            )
        )
        event_log(
            run_id,
            "train.perturb",
            f"{describe(name)}: water_iou={perturbed['water_iou']:.4f} "
            f"(baseline {metrics['water_iou']:.4f})",
        )
    if any(resolve(name)["kind"] != "none" for name in perturbations_from_env()):
        writer.update(
            tags={
                "perturbations": ",".join(perturbations_from_env()),
                "perturbation_note": "inference-time input robustness, not re-adaptation",
            }
        )
    writer.update_training(
        epochs_completed=epochs,
        optimizer_steps=optimizer_steps,
        wall_time_seconds=wall,
        peak_gpu_memory_mb=torch.cuda.max_memory_allocated() / (1024**2)
        if device == "cuda"
        else 0.0,
        phases=phases,
    )
    summary = (
        f"water_iou={metrics['water_iou']:.4f} mean_iou={metrics['mean_iou']:.4f} "
        f"({epochs} epochs, frozen Tiny encoder {encoder_params} + head {head_params} params, "
        f"{wall / 60:.1f} min)"
    )
    if epochs != OFFICIAL_EPOCHS or limit:
        summary += " [SMOKE]"
    writer.complete(summary)
    event_log(run_id, "train.metrics", summary)
    return 0
