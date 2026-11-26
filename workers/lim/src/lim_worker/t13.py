"""T13 (WHU-OPT-SAR optical) training for the image-only arms (plan v3.4 / G1-G3).

The LIM encoder is frozen and run live in fp16 (no feature cache: one train
epoch of 10,800 tiles costs ~2.7 min of encoder time, measured 2026-11-18).
Every arm trains on the same tiles, budget, schedule and flips; what changes is
declared per protocol in ``PROTOCOLS``:

=====================  =====  ===========================================
protocol               arm    note
=====================  =====  ===========================================
whu_b0(_lr*)           B0     head only, CE
whu_r                  R      head + deep injectors, real AEF
whu_r_null_zero        R      same, AEF replaced by zeros
whu_r_null_shuffle     R      same, AEF of the same tile position on a far sheet
whu_d                  D      B0 structure, CE + tau^2 KL to frozen whu_r
whu_bcap               Bcap   plugin on the locked whu_b0 head, random codebook, CE
whu_p                  P      same, train-fold AEF k-means codebook + alignment
whu_ft_b0              B0     full fine-tune: LIM + head, RGB, CE (no frozen encoder)
whu_ft_r               R      full fine-tune: LIM + head + AEF injectors, real AEF
=====================  =====  ===========================================

Dependencies (D's teacher, Bcap/P's head) are the completed, non-superseded run
of the named protocol with the same seed and model; their run ids and
checkpoint hashes are written into the tags.

Scoring is per sheet: tile logits are averaged into the full 5556 x 3704 canvas
(tiles overlap at stride 384), background pixels are ignored, and class 0 is
never predicted (it is never trained), so ``mean_iou`` is over the 7 classes.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch import nn
from torch.utils.data import DataLoader

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.data.manifest import manifest_sha256, require_no_group_leakage
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.confusion import GroupConfusion
from rs_backbone_eval.evaluation.segmentation import multiclass_metrics
from rs_backbone_eval.matrix import stopped_short_of_budget
from rs_backbone_eval.normalization import imagenet_rgb, normalization_seed
from rs_backbone_eval.registry import ProtocolSpec, TaskSpec
from rs_backbone_eval.schedule import build_scheduler
from rs_backbone_eval.selection import BestCheckpoint, validation_every
from rs_backbone_eval.worker_runtime import RunWriter

from .arms import (
    AEF_DIM,
    ArmModel,
    alignment_loss,
    decode_aef,
    distillation_loss,
    head_hash,
    pool_teacher,
    safe_cross_entropy,
)
from .mit import load_encoder, mit_huge
from .t13_data import (
    AEF_DIR,
    AEF_YEAR,
    IGNORE_INDEX,
    NUM_CLASSES,
    T13Tiles,
    band_statistics,
    donor_rows,
    load_rows,
    read_aef,
    read_galileo,
    read_image,
    read_label,
)

EFFECTIVE_BATCH = 8
CHANNELS = (256, 512, 1024, 2048)

PROTOCOLS: dict[str, dict] = {
    "whu_b0": {"arm": "B0"},
    "whu_b0_lr1e4": {"arm": "B0"},
    "whu_b0_lr3e3": {"arm": "B0"},
    "whu_r": {"arm": "R", "aef": "real"},
    "whu_r_null_zero": {"arm": "R", "aef": "zero"},
    "whu_r_null_shuffle": {"arm": "R", "aef": "shuffle"},
    "whu_d": {"arm": "D", "teacher": "whu_r", "kd_beta": 1.0, "kd_tau": 4.0},
    "whu_bcap": {"arm": "Bcap", "head": "whu_b0", "codebook": "random", "lambda_aef": 0.0},
    "whu_p": {"arm": "P", "head": "whu_b0", "codebook": "kmeans", "lambda_aef": 0.5},
    # Full fine-tune pair: LIM and the head train together. Only B0 and R have a
    # fine-tune recipe; D, Bcap and P stay frozen-backbone arms. The protocol must
    # declare regime full_finetune with train_backbone true; train_t13 checks both.
    "whu_ft_b0": {"arm": "B0", "finetune": True},
    # Four-band full fine-tune from the same LIM weights: NIR input weights start at zero,
    # validation only (the test split is scored by a separate locked evaluate).
    "whu_ft_b0_4band": {"arm": "B0", "finetune": True, "bands": 4, "test_at_end": False},
    "whu_ft_r": {"arm": "R", "aef": "real", "finetune": True},
}
FINETUNE_ARMS = ("B0", "R")
# Full-fine-tune checkpoint. Not ``best_checkpoint.pt``: that file holds only the head
# and is what find_dependency loads as a frozen teacher or head. A separate name keeps
# a fine-tuned backbone from ever being read as a frozen one.
FULL_CHECKPOINT = "best_full_checkpoint.pt"
FULL_FORMAT = "t13-full-finetune-v1"


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def find_dependency(
    runs_root: Path,
    *,
    task_id: str,
    protocol_id: str,
    seed: int,
    model_id: str,
    manifest_sha: str,
    backbone_sha: str,
    epochs: int,
    batch: int,
    smoke: bool = False,
) -> Path:
    """The one completed run of ``protocol_id`` that the current run may build on.

    A candidate must match seed and model, be completed and not superseded, and
    agree with the current run on what it trained on: manifest content, backbone
    checkpoint, epoch budget and effective batch. A formal run only accepts a
    formal dependency and a smoke run only a smoke one, so a smoke can never feed
    a formal teacher or head and vice versa. Zero or several matches raise.
    """
    hits = []
    for record in runs_root.glob("*/run.json"):
        try:
            run = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        tags = run.get("tags") or {}
        if not (
            run.get("task_id") == task_id
            and run.get("protocol_id") == protocol_id
            and run.get("seed") == seed
            and run.get("model_id") == model_id
            and run.get("status") == "completed"
            and not tags.get("superseded")
            and ("SMOKE" in (run.get("notes") or "")) == smoke
        ):
            continue
        if (
            (run.get("dataset") or {}).get("manifest_sha256") == manifest_sha
            and run.get("model_checkpoint_sha256") == backbone_sha
            and tags.get("epochs_budget") == str(epochs)
            and (run.get("training") or {}).get("epochs_completed") == epochs
            and tags.get("effective_batch") == str(batch)
        ):
            hits.append(record.parent)
    if len(hits) != 1:
        raise RuntimeError(
            f"need exactly one completed {protocol_id} run for seed {seed} / {model_id} with "
            f"manifest {manifest_sha[:12]}, backbone {backbone_sha[:12]}, {epochs} epochs and "
            f"batch {batch}; found {len(hits)}: {sorted(h.name for h in hits)}"
        )
    checkpoint = hits[0] / "best_checkpoint.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"{hits[0].name} has no best_checkpoint.pt")
    return checkpoint


def amp_context(device: str, enabled: bool):
    """bf16 autocast on CUDA; a no-op on CPU and whenever ``enabled`` is false.

    bf16 rather than fp16: the 4090 and 3080 (both Ampere/Ada) run bf16 tensor cores
    at the same rate as fp16, bf16 has the fp32 exponent range so the MiT attention
    logits cannot overflow, and no GradScaler is needed.
    """
    if enabled and device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class Inputs:
    """Turns a uint8 batch into encoder features, the plugin image and AEF.

    ``finetune=False`` (frozen arms): the encoder is fp16 on CUDA and every forward
    runs under ``no_grad``. ``finetune=True``: the encoder keeps fp32 weights, a
    forward keeps its graph when ``grad`` is set, and the arithmetic runs under
    bf16 autocast on CUDA.
    """

    def __init__(
        self,
        encoder,
        band_mean: np.ndarray,
        band_std: np.ndarray,
        device: str,
        *,
        finetune: bool = False,
        in_bands: int = 3,
    ):
        rgb_mean, rgb_std = imagenet_rgb()  # 0-255 scale, what LIM was pretrained on
        self.encoder = encoder
        self.device = device
        self.finetune = finetune
        self.in_bands = in_bands
        self.rgb_mean = torch.tensor(rgb_mean, device=device).view(1, 3, 1, 1)
        self.rgb_std = torch.tensor(rgb_std, device=device).view(1, 3, 1, 1)
        self.band_mean = torch.tensor(band_mean, device=device).view(1, 4, 1, 1)
        self.band_std = torch.tensor(band_std, device=device).view(1, 4, 1, 1)
        self.dtype = torch.float16 if device == "cuda" else torch.float32

    def autocast(self):
        return amp_context(self.device, self.finetune)

    def features(self, image: torch.Tensor, *, grad: bool = False) -> list[torch.Tensor]:
        rgb = (image[:, :3].float() - self.rgb_mean) / self.rgb_std
        if self.in_bands == 4:  # NIR normalised with its train-fold statistics
            nir = (image[:, 3:4].float() - self.band_mean[:, 3:4]) / self.band_std[:, 3:4]
            rgb = torch.cat([rgb, nir], dim=1)
        if not self.finetune:
            with torch.no_grad():
                return [level.float() for level in self.encoder(rgb.to(self.dtype))]
        with torch.set_grad_enabled(grad), self.autocast():
            levels = self.encoder(rgb)
        return [level.float() for level in levels]

    def plugin_image(self, image: torch.Tensor, bands: int = 4) -> torch.Tensor:
        z = (image.float() - self.band_mean) / self.band_std
        return z[:, :bands]


def aef_input(codes: torch.Tensor, mode: str) -> torch.Tensor:
    """Decoded AEF for arm R; ``zero`` replaces it by zeros (R_null).

    The shuffled control is not done here: its codes are already the donor
    tile's (``t13_data.donor_rows``), so it decodes like real AEF.
    """
    features, valid = decode_aef(codes)
    if mode == "zero":
        return torch.zeros_like(features)
    if mode not in ("real", "shuffle"):
        raise ValueError(f"unknown AEF mode {mode!r}")
    return features * valid


def kmeans_vectors(
    rows: list[dict], data_root: Path, tiles: int, per_tile: int, seed: int
) -> torch.Tensor:
    """Valid train-fold AEF vectors for the codebook (decoded, fixed sample)."""
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(rows), size=min(tiles, len(rows)), replace=False)
    out = []
    for index in sorted(int(i) for i in picks):
        features, valid = decode_aef(torch.from_numpy(read_aef(data_root, rows[index]))[None])
        flat = features[0].reshape(AEF_DIM, -1).T[valid.reshape(-1) > 0.5]
        take = rng.choice(flat.shape[0], size=min(per_tile, flat.shape[0]), replace=False)
        out.append(flat[torch.from_numpy(np.sort(take))])
    return torch.cat(out)


def score_sheets(
    rows: list[dict],
    data_root: Path,
    predict: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    *,
    with_aef: bool,
    batch: int,
    limit: int = 0,
    aef_source: dict[str, dict] | None = None,
    on_sheet: Callable[[np.ndarray, np.ndarray], None] | None = None,
    galileo_dir: Path | None = None,
) -> tuple[dict, dict]:
    """Average overlapping tile logits per sheet; confusion over labelled pixels.

    ``on_sheet(label, prediction)`` runs once per sheet when given; the test
    evaluation uses it for boundary scores.
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], []).append(row)
    keys = sorted(groups)[: limit or None]
    collector = GroupConfusion(NUM_CLASSES)
    uncovered = 0
    for key in keys:
        tiles = groups[key]
        with rasterio.open(data_root / tiles[0]["label_path"]) as src:
            height, width = src.height, src.width
        canvas = np.zeros((NUM_CLASSES, height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
        label = np.zeros((height, width), dtype=np.int64)
        for start in range(0, len(tiles), batch):
            chunk = tiles[start : start + batch]
            image = torch.from_numpy(np.stack([read_image(data_root, r) for r in chunk]))
            aef = (
                torch.from_numpy(
                    np.stack(
                        [
                            read_aef(data_root, aef_source[r["sample_id"]] if aef_source else r)
                            for r in chunk
                        ]
                    )
                )
                if with_aef
                else None
            )
            if galileo_dir is not None:
                # The second argument carries the precomputed Galileo tokens for this chunk.
                tokens = np.stack([read_galileo(galileo_dir, r) for r in chunk])
                logits = predict(image, torch.from_numpy(tokens.astype(np.float32)))
            else:
                logits = predict(image, aef)
            logits = logits.float().cpu().numpy()
            for row, tile_logits in zip(chunk, logits, strict=True):
                x, y, w, h = (int(v) for v in row["window"])
                canvas[:, y : y + h, x : x + w] += tile_logits
                counts[y : y + h, x : x + w] += 1
                label[y : y + h, x : x + w] = read_label(data_root, row)
        # 5555 px sheets leave their last column untiled (even tile origins);
        # those pixels keep label 0 and are ignored like background.
        uncovered += int((counts == 0).sum())
        canvas[IGNORE_INDEX] = -np.inf  # background is never trained, never predicted
        prediction = canvas.argmax(axis=0)
        if on_sheet is not None:
            on_sheet(label, prediction)
        collector.update(key, label, prediction, ignore_index=IGNORE_INDEX)
    metrics = multiclass_metrics(collector.combined())
    metrics["uncovered_pixels"] = float(uncovered)
    return metrics, collector.to_json()


class FullModel(nn.Module):
    """Encoder and arm model as one module, so BestCheckpoint keeps both together."""

    def __init__(self, encoder: nn.Module, model: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.model = model


def make_optimizer(
    head_params,
    backbone_params,
    *,
    head_lr: float,
    backbone_lr: float | None = None,
) -> torch.optim.AdamW:
    """AdamW over the head/injector group, plus the backbone group when fine-tuning.

    The frozen path passes no backbone parameters and gets one group at ``head_lr``,
    the same optimiser as before. Both groups share the scheduler, which scales each
    group from its own base learning rate.
    """
    groups = [{"params": list(head_params), "lr": head_lr}]
    backbone = list(backbone_params)
    if backbone:
        if backbone_lr is None:
            raise ValueError("a backbone parameter group needs backbone_lr")
        groups.append({"params": backbone, "lr": backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def save_full_checkpoint(best: BestCheckpoint, path: Path) -> None:
    """Write the validation-selected encoder and arm weights to ``FULL_CHECKPOINT``.

    The file holds ``encoder_state_dict`` (MiT-Huge, fp32) and ``state_dict`` (head
    and injectors), which ``load_full_checkpoint`` reads back.
    """
    if best.state is None:
        raise RuntimeError("no validation-selected weights to save")
    encoder = {k[len("encoder.") :]: v for k, v in best.state.items() if k.startswith("encoder.")}
    arm = {k[len("model.") :]: v for k, v in best.state.items() if k.startswith("model.")}
    if not encoder or not arm or len(encoder) + len(arm) != len(best.state):
        raise RuntimeError("best state is not an encoder/model FullModel state")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": FULL_FORMAT,
            "epoch": best.epoch,
            "score": best.score,
            "encoder_state_dict": encoder,
            "state_dict": arm,
        },
        path,
    )


def load_full_checkpoint(path: Path, encoder: nn.Module, model: nn.Module) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FULL_FORMAT:
        raise ValueError(f"not a {FULL_FORMAT} file: {payload.get('format')!r}")
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    return payload


def train_loader(dataset: T13Tiles, batch: int, workers: int, seed: int) -> DataLoader:
    """Shuffled training loader.

    Not persistent: a persistent worker keeps the dataset copy it got at the
    first epoch, so ``set_epoch`` never reached it and every tile kept one flip
    for the whole run (review 2026-11-18, reproduced on CPU).
    """
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=workers,
        drop_last=True,
        persistent_workers=False,
    )


def _load_state(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["state_dict"]


def finetune_limit(limit: int, accumulation: int) -> int:
    """Micro batches a smoke epoch runs under full fine-tune: whole accumulation groups.

    A limit below one group would run no optimizer step: the run would save a
    checkpoint and never allocate the AdamW state it is meant to exercise. ``0``
    (no limit) stays ``0``.
    """
    if not limit:
        return 0
    return math.ceil(limit / accumulation) * accumulation


def require_finetune_steps(optimizer_steps: int, protocol_id: str) -> None:
    """A full fine-tune run that took no optimizer step is not a completed run."""
    if optimizer_steps < 1:
        raise ValueError(
            f"{protocol_id}: full fine-tune took no optimizer step (the loader or the limit "
            "holds fewer micro batches than one accumulation group); refusing to complete"
        )


def run_train_epoch(
    loader,
    micro_loss: Callable[[dict], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    scheduler,
    accumulation: int,
    limit: int,
) -> tuple[dict[str, float], int, int]:
    """One training epoch with gradient accumulation.

    ``micro_loss(batch)`` returns ``(loss, ce, aux)`` for one micro batch; ``loss`` is
    back-propagated. Returns ``(loss, ce and aux summed over micro batches,
    micro batches run, optimizer steps)`` as ``(totals, batches, steps)``. A trailing
    partial accumulation group takes no step.
    """
    totals = {"loss": 0.0, "ce": 0.0, "aux": 0.0}
    batches = 0
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    for batch in loader:
        loss, ce, aux = micro_loss(batch)
        (loss / accumulation).backward()
        totals["loss"] += float(loss.detach())
        totals["ce"] += float(ce.detach())
        totals["aux"] += float(aux.detach())
        batches += 1
        if batches % accumulation == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            steps += 1
        if limit and batches >= limit:
            break
    return totals, batches, steps


def expand_patch_embed(state: dict) -> dict:
    """RGB patch-embedding weights gain a fourth input channel that starts at zero.

    The NIR channel then contributes nothing at step 0, and the RGB path is unchanged.
    Every other key is copied as is.
    """
    key = "patch_embed1.proj.weight"
    weight = state[key]
    widened = dict(state)
    widened[key] = torch.cat([weight, weight.new_zeros(weight.shape[0], 1, *weight.shape[2:])], 1)
    return widened


def widen_to_four_bands(encoder3: nn.Module) -> nn.Module:
    """A four-band MiT-Huge carrying the weights of a loaded three-band one."""
    encoder4 = mit_huge(in_chans=4)
    encoder4.load_state_dict(expand_patch_embed(encoder3.state_dict()), strict=True)
    return encoder4


def train_t13(
    request: WorkerRequest,
    writer: RunWriter,
    emit: Callable[..., None],
    *,
    protocol: ProtocolSpec,
    task_spec: TaskSpec,
    checkpoint: Path,
    model_id: str,
) -> int:
    recipe = PROTOCOLS.get(protocol.protocol_id)
    if recipe is None:
        raise ValueError(f"T13 does not implement protocol {protocol.protocol_id}")
    arm = recipe["arm"]
    run_id, seed = request.run_id, request.seed
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(request.data_root)
    runs_root = Path(request.runs_root)
    manifest = Path(request.manifest_path)

    epochs = int(protocol.epochs or 0)
    lr = float(protocol.learning_rate or 0)
    if not epochs or not lr:
        raise ValueError(
            f"{protocol.protocol_id}: T13 protocols must declare epochs and learning_rate"
        )
    ft = bool(recipe.get("finetune"))
    if ft != (protocol.regime == "full_finetune") or ft != protocol.train_backbone:
        raise ValueError(
            f"{protocol.protocol_id}: recipe fine-tune={ft} disagrees with regime "
            f"{protocol.regime} and train_backbone={protocol.train_backbone}"
        )
    backbone_lr = float(protocol.backbone_learning_rate or 0) if ft else None
    if ft and not backbone_lr:
        raise ValueError(
            f"{protocol.protocol_id}: full fine-tune must declare backbone_learning_rate"
        )
    # Gradient checkpointing per MiT block trades one extra encoder forward for the
    # stored activations. Off unless asked; the tag records the choice.
    grad_checkpoint = ft and _int_env("LIM_WORKER_GRAD_CHECKPOINT", 0) == 1
    # Full fine-tune defaults to micro batch 2 (estimate: ~15 GiB on a 24 GiB card,
    # see docs/design/实验执行方案_v3.4_补充_FT_2026-11-19.md). The effective batch stays 8.
    micro = _int_env("LIM_WORKER_MICRO_BATCH", 2 if ft else EFFECTIVE_BATCH)
    accumulation = _int_env("LIM_WORKER_GRAD_ACCUM", EFFECTIVE_BATCH // max(micro, 1))
    if micro * accumulation != EFFECTIVE_BATCH:
        raise ValueError(
            f"T13 needs effective batch {EFFECTIVE_BATCH}; got {micro} x {accumulation}"
        )
    limit = _int_env("LIM_WORKER_LIMIT_BATCHES", 0)
    eval_sheets = _int_env("LIM_WORKER_EVAL_IMAGES", 0)
    smoke = bool(limit or eval_sheets)
    workers = _int_env("LIM_WORKER_NUM_WORKERS", 6)

    train_rows = load_rows(manifest, "train")
    val_rows = load_rows(manifest, "validation")
    test_rows = load_rows(manifest, "test")
    require_no_group_leakage(manifest, task_spec.task_id)
    if not (train_rows and val_rows and test_rows):
        raise ValueError("T13 manifest needs train, validation and test tiles")
    needs_aef = arm in ("R", "D", "P")
    plugin_arm = arm in ("Bcap", "P")
    if needs_aef or arm == "Bcap":
        missing = sorted(
            {
                r["group_id"]
                for r in train_rows + val_rows + test_rows
                if not (data_root / "derived" / AEF_DIR / f"{r['group_id']}.tif").is_file()
            }
        )
        if needs_aef and missing:
            raise FileNotFoundError(
                f"AEF grid missing for {len(missing)} sheets, e.g. {missing[:3]}"
            )

    torch.manual_seed(seed)
    np.random.seed(seed)
    band_mean, band_std = band_statistics(train_rows, data_root, 256, normalization_seed())

    in_bands = int(recipe.get("bands", 3))
    encoder = mit_huge(in_chans=3)
    load_stats = load_encoder(checkpoint, encoder)
    if load_stats["missing"] or load_stats["unexpected"]:
        raise RuntimeError(f"LIM encoder did not load cleanly: {load_stats}")
    if in_bands == 4:
        encoder = widen_to_four_bands(encoder)
    encoder = encoder.to(device)
    if ft:
        encoder.use_checkpoint = grad_checkpoint
        for p in encoder.parameters():
            p.requires_grad_(True)
    else:
        encoder = encoder.eval()
        for p in encoder.parameters():
            p.requires_grad_(False)
        if device == "cuda":
            encoder = encoder.half()
    inputs = Inputs(encoder, band_mean, band_std, device, finetune=ft, in_bands=in_bands)
    backbone_id = writer.record.model_checkpoint_sha256 or _file_sha256(checkpoint)

    model = ArmModel(arm, CHANNELS, NUM_CLASSES).to(device)
    tags: dict[str, str] = {
        "arm": arm,
        "encoder": "MiT-Huge fine-tuned, fp32 weights, bf16 autocast"
        if ft
        else "MiT-Huge frozen, fp16 live forward (no feature cache)",
        "encoder_keys_loaded": str(load_stats["loaded"]),
        "encoder_normalization": "imagenet RGB (0-255 stats) on bands 1-3",
        "plugin_normalization": (
            f"train-fold z-score, mean {band_mean.round(3).tolist()} "
            f"std {band_std.round(3).tolist()}"
        ),
        "band_order": "R,G,B,NIR",
        "label_mapping": "value // 10; 0 background = ignore, never predicted",
        "metric_classes": "1..7 (mean_iou over the 7 labelled classes)",
        "split": "upstream 80/20 sheets; 8 train sheets held out as validation (seed 17)",
        "augmentation": "h/v flips, shared by image, label and AEF",
        "learning_rate": f"{lr:g}",
        "lr_schedule": protocol.lr_schedule or "constant",
        "epochs_budget": str(epochs),
        "micro_batch": str(micro),
        "grad_accum": str(accumulation),
        "effective_batch": str(EFFECTIVE_BATCH),
        "teacher_product": f"AEF {AEF_YEAR} annual (int8, warped to sheet grid)"
        if needs_aef
        else "none",
        "deploy_inputs": "image + AEF (reference)" if arm == "R" else "image only",
    }

    teacher = None
    if arm == "R":
        tags["aef_mode"] = recipe["aef"]
    if arm == "D":
        path = find_dependency(
            runs_root,
            task_id="T13",
            protocol_id=recipe["teacher"],
            seed=seed,
            model_id=model_id,
            manifest_sha=manifest_sha256(manifest),
            backbone_sha=backbone_id,
            epochs=epochs,
            batch=EFFECTIVE_BATCH,
            smoke=smoke,
        )
        teacher = ArmModel("R", CHANNELS, NUM_CLASSES).to(device)
        teacher.load_state_dict(_load_state(path))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        tags |= {
            "teacher_run": path.parent.name,
            "teacher_checkpoint_sha256": _file_sha256(path),
            "kd_beta": str(recipe["kd_beta"]),
            "kd_tau": str(recipe["kd_tau"]),
        }
    if plugin_arm:
        path = find_dependency(
            runs_root,
            task_id="T13",
            protocol_id=recipe["head"],
            seed=seed,
            model_id=model_id,
            manifest_sha=manifest_sha256(manifest),
            backbone_sha=backbone_id,
            epochs=epochs,
            batch=EFFECTIVE_BATCH,
            smoke=smoke,
        )
        state = _load_state(path)
        model.head.load_state_dict(
            {k[len("head.") :]: v for k, v in state.items() if k.startswith("head.")}
        )
        tags |= {
            "head_run": path.parent.name,
            "head_checkpoint_sha256": _file_sha256(path),
            "lambda_aef": str(recipe["lambda_aef"]),
            "codebook_init": recipe["codebook"],
            "nir_drop_prob": str(model.plugin.nir_drop_prob),
        }
        if recipe["codebook"] == "kmeans":
            vectors = kmeans_vectors(train_rows, data_root, tiles=400, per_tile=512, seed=seed)
            model.plugin.init_codebook_kmeans(
                vectors, seed=seed, region="hubei", aef_year=AEF_YEAR, scope="train fold"
            )
            tags["codebook_train_vectors"] = str(vectors.shape[0])
    trainable = model.configure_trainable(train_head=not plugin_arm)
    backbone_params = list(encoder.parameters()) if ft else []
    count_backbone = sum(p.numel() for p in backbone_params)
    count_head = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
    count_injector = (
        sum(p.numel() for p in model.injectors.parameters() if p.requires_grad)
        if model.injectors is not None
        else 0
    )
    count_trainable = count_backbone + sum(p.numel() for p in trainable)
    tags["head_hash"] = head_hash(model.head, backbone_id)
    tags["regime"] = protocol.regime
    tags["trainable_parameters"] = str(count_trainable)
    tags["trainable_parameters_backbone"] = str(count_backbone)
    tags["trainable_parameters_head"] = str(count_head)
    tags["trainable_parameters_injector"] = str(count_injector)
    if ft:
        tags |= {
            "backbone_learning_rate": f"{backbone_lr:g}",
            "amp": "bf16 autocast (CUDA only; fp32 on CPU), fp32 master weights, no GradScaler",
            "gradient_checkpointing": "on (per MiT block)" if grad_checkpoint else "off",
        }
    writer.update(tags=tags)
    writer.update_training(
        head_id="upernet_gn",
        trainable_parameters=count_trainable,
        total_parameters=sum(p.numel() for p in model.parameters())
        + sum(p.numel() for p in encoder.parameters()),
    )
    writer.update_input(preprocessing_id="t13_imagenet_rgb_plus_trainfold_zscore")
    emit(
        run_id,
        "train.prepare",
        f"T13 {protocol.protocol_id} ({arm}): {len(train_rows)} train / {len(val_rows)} val / "
        f"{len(test_rows)} test tiles, {tags['trainable_parameters']} trainable",
    )

    aef_mode = recipe.get("aef", "real")
    shuffled = arm == "R" and aef_mode == "shuffle"
    if shuffled:
        tags_shuffle = "AEF of the same tile position on the sheet half the list away"
        writer.update(tags={"aef_shuffle": tags_shuffle})

    def forward(
        image_u8: torch.Tensor, codes: torch.Tensor | None, bands: int = 4, *, grad: bool = False
    ) -> dict:
        image = image_u8.to(device, non_blocking=True)
        feats = inputs.features(image, grad=grad)
        aef = aef_input(codes.to(device), aef_mode) if arm == "R" else None
        with inputs.autocast():
            out = model(feats, inputs.plugin_image(image, bands), aef)
        # Loss and metrics run in fp32 even when the head ran in bf16.
        out["logits"] = out["logits"].float()
        return out

    def evaluate(rows: list[dict], bands: int) -> tuple[dict, dict]:
        model.eval()
        encoder.eval()
        with torch.inference_mode():
            return score_sheets(
                rows,
                data_root,
                lambda image, codes: forward(image, codes, bands)["logits"],
                with_aef=arm == "R",
                batch=micro,
                limit=eval_sheets,
                aef_source=donor_rows(rows) if shuffled else None,
            )

    # The default fd-passing strategy ran out of descriptors after a few epochs
    # of smoke runs (each epoch abandons a mid-flight iterator); file_system
    # sharing does not hold one descriptor per batch.
    torch.multiprocessing.set_sharing_strategy("file_system")
    dataset = T13Tiles(
        train_rows,
        data_root,
        with_aef=needs_aef,
        augment=True,
        seed=seed,
        aef_source=donor_rows(train_rows) if shuffled else None,
    )
    loader = train_loader(dataset, micro, workers, seed)
    optimizer = make_optimizer(trainable, backbone_params, head_lr=lr, backbone_lr=backbone_lr)
    # A full fine-tune smoke runs whole accumulation groups, so it always takes a step.
    limit_epoch = finetune_limit(limit, accumulation) if ft else limit
    steps_per_epoch = math.ceil((limit_epoch or len(loader)) / accumulation)
    scheduler = (
        build_scheduler(optimizer, epochs * steps_per_epoch, steps_per_epoch)
        if protocol.lr_schedule == "warmup1-cosine"
        else None
    )
    best = BestCheckpoint()
    # Full fine-tune selects and saves the encoder with the arm model; frozen arms keep
    # the head-only state they always had.
    holder: nn.Module = FullModel(encoder, model) if ft else model
    val_every = validation_every()
    optimizer_steps = 0
    started = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def micro_loss(batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = batch["label"].to(device, non_blocking=True)
        codes = batch.get("aef")
        out = forward(batch["image"], codes if arm == "R" else None, grad=ft)
        ce = safe_cross_entropy(out["logits"], labels, IGNORE_INDEX)
        aux = out["logits"].sum() * 0.0
        if arm == "D":
            with torch.no_grad():
                feats = inputs.features(batch["image"].to(device))
                target = teacher(
                    feats,
                    inputs.plugin_image(batch["image"].to(device)),
                    aef_input(codes.to(device), "real"),
                )["logits"]
            aux = recipe["kd_beta"] * distillation_loss(
                out["logits"], target, labels, IGNORE_INDEX, recipe["kd_tau"]
            )
        elif arm == "P" and recipe["lambda_aef"] > 0:
            teacher_aef, valid = decode_aef(codes.to(device))
            proxies = out["proxy_rgb"] + out["proxy_full"]
            terms = []
            for proxy in proxies:
                target, target_valid = pool_teacher(teacher_aef, valid, tuple(proxy.shape[-2:]))
                terms.append(alignment_loss(proxy, target, target_valid))
            aux = recipe["lambda_aef"] * torch.stack(terms).mean()
        return ce + aux, ce, aux

    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        model.train()
        encoder.train(ft)
        totals, batches, epoch_steps = run_train_epoch(
            loader, micro_loss, optimizer, scheduler, accumulation, limit_epoch
        )
        optimizer_steps += epoch_steps
        for name, value in totals.items():
            writer.add_curve(name, "train", epoch, value / max(batches, 1))
        emit(
            run_id,
            "train.epoch",
            f"epoch {epoch + 1}/{epochs} loss={totals['loss'] / max(batches, 1):.4f} "
            f"ce={totals['ce'] / max(batches, 1):.4f} aux={totals['aux'] / max(batches, 1):.4f}",
            progress=(epoch + 1) / epochs,
        )
        if (epoch + 1) % val_every == 0 or epoch + 1 == epochs:
            val4, _ = evaluate(val_rows, 4)
            selection = val4["mean_iou"]
            writer.add_curve("mean_iou", "validation", epoch, val4["mean_iou"])
            if plugin_arm:
                val3, _ = evaluate(val_rows, 3)
                writer.add_curve("mean_iou_rgb", "validation", epoch, val3["mean_iou"])
                selection = 0.5 * (val4["mean_iou"] + val3["mean_iou"])
            best.offer(selection, epoch + 1, holder)
    train_seconds = time.time() - started
    if ft:
        require_finetune_steps(optimizer_steps, protocol.protocol_id)

    best.restore(holder)
    if ft:
        save_full_checkpoint(best, writer.run_dir / FULL_CHECKPOINT)
        writer.add_artifact(
            "checkpoint",
            FULL_CHECKPOINT,
            "validation-selected backbone, head and injector weights (full fine-tune)",
        )
    else:
        best.save(writer.run_dir / "best_checkpoint.pt")
        writer.add_artifact("checkpoint", "best_checkpoint.pt", "validation-selected weights")
    if plugin_arm:
        model.eval()
        model.save_plugin(
            writer.run_dir / "regional_plugin.pt",
            backbone_id,
            region="hubei",
            protocol=protocol.protocol_id,
            seed=seed,
        )
        writer.add_artifact("checkpoint", "regional_plugin.pt", "deployable regional plugin")

    if recipe.get("test_at_end", True):
        metrics, groups = evaluate(test_rows, 4)
        records = to_metric_records(
            metrics, primary_name="mean_iou", direction="maximize", split="test"
        )
        extra = {}
        if plugin_arm:
            rgb_metrics, _ = evaluate(test_rows, 3)
            extra["mean_iou_rgb_only"] = rgb_metrics["mean_iou"]
            records += to_metric_records(
                {"mean_iou_rgb_only": rgb_metrics["mean_iou"]},
                primary_name="mean_iou_rgb_only",
                split="test",
                primary=False,
            )
        writer.add_metrics(records)
        writer.add_confusion(groups, NUM_CLASSES)
    else:
        writer.update(tags={"test_evaluated": "no", "selection": "validation only"})
    peak = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
    writer.update_training(
        epochs_completed=epochs,
        optimizer_steps=optimizer_steps,
        wall_time_seconds=time.time() - started,
        peak_gpu_memory_mb=peak,
    )
    writer.update(tags={"peak_gpu_memory_mb": f"{peak:.1f}"})
    writer.update(
        tags={
            "selection": "best_validation_mean_iou"
            + ("(avg of 4-band and RGB-only)" if plugin_arm else ""),
            "selected_epoch": str(best.epoch),
            "validation_selection_score": f"{best.score:.4f}",
            "optimizer_steps": str(optimizer_steps),
            "train_seconds": f"{train_seconds:.1f}",
            "test_sheets": str(len({r["group_id"] for r in test_rows})),
        }
    )
    if recipe.get("test_at_end", True):
        summary = (
            f"mean_iou={metrics['mean_iou']:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"({arm}{' full fine-tune' if ft else ''}, {epochs} epochs, best epoch {best.epoch})"
        )
    else:
        summary = (
            f"validation mean_iou={best.score:.4f} "
            f"({arm}{' full fine-tune' if ft else ''}, {epochs} epochs, best epoch {best.epoch}; "
            "test not evaluated)"
        )
    if extra:
        summary += f" rgb_only={extra['mean_iou_rgb_only']:.4f}"
    if limit or eval_sheets:
        summary += " [SMOKE]"
    elif stopped_short_of_budget(task_spec.task_id, epochs_completed=epochs):
        summary += " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(run_id, "train.metrics", summary)
    return 0
