"""T11P (PASTIS-HD six-band) for the Prithvi-EO-2.0 worker.

Six-band matched pair with Clay's T11P track: the input is the PASTIS-HD raw S2
stack subset B02/B03/B04/B8A/B11/B12 at four 2019 dates. Those six bands are
the Prithvi pretraining set (HLS: blue, green, red, NIR-narrow, SWIR-1,
SWIR-2), so the encoder's native patch embedding is used and no band adapter is
needed.

Readout protocol (same as the Clay T11P track):

* frozen encoder, four dates encoded independently and averaged in feature
  space, then a 1x1-convolution head (``raw`` control, width 1024);
* ``s128``: the same encoder plus a trainable E(1024->128) trained jointly with
  the head on labels;
* ``o128``: a frozen E distilled from OlmoEarth v1.3 128-dim embeddings;
* B stage: last four transformer blocks + final norm + E + head, with and
  without the teacher cosine. This is the only difference between the pair.

Caveats written into every run: PASTIS S2 is not HLS and the band centres differ
slightly; the manifest has no coordinates, so the position metadata is zero
filled; the 128 px patch is resized to 224 px for the pretrained positional
grid (a readout choice, not a new observation); teacher vectors are pooled from
the 128 px grid to Prithvi's 14x14 tokens with adaptive averaging.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.data.datasets.pastis_hd import select_2019_frames
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.groups import GroupedSegmentation
from rs_backbone_eval.evaluation.segmentation import multiclass_metrics
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.worker_runtime import RunWriter

TASK_ID = "T11P"
BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
S2_INDICES = [0, 1, 2, 7, 8, 9]
MAX_FRAMES = 4
CHIP = 224
LABEL_SIZE = 128
GRID = CHIP // 16
EMBED_DIM = 1024
NUM_CLASSES = 20
IGNORE_INDEX = 19
VOID_CLASS = 19
GSD_M = 10.0
OFFICIAL_EPOCHS = 8
FEATURE_CACHE_VERSION = "2026-11-12.t11p-prithvi-tokens"
B_STAGE_BLOCKS = 4
B_STAGE_DIMS = 128
B_STAGE_TEACHER_WEIGHT = 1.0

PROBE_PROTOCOLS = {
    "prithvi_t11p_probe_raw": ("raw", None),
    "prithvi_t11p_probe_s128": ("supervised", 128),
    "prithvi_t11p_probe_o128": ("distilled", "oe128"),
    "prithvi_t11p_probe_pca128": ("pca128", 128),
    "prithvi_t11p_probe_o128_mse": ("distilled", "oe128mse"),
    "prithvi_t11p_probe_o128_nce": ("distilled", "oe128nce"),
}
DISTILL_PROTOCOLS = {
    "prithvi_t11p_distill_o128": ("oe128", "olmoearth-2019-128", 128),
    "prithvi_t11p_distill_o128_mse": ("oe128mse", "olmoearth-2019-128", 128),
    "prithvi_t11p_distill_o128_nce": ("oe128nce", "olmoearth-2019-128", 128),
}
DISTILL_LOSS = {
    "prithvi_t11p_distill_o128_mse": "mse",
    "prithvi_t11p_distill_o128_nce": "infonce",
}
B_STAGE_PROTOCOLS = {
    "prithvi_t11p_ft_last4": None,
    "prithvi_t11p_ft_last4_distill": "oe128",
    "prithvi_t11p_ft_last4_raw1024": None,
    "prithvi_t11p_ft_last4_raw1024_distill": "oe128",
}
B_STAGE_VARIANTS = {
    "prithvi_t11p_ft_last4": (B_STAGE_BLOCKS, B_STAGE_DIMS, True),
    "prithvi_t11p_ft_last4_distill": (B_STAGE_BLOCKS, B_STAGE_DIMS, True),
    "prithvi_t11p_ft_last4_raw1024": (B_STAGE_BLOCKS, EMBED_DIM, False),
    "prithvi_t11p_ft_last4_raw1024_distill": (B_STAGE_BLOCKS, EMBED_DIM, False),
}
# Wide-head arms use a separate training function so the admitted 128-d pair
# keeps its original RNG/accumulation path.
WIDE_B_STAGE_PROTOCOLS = {
    "prithvi_t11p_ft_last4_raw1024": None,
    "prithvi_t11p_ft_last4_raw1024_distill": "oe128",
}


class T11PFailure(Exception):
    """A gate turns into a failed run with a readable reason."""


def _minimum_train_rows() -> int:
    """1200 full-budget rows, or any size in a declared label-fraction panel."""
    return 1 if os.environ.get("RS_EVAL_LABEL_FRACTION") else 1200


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _rows(manifest: Path, split: str | None = None) -> list[dict]:
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if split is not None:
        rows = [row for row in rows if row["split"] == split]
    return rows


def _groups(rows: list[dict]) -> dict[str, list[dict]]:
    return {
        split: sorted((row for row in rows if row["split"] == split), key=lambda r: r["sample_id"])
        for split in ("train", "validation", "test")
    }


def _labels(row: dict, data_root: Path) -> np.ndarray:
    labels = np.load(data_root / row["label_path"])[0].astype(np.int64)
    return np.where(labels == VOID_CLASS, IGNORE_INDEX, labels)


def _frames(row: dict, data_root: Path) -> np.ndarray:
    """(T,6,128,128) raw DN for the four 2019 frames declared by the manifest."""
    aux = row.get("aux_paths") or {}
    stack_path = aux.get("s2_path")
    if not stack_path:
        raise T11PFailure(f"sample {row['sample_id']}: aux_paths has no s2_path")
    sidecar = data_root / aux["dates_source"]
    dates = [int(date) for date in json.loads(sidecar.read_text())["dates_s2"]]
    frames = select_2019_frames(dates, MAX_FRAMES)
    stack = np.load(data_root / stack_path, mmap_mode="r")
    if not 0 <= max(S2_INDICES) < stack.shape[1]:
        raise T11PFailure(f"sample {row['sample_id']}: S2 stack has {stack.shape[1]} bands")
    return np.asarray(stack[frames][:, S2_INDICES], dtype=np.float32)


class PastisS2(Dataset):
    """One PASTIS patch as four normalised six-band frames plus a 10 m label."""

    def __init__(self, data_root: Path, rows: list[dict]) -> None:
        self.data_root = data_root
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "image": torch.from_numpy(_frames(row, self.data_root)),
            "label": torch.from_numpy(_labels(row, self.data_root)),
            "group": row["group_id"],
        }


def channel_statistics(rows: list[dict], data_root: Path, samples: int = 40) -> tuple[list, list]:
    """Per-band mean/std of the raw DN over a fixed, seeded train sample."""
    rng = np.random.default_rng(17)
    chosen = rows if len(rows) <= samples else [rows[i] for i in rng.choice(len(rows), samples)]
    total, squares, pixels = 0.0, 0.0, 0
    for row in chosen:
        frames = _frames(row, data_root)
        total += frames.sum(axis=(0, 2, 3), dtype=np.float64)
        squares += np.square(frames, dtype=np.float64).sum(axis=(0, 2, 3))
        pixels += frames.shape[0] * frames.shape[2] * frames.shape[3]
    mean = total / pixels
    std = np.sqrt(np.maximum(squares / pixels - mean**2, 1e-6))
    return [float(value) for value in mean], [float(value) for value in std]


def build_encoder(checkpoint: Path):
    from terratorch.registry import BACKBONE_REGISTRY

    encoder = BACKBONE_REGISTRY.build(
        "prithvi_eo_v2_300_tl",
        pretrained=not random_init_requested(),
        ckpt_path=None if random_init_requested() else str(checkpoint),
        bands=BANDS,
    )
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder


def _unfreeze_last_blocks(encoder) -> int:
    blocks = getattr(encoder, "blocks", None)
    if blocks is None:
        raise T11PFailure("Prithvi encoder does not expose .blocks; cannot unfreeze the last four")
    for block in blocks[-B_STAGE_BLOCKS:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    if hasattr(encoder, "norm"):
        for parameter in encoder.norm.parameters():
            parameter.requires_grad = True
    return sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad)


def build_projector(dims: int = B_STAGE_DIMS, hidden: int = 256) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(EMBED_DIM, hidden, 1),
        nn.GELU(),
        nn.Conv2d(hidden, dims, 1),
    )


def build_head(in_channels: int, hidden: int = 128) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 1),
        nn.GroupNorm(8, hidden),
        nn.GELU(),
        nn.Conv2d(hidden, NUM_CLASSES, 1),
    )


class PrithviT11P(nn.Module):
    def __init__(self, encoder, *, train_encoder: bool, projector: nn.Module | None, dims: int):
        super().__init__()
        self.encoder = encoder
        self.train_encoder = train_encoder
        if not train_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        self.projector = projector
        self.head = build_head(dims)

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_encoder:
            self.encoder.eval()
        return self

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """(B,T,6,H,W) normalised -> (B,1024,14,14) temporal-mean tokens."""
        import torch.nn.functional as F

        batch, time, channels, height, width = images.shape
        flat = images.reshape(batch * time, channels, height, width)
        flat = F.interpolate(flat, size=(CHIP, CHIP), mode="bilinear", align_corners=False)
        context = torch.enable_grad() if self.train_encoder else torch.no_grad()
        with context:
            layers = self.encoder(flat)
        tokens = layers[-1][:, 1:, :] if isinstance(layers, (list, tuple)) else layers[:, 1:, :]
        _, length, dim = tokens.shape
        grid = int(round(length**0.5))
        features = tokens.transpose(1, 2).reshape(batch * time, dim, grid, grid)
        return features.reshape(batch, time, dim, grid, grid).mean(dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encode(images)
        if self.projector is not None:
            features = self.projector(features)
        logits = self.head(features)
        return torch.nn.functional.interpolate(
            logits, size=(LABEL_SIZE, LABEL_SIZE), mode="bilinear", align_corners=False
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_root() -> Path:
    base = os.environ.get("RS_EVAL_CACHE_ROOT")
    return Path(base) if base else Path.home() / "rs-eval-cache"


def _fingerprint(
    checkpoint_sha256: str,
    manifest_sha256: str,
    target: str,
    seed: int,
    *,
    init: str,
    normalization: str,
) -> str:
    payload = {
        "version": "2026-11-12.t11p-prithvi-projector",
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "task": TASK_ID,
        "bands": BANDS,
        "frames": MAX_FRAMES,
        "grid": GRID,
        "target": target,
        "seed": seed,
        # The projector is seed-specific by path, but include the condition
        # explicitly so a random-init projector can never be reused as a
        # pretrained one (review 2026-11-12, P1).
        "init": init,
        "normalization": normalization,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _feature_fingerprint(
    checkpoint_sha256: str,
    manifest_sha256: str,
    *,
    init: str,
    normalization: str,
    seed: int,
) -> str:
    payload = {
        "version": FEATURE_CACHE_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "task": TASK_ID,
        "bands": BANDS,
        "frames": MAX_FRAMES,
        "grid": GRID,
        "init": init,
        "normalization": normalization,
    }
    # Pretrained weights are seed-independent, so their token cache is shared
    # across seeds. Random init is not: every seed has different weights and
    # must get its own cache (review 2026-11-12, P1).
    if init != "pretrained":
        payload["seed"] = seed
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _feature_dir(fingerprint: str) -> Path:
    return _cache_root() / "pastis-t11p" / "prithvi" / "features" / fingerprint


def _projector_path(target: str, seed: int, fingerprint: str) -> Path:
    return (
        _cache_root()
        / "pastis-t11p"
        / "prithvi"
        / "projector"
        / target
        / f"seed{seed}"
        / (f"{fingerprint}.pt")
    )


def _save_atomic(path: Path, payload: dict) -> None:
    import io

    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(path)


def decode_teacher(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    features = np.asarray(raw, dtype=np.float32)
    features = np.sign(features) * (features / 127.5) ** 2
    features[:, ~valid] = 0
    return features


def teacher_tokens(
    target: str, row: dict, data_root: Path, tree_dims: dict
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher pooled from the 128 px grid to Prithvi's 14x14 token grid."""
    import torch.nn.functional as F

    tree, dims = tree_dims[target]
    path = data_root / "derived" / tree / "features" / "2019" / f"{row['sample_id']}.npz"
    if not path.is_file():
        raise T11PFailure(f"sample {row['sample_id']}: teacher payload {path} missing")
    with np.load(path) as archive:
        raw = archive["features"]
        valid = archive["valid"] if "valid" in archive.files else np.ones(raw.shape[1:], bool)
    tensor = torch.from_numpy(decode_teacher(raw, valid))
    pooled = F.adaptive_avg_pool2d(tensor.unsqueeze(0), GRID).squeeze(0)
    support = (
        F.adaptive_avg_pool2d(
            torch.from_numpy(valid.astype(np.float32))[None, None], GRID
        ).squeeze()
        > 0
    )
    return pooled, support


def _cosine(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F

    unit_prediction = F.normalize(prediction, dim=1)
    unit_target = F.normalize(target, dim=1)
    cosine = (unit_prediction * unit_target).sum(dim=1)
    if bool(mask.any()):
        return cosine[mask].mean()
    return cosine.mean()


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over observed tokens, with no length normalisation."""
    error = ((prediction - target) ** 2).mean(dim=1)
    if bool(mask.any()):
        return error[mask.bool()].mean()
    return error.mean()


def _loss_improved(score: float, best: float, loss_kind: str) -> bool:
    """Epoch selection: MSE minimises, cosine maximises."""
    return score < best if loss_kind == "mse" else score > best


def _mean_teacher_baseline(
    train_rows: list[dict],
    rows_split: list[dict],
    target: str,
    data_root: Path,
    tree_dims: dict,
    device: str,
    dims: int,
) -> tuple[float, float]:
    """Cosine/MSE of predicting the train-set mean teacher vector."""
    total = torch.zeros(dims, device=device)
    count = 0
    for row in train_rows:
        grid, support = teacher_tokens(target, row, data_root, tree_dims)
        values = torch.as_tensor(grid, device=device)
        mask = torch.as_tensor(support, device=device).bool()
        total += values[:, mask].sum(dim=1)
        count += int(mask.sum())
    mean_vector = total / max(count, 1)
    cosine_sum, mse_sum, tokens = 0.0, 0.0, 0
    for start in range(0, len(rows_split), 2):
        targets, supports = [], []
        for row in rows_split[start : start + 2]:
            grid, support = teacher_tokens(target, row, data_root, tree_dims)
            targets.append(grid)
            supports.append(support)
        target_tensor = torch.stack(targets).to(device)
        support_tensor = torch.stack(supports).to(device)
        prediction = mean_vector.view(1, dims, 1, 1).expand_as(target_tensor)
        cosine = (
            torch.nn.functional.normalize(prediction, dim=1)
            * torch.nn.functional.normalize(target_tensor, dim=1)
        ).sum(dim=1)
        error = ((prediction - target_tensor) ** 2).mean(dim=1)
        mask = support_tensor.bool()
        cosine_sum += float(cosine[mask].sum())
        mse_sum += float(error[mask].sum())
        tokens += int(mask.sum())
    return cosine_sum / max(tokens, 1), mse_sum / max(tokens, 1)


def _metric(name: str, value: float, split: str, direction: str = "maximize"):
    from rs_backbone_eval.contracts import MetricRecord

    return MetricRecord(
        name=name,
        direction=direction,
        value=round(float(value), 6),
        split=split,
        is_primary=False,
    )


def normalize_images(images: torch.Tensor, means: torch.Tensor, stds: torch.Tensor) -> torch.Tensor:
    """Apply the PASTIS train-split per-band statistics to a (B,T,C,H,W) batch."""
    return (images - means) / stds


def _evaluate(
    model: PrithviT11P,
    loader: DataLoader,
    means: torch.Tensor,
    stds: torch.Tensor,
    device: str,
) -> tuple[dict, dict[str, list[list[int]]]]:
    collector = GroupedSegmentation(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = normalize_images(batch["image"].to(device), means, stds)
            logits = model(images)
            collector.update(
                list(batch["group"]), batch["label"].numpy(), logits.argmax(1).cpu().numpy()
            )
    return multiclass_metrics(collector.global_matrix), collector.group_confusion()


def _extract_split(
    model: PrithviT11P,
    loader: DataLoader,
    means: torch.Tensor,
    stds: torch.Tensor,
    device: str,
    *,
    cache_dir: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Frozen temporal-mean tokens, labels and groups for one split.

    When ``cache_dir`` is set, every row is cached by the frozen-feature
    fingerprint. The raw / S128 / O128 probes and the projector distillation
    share that cache, so only the first run pays the encoder forward cost.
    """
    rows = list(loader.dataset.rows)
    if cache_dir is not None:
        cached = _load_feature_cache(cache_dir, rows)
        if cached is not None:
            labels = torch.stack(
                [torch.from_numpy(_labels(row, loader.dataset.data_root)) for row in rows]
            )
            return cached, labels, [row["group_id"] for row in rows]
    model.eval()
    features, labels, groups = [], [], []
    with torch.inference_mode():
        for batch in loader:
            images = (batch["image"].to(device) - means) / stds
            batch_features = model.encode(images).to(torch.float16).cpu()
            features.append(batch_features)
            labels.append(batch["label"])
            groups.extend(batch["group"])
            if cache_dir is not None:
                _save_feature_cache(cache_dir, batch["group"], batch_features)
    return torch.cat(features), torch.cat(labels), groups


def _load_feature_cache(cache_dir: Path, rows: list[dict]) -> torch.Tensor | None:
    arrays = []
    for row in rows:
        path = cache_dir / f"{row['sample_id']}.npy"
        meta_path = cache_dir / f"{row['sample_id']}.json"
        if not path.is_file() or not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if meta.get("version") != FEATURE_CACHE_VERSION:
            return None
        if meta.get("fingerprint") != cache_dir.name:
            return None
        array = np.load(path, mmap_mode="r")
        if tuple(array.shape) != (EMBED_DIM, GRID, GRID) or array.dtype != np.float16:
            return None
        arrays.append(np.asarray(array))
    return torch.from_numpy(np.stack(arrays))


def _save_feature_cache(cache_dir: Path, groups: list[str], batch: torch.Tensor) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, group in enumerate(groups):
        array = batch[index].numpy()
        path = cache_dir / f"{group}.npy"
        temporary = path.with_suffix(".npy.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, array)
        temporary.replace(path)
        (cache_dir / f"{group}.json").write_text(
            json.dumps(
                {
                    "version": FEATURE_CACHE_VERSION,
                    "fingerprint": cache_dir.name,
                    "shape": list(array.shape),
                    "dtype": "float16",
                }
            ),
            encoding="utf-8",
        )


def _evaluate_features(
    head: nn.Module,
    projector: nn.Module | None,
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: list[str],
    device: str,
    chunk: int = 16,
) -> tuple[dict, dict[str, list[list[int]]]]:
    collector = GroupedSegmentation(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    head.eval()
    with torch.inference_mode():
        for start in range(0, features.shape[0], chunk):
            tokens = features[start : start + chunk].to(device).float()
            if projector is not None:
                tokens = projector(tokens)
            logits = head(tokens)
            logits = torch.nn.functional.interpolate(
                logits, size=(LABEL_SIZE, LABEL_SIZE), mode="bilinear", align_corners=False
            )
            collector.update(
                groups[start : start + chunk],
                labels[start : start + chunk].numpy(),
                logits.argmax(1).cpu().numpy(),
            )
    return multiclass_metrics(collector.global_matrix), collector.group_confusion()


def _head_train(
    head: nn.Module,
    projector: nn.Module | None,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    device: str,
    seed: int,
    limit: int | None,
    emit,
    run_id: str,
    writer: RunWriter,
    label: str,
) -> None:
    parameters = list(head.parameters())
    if projector is not None and any(p.requires_grad for p in projector.parameters()):
        parameters += list(projector.parameters())
    optimiser = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    for epoch in range(epochs):
        head.train()
        if projector is not None and any(p.requires_grad for p in projector.parameters()):
            projector.train()
        order = torch.randperm(
            train_features.shape[0],
            generator=torch.Generator().manual_seed(1_000 * seed + epoch),
        ).tolist()
        total, steps = 0.0, 0
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            tokens = train_features[chosen].to(device).float()
            if projector is not None:
                tokens = projector(tokens)
            logits = head(tokens)
            logits = torch.nn.functional.interpolate(
                logits, size=(LABEL_SIZE, LABEL_SIZE), mode="bilinear", align_corners=False
            )
            labels = train_labels[chosen].to(device)
            if int((labels != IGNORE_INDEX).sum()) == 0:
                continue
            loss = criterion(logits, labels)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            steps += 1
            if limit and steps >= limit:
                break
        writer.add_curve("loss", "train", epoch, total / max(steps, 1))
        emit(
            run_id,
            f"{label}.epoch",
            f"epoch {epoch + 1}/{epochs} loss={total / max(steps, 1):.4f}",
            progress=(epoch + 1) / epochs,
        )


def train_t11p(
    request: WorkerRequest,
    writer: RunWriter,
    emit,
    *,
    checkpoint: Path,
    protocol_id: str,
    task_spec,
) -> int:
    """Dispatcher for the probe, projector-distillation and B-stage protocols."""
    device = os.environ.get("PRITHVI_WORKER_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    manifest = Path(request.manifest_path)
    data_root = Path(request.data_root)
    from rs_backbone_eval.data.manifest import audit_task_manifest

    audit = audit_task_manifest(manifest, task_spec)
    strict_errors = audit.error_count - len(audit.leakage)
    if strict_errors:
        raise T11PFailure(f"manifest audit failed: {audit.all_errors()[:5]}")
    rows = _rows(manifest)
    groups = _groups(rows)
    train_rows = groups["train"]
    eval_limit = _int_env("PRITHVI_WORKER_EVAL_LIMIT", 0)
    if eval_limit:
        groups["validation"] = groups["validation"][:eval_limit]
        groups["test"] = groups["test"][:eval_limit]
    if len(train_rows) < _minimum_train_rows():
        raise T11PFailure(
            f"train fold has {len(train_rows)} rows; T11P formal runs need the full "
            "1,455-row fold unless RS_EVAL_LABEL_FRACTION declares a few-shot panel"
        )
    train_limit = _int_env("PRITHVI_WORKER_TRAIN_LIMIT", 0)
    if train_limit:
        groups["train"] = groups["train"][:train_limit]
        train_rows = groups["train"]
    from rs_backbone_eval.repro import seed_everything

    seed_everything(request.seed)
    mean, std = channel_statistics(train_rows, data_root)
    means = torch.tensor(mean, device=device).view(1, 1, -1, 1, 1)
    stds = torch.tensor(std, device=device).view(1, 1, -1, 1, 1)
    epochs = _int_env("PRITHVI_WORKER_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("PRITHVI_WORKER_BATCH_SIZE", 1)
    accumulation = _int_env("PRITHVI_WORKER_GRAD_ACCUM", 2)
    lr = _float_env("PRITHVI_WORKER_LR", 3e-3)
    backbone_lr = _float_env("PRITHVI_WORKER_BACKBONE_LR", 1e-4)
    workers = _int_env("PRITHVI_WORKER_NUM_WORKERS", 4)
    limit = _int_env("PRITHVI_WORKER_LIMIT_BATCHES", 0) or None
    partial_eval = os.environ.get("PRITHVI_WORKER_ALLOW_PARTIAL_EVAL") == "1"
    smoke = bool(limit or eval_limit or train_limit or partial_eval)
    from rs_backbone_eval.matrix import (
        OFFICIAL_TEST_SAMPLES,
        OFFICIAL_VALIDATION_SAMPLES,
    )

    if not partial_eval:
        for split, expected in (
            ("validation", OFFICIAL_VALIDATION_SAMPLES[TASK_ID]),
            ("test", OFFICIAL_TEST_SAMPLES[TASK_ID]),
        ):
            if len(groups[split]) != expected:
                raise T11PFailure(
                    f"{split} split has {len(groups[split])} rows; T11P formal runs "
                    f"need {expected} (set PRITHVI_WORKER_ALLOW_PARTIAL_EVAL=1 only for smoke)"
                )
    init = "random" if random_init_requested() else "pretrained"
    normalization = hashlib.sha256(
        json.dumps([mean, std], separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]

    def loader(rows_split: list[dict], shuffle: bool) -> DataLoader:
        return DataLoader(
            PastisS2(data_root, rows_split),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
        )

    # Extraction must be order-stable: the head reshuffles features itself, so
    # a shuffled DataLoader here would make the cold-cache and warm-cache
    # training orders differ (review 2026-11-12, P2).
    train_loader = loader(train_rows, False)
    val_loader = loader(groups["validation"], False)
    test_loader = loader(groups["test"], False)
    encoder = build_encoder(checkpoint).to(device)
    writer.update_input(
        preprocessing_id="pastis_s2_sixband_train_stats",
        bands=list(BANDS),
        time_steps=MAX_FRAMES,
        gsd_m=GSD_M,
    )
    common_tags = {
        "student_input": "S2 B02/B03/B04/B8A/B11/B12 raw DN, 4 frames from 2019",
        "band_semantics": "B8A~HLS NIR_NARROW, B11~SWIR_1, B12~SWIR_2",
        "normalization": "PASTIS train-split per-band mean/std (raw DN)",
        "normalization_mean": ",".join(f"{value:.2f}" for value in mean),
        "normalization_std": ",".join(f"{value:.2f}" for value in std),
        "resize": f"{LABEL_SIZE}px -> {CHIP}px for the pretrained positional grid",
        "token_grid": f"{GRID}x{GRID}",
        "coords": "zero-filled (PASTIS manifest has no coordinates)",
        "teacher_used_at_inference": "no",
        "label_mapping": "0..19 kept; void 19 ignored",
        "effective_batch_patches": str(batch_size * accumulation),
        "device": device,
        "host": (
            torch.cuda.get_device_name(0).lower().replace(" ", "-") if device == "cuda" else "cpu"
        ),
        "torch": torch.__version__,
        "encoder_parameters": str(sum(p.numel() for p in encoder.parameters())),
        "train_samples": str(len(train_rows)),
        "validation_samples": str(len(groups["validation"])),
        "test_samples": str(len(groups["test"])),
        "init": init,
        "normalization_digest": normalization,
        "test_expected_samples": str(OFFICIAL_TEST_SAMPLES[TASK_ID]),
        "validation_expected_samples": str(OFFICIAL_VALIDATION_SAMPLES[TASK_ID]),
    }
    if smoke:
        # Any limit is a diagnostic run even when it still completes 8 epochs.
        common_tags["smoke"] = "true"
        common_tags["panel"] = "diagnostic"
    writer.update(tags=common_tags)

    if protocol_id in DISTILL_PROTOCOLS:
        return _train_distill(
            request,
            writer,
            emit,
            protocol_id,
            rows,
            groups,
            data_root,
            device,
            means,
            stds,
            encoder,
            checkpoint,
            epochs,
            lr,
            workers,
            limit,
            common_tags,
            loader,
        )
    if protocol_id in PROBE_PROTOCOLS:
        return _train_probe(
            request,
            writer,
            emit,
            protocol_id,
            data_root,
            device,
            means,
            stds,
            checkpoint,
            epochs,
            lr,
            batch_size,
            accumulation,
            limit,
            common_tags,
            train_loader,
            val_loader,
            test_loader,
            encoder,
        )
    if protocol_id in WIDE_B_STAGE_PROTOCOLS:
        return _train_wide_b_stage(
            request,
            writer,
            emit,
            protocol_id,
            rows,
            groups,
            data_root,
            device,
            means,
            stds,
            checkpoint,
            epochs,
            lr,
            backbone_lr,
            batch_size,
            accumulation,
            workers,
            limit,
            common_tags,
            encoder,
        )
    if protocol_id in B_STAGE_PROTOCOLS:
        return _train_b_stage(
            request,
            writer,
            emit,
            protocol_id,
            rows,
            groups,
            data_root,
            device,
            means,
            stds,
            checkpoint,
            epochs,
            lr,
            backbone_lr,
            batch_size,
            accumulation,
            workers,
            limit,
            common_tags,
            encoder,
        )
    raise T11PFailure(f"Prithvi T11P has no protocol {protocol_id!r}")


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _train_probe(
    request,
    writer,
    emit,
    protocol_id,
    data_root,
    device,
    means,
    stds,
    checkpoint,
    epochs,
    lr,
    batch_size,
    accumulation,
    limit,
    tags,
    train_loader,
    val_loader,
    test_loader,
    encoder,
) -> int:
    kind, target = PROBE_PROTOCOLS[protocol_id]
    projector = None
    dims = EMBED_DIM
    if kind == "pca128":
        dims = 128
    projector_meta = None
    if kind in {"supervised", "distilled"}:
        dims = target if isinstance(target, int) else 128
        projector = build_projector(dims).to(device)
        if kind == "distilled":
            fingerprint = _fingerprint(
                writer.record.model_checkpoint_sha256 or "",
                writer.record.dataset.manifest_sha256,
                target,
                request.seed,
                init=tags["init"],
                normalization=tags["normalization_digest"],
            )
            path = _projector_path(target, request.seed, fingerprint)
            if not path.is_file():
                raise T11PFailure(
                    f"no distilled projector for {target} seed {request.seed} at {path}; "
                    f"run {DISTILL_PROTOCOLS_BY_TARGET[target]} first"
                )
            payload = torch.load(path, map_location="cpu")
            projector.load_state_dict(payload["state_dict"])
            for parameter in projector.parameters():
                parameter.requires_grad = False
            projector.eval()
            projector_meta = {"path": str(path), "sha256": _sha256(path)}
            writer.update(
                tags={
                    "transfer_projector": str(path),
                    "transfer_projector_sha256": projector_meta["sha256"],
                    "transfer_projector_source_run": str(payload.get("source_run_id", "")),
                }
            )
    model = PrithviT11P(encoder, train_encoder=False, projector=projector, dims=dims).to(device)
    parameters = list(model.head.parameters())
    if projector is not None and any(p.requires_grad for p in projector.parameters()):
        parameters += list(projector.parameters())
    writer.update_training(
        head_id=f"prithvi_t11p_probe_{kind}_{dims}",
        trainable_parameters=sum(p.numel() for p in parameters),
        total_parameters=sum(p.numel() for p in model.parameters()),
    )
    writer.update(tags={**tags, "probe_readout": kind, "probe_dims": str(dims)})
    emit(
        request.run_id,
        "probe.prepare",
        f"T11P {kind} readout {dims}, {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val / {len(test_loader.dataset)} test",
    )
    started = time.time()
    feature_fingerprint = _feature_fingerprint(
        writer.record.model_checkpoint_sha256 or "",
        writer.record.dataset.manifest_sha256,
        init=tags["init"],
        normalization=tags["normalization_digest"],
        seed=request.seed,
    )
    feature_dir = _feature_dir(feature_fingerprint)
    train_features, train_labels, _ = _extract_split(
        model, train_loader, means, stds, device, cache_dir=feature_dir
    )
    val_features, val_labels, val_groups = _extract_split(
        model, val_loader, means, stds, device, cache_dir=feature_dir
    )
    test_features, test_labels, test_groups = _extract_split(
        model, test_loader, means, stds, device, cache_dir=feature_dir
    )
    writer.update(
        tags={"feature_fingerprint": feature_fingerprint, "feature_cache": str(feature_dir)}
    )
    if kind == "pca128":
        from rs_backbone_eval.pca import fit_token_pca, project_token_pca

        mean_pca, components_pca, explained = fit_token_pca(train_features.numpy(), components=dims)
        train_features = torch.from_numpy(
            project_token_pca(train_features.numpy(), mean_pca, components_pca)
        )
        val_features = torch.from_numpy(
            project_token_pca(val_features.numpy(), mean_pca, components_pca)
        )
        test_features = torch.from_numpy(
            project_token_pca(test_features.numpy(), mean_pca, components_pca)
        )
        run_dir = Path(request.runs_root) / request.run_id
        np.save(run_dir / "pca_mean.npy", mean_pca)
        np.save(run_dir / "pca_components.npy", components_pca)
        writer.add_artifact("table", "pca_mean.npy", "train-only PCA mean")
        writer.add_artifact("table", "pca_components.npy", "train-only PCA basis")
        writer.update(
            tags={
                "pca_components": str(dims),
                "pca_explained_variance": f"{explained:.4f}",
            }
        )
    _head_train(
        model.head,
        projector,
        train_features,
        train_labels,
        epochs=epochs,
        lr=lr,
        batch_size=max(batch_size * accumulation, 1),
        device=device,
        seed=request.seed,
        limit=limit,
        emit=emit,
        run_id=request.run_id,
        writer=writer,
        label="probe",
    )
    validation, _ = _evaluate_features(
        model.head, projector, val_features, val_labels, val_groups, device
    )
    writer.add_curve("mean_iou", "validation", epochs, validation["mean_iou"])
    test, group_json = _evaluate_features(
        model.head, projector, test_features, test_labels, test_groups, device
    )
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
    writer.add_confusion(group_json, NUM_CLASSES)
    writer.update(
        tags={
            "validation_samples_evaluated": str(val_features.shape[0]),
            "validation_manifest_samples": str(val_features.shape[0]),
            "validation_expected_samples": tags["validation_expected_samples"],
            "test_samples_evaluated": str(test_features.shape[0]),
            "test_manifest_samples": str(test_features.shape[0]),
            "test_expected_samples": tags["test_expected_samples"],
        }
    )
    wall = time.time() - started
    writer.update_training(epochs_completed=epochs, wall_time_seconds=round(wall, 1))
    summary = (
        f"T11P Prithvi {kind}/{dims}: test mIoU {test['mean_iou']:.4f}, "
        f"{epochs} epochs, frozen encoder"
    )
    if tags.get("smoke") == "true":
        summary += " [SMOKE]"
    elif epochs != OFFICIAL_EPOCHS:
        summary += " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(request.run_id, "probe.metrics", summary)
    return 0


DISTILL_PROTOCOLS_BY_TARGET = {
    target: protocol for protocol, (target, _, _) in DISTILL_PROTOCOLS.items()
}


def _train_distill(
    request,
    writer,
    emit,
    protocol_id,
    rows,
    groups,
    data_root,
    device,
    means,
    stds,
    encoder,
    checkpoint,
    epochs,
    lr,
    workers,
    limit,
    tags,
    loader,
) -> int:
    from rs_backbone_eval.nce import (
        INFONCE_TEMPERATURE,
        INFONCE_TOKENS,
        info_nce,
        patches_per_step,
        retrieval_top1,
        sample_token_pairs,
    )

    target, tree, dims = DISTILL_PROTOCOLS[protocol_id]
    loss_kind = DISTILL_LOSS.get(protocol_id, "cosine")
    tree_dims = {target: (tree, dims)}
    model = PrithviT11P(encoder, train_encoder=False, projector=None, dims=EMBED_DIM).to(device)
    projector = build_projector(dims).to(device)
    writer.update_training(
        head_id=f"prithvi_t11p_projector_{dims}",
        trainable_parameters=sum(p.numel() for p in projector.parameters()),
        total_parameters=sum(p.numel() for p in projector.parameters()),
    )
    writer.update(
        tags={
            **tags,
            "distill_target": target,
            "distill_teacher_tree": tree,
            "distill_dims": str(dims),
            "distill_loss": (
                "symmetric InfoNCE over sampled observed tokens"
                if loss_kind == "infonce"
                else "mean((E(time-mean F) - teacher)^2) over observed tokens, no L2 normalisation"
                if loss_kind == "mse"
                else "mean(1 - cos(E(time-mean F), teacher)) over observed tokens"
            ),
            "distill_selection": (
                "val_infonce"
                if loss_kind == "infonce"
                else "val_mse"
                if loss_kind == "mse"
                else "val_cosine"
            ),
            **(
                {
                    "infonce_temperature": str(INFONCE_TEMPERATURE),
                    "infonce_tokens_per_step": str(INFONCE_TOKENS),
                    "infonce_patches_per_step": str(patches_per_step("infonce")),
                    "false_negative_handling": "none",
                }
                if loss_kind == "infonce"
                else {}
            ),
            "distill_aggregation": "adaptive average 128px teacher -> 14x14 tokens",
        }
    )
    feature_fingerprint = _feature_fingerprint(
        writer.record.model_checkpoint_sha256 or "",
        writer.record.dataset.manifest_sha256,
        init=tags["init"],
        normalization=tags["normalization_digest"],
        seed=request.seed,
    )
    feature_dir = _feature_dir(feature_fingerprint)
    train_features, _, _ = _extract_split(
        model, loader(groups["train"], False), means, stds, device, cache_dir=feature_dir
    )
    val_features, _, _ = _extract_split(
        model, loader(groups["validation"], False), means, stds, device, cache_dir=feature_dir
    )
    optimiser = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=1e-4)
    minimize = loss_kind in {"mse", "infonce"}
    best = (float("inf") if minimize else -1.0, 0, None)
    best_cosine = float("nan")
    best_mse = float("nan")
    best_infonce = float("nan")
    best_top1 = float("nan")
    started = time.time()
    for epoch in range(epochs):
        projector.train()
        order = torch.randperm(
            train_features.shape[0],
            generator=torch.Generator().manual_seed(1_000 * request.seed + epoch),
        ).tolist()
        total, steps = 0.0, 0
        patches = patches_per_step(loss_kind)
        for step, start in enumerate(range(0, len(order), patches)):
            chosen_indices = order[start : start + patches]
            chosen = [groups["train"][index] for index in chosen_indices]
            tokens = train_features[chosen_indices].to(device).float()
            predictions = projector(tokens)
            targets, supports = [], []
            for row in chosen:
                grid, support = teacher_tokens(target, row, data_root, tree_dims)
                targets.append(grid)
                supports.append(support)
            target_tensor = torch.stack(targets).to(device)
            support_tensor = torch.stack(supports).to(device)
            if loss_kind == "infonce":
                generator = torch.Generator().manual_seed(
                    1_000_000 * request.seed + 1_000 * epoch + step
                )
                student, teacher = sample_token_pairs(
                    predictions, target_tensor, support_tensor, INFONCE_TOKENS, generator
                )
                loss = info_nce(student, teacher)
            elif loss_kind == "mse":
                loss = _masked_mse(predictions, target_tensor, support_tensor)
            else:
                loss = 1.0 - _cosine(predictions, target_tensor, support_tensor)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            steps += 1
            if limit and steps >= limit:
                break
        projector.eval()
        with torch.inference_mode():
            cosine_total, mse_total, infonce_total = 0.0, 0.0, 0.0
            top1_correct, top1_total, count = 0, 0, 0
            patches = patches_per_step(loss_kind)
            for start in range(0, val_features.shape[0], patches):
                chosen = groups["validation"][start : start + patches]
                predictions = projector(val_features[start : start + patches].to(device).float())
                targets, supports = [], []
                for row in chosen:
                    grid, support = teacher_tokens(target, row, data_root, tree_dims)
                    targets.append(grid)
                    supports.append(support)
                target_tensor = torch.stack(targets).to(device)
                support_tensor = torch.stack(supports).to(device)
                cosine_total += float(_cosine(predictions, target_tensor, support_tensor))
                mse_total += float(_masked_mse(predictions, target_tensor, support_tensor))
                if loss_kind == "infonce":
                    generator = torch.Generator().manual_seed(10_000_000 + start)
                    student, teacher = sample_token_pairs(
                        predictions, target_tensor, support_tensor, INFONCE_TOKENS, generator
                    )
                    infonce_total += float(info_nce(student, teacher))
                    correct, total_tokens = retrieval_top1(student, teacher)
                    top1_correct += correct
                    top1_total += total_tokens
                count += 1
            validation_cosine = cosine_total / max(count, 1)
            validation_mse = mse_total / max(count, 1)
            validation_infonce = (
                infonce_total / max(count, 1) if loss_kind == "infonce" else float("nan")
            )
            validation_top1 = (
                top1_correct / max(top1_total, 1) if loss_kind == "infonce" else float("nan")
            )
        mean_loss = total / max(steps, 1)
        writer.add_curve("distill_loss", "train", epoch + 1, mean_loss)
        writer.add_curve("cosine", "validation", epoch + 1, validation_cosine)
        if loss_kind == "mse":
            writer.add_curve("mse", "validation", epoch + 1, validation_mse)
        if loss_kind == "infonce":
            writer.add_curve("infonce", "validation", epoch + 1, validation_infonce)
            writer.add_curve("retrieval_top1", "validation", epoch + 1, validation_top1)
        emit(
            request.run_id,
            "distill.epoch",
            f"epoch {epoch + 1}/{epochs}: loss {mean_loss:.4f}, "
            f"val cosine {validation_cosine:.4f}, val MSE {validation_mse:.5f}, "
            f"val InfoNCE {validation_infonce:.4f}, top1 {validation_top1:.4f}",
            progress=(epoch + 1) / epochs,
        )
        if loss_kind == "infonce":
            score = validation_infonce
        elif loss_kind == "mse":
            score = validation_mse
        else:
            score = validation_cosine
        if _loss_improved(score, best[0], loss_kind):
            best = (
                score,
                epoch + 1,
                {key: value.clone() for key, value in projector.state_dict().items()},
            )
            best_cosine = validation_cosine
            best_mse = validation_mse
            best_infonce = validation_infonce
            best_top1 = validation_top1
    if best[2] is not None:
        projector.load_state_dict(best[2])
    fingerprint = _fingerprint(
        writer.record.model_checkpoint_sha256 or "",
        writer.record.dataset.manifest_sha256,
        target,
        request.seed,
        init=tags["init"],
        normalization=tags["normalization_digest"],
    )
    meta = {
        "target": target,
        "dims": dims,
        "hidden": 256,
        "in_channels": EMBED_DIM,
        "seed": request.seed,
        "epochs": epochs,
        "learning_rate": lr,
        "projector_fingerprint": fingerprint,
        "manifest_sha256": writer.record.dataset.manifest_sha256,
        "checkpoint_sha256": writer.record.model_checkpoint_sha256,
        "source_run_id": request.run_id,
        "bands": BANDS,
        "frames": MAX_FRAMES,
        "grid": GRID,
        "teacher_used_at_inference": False,
        "distill_loss": loss_kind,
        "distill_selection": (
            "val_infonce"
            if loss_kind == "infonce"
            else "val_mse"
            if loss_kind == "mse"
            else "val_cosine"
        ),
        **(
            {"infonce_temperature": INFONCE_TEMPERATURE, "infonce_tokens": INFONCE_TOKENS}
            if loss_kind == "infonce"
            else {}
        ),
    }
    payload = {"state_dict": projector.state_dict(), **meta}
    local = Path(request.runs_root) / request.run_id / "projector.pt"
    _save_atomic(local, payload)
    published = _projector_path(target, request.seed, fingerprint)
    _save_atomic(published, payload)
    writer.add_artifact("checkpoint", "projector.pt", "distilled projector E (Prithvi tokens)")
    metrics = [
        _metric("val_cosine", best_cosine, "validation"),
        _metric(
            "distill_train_loss",
            total / max(steps, 1),
            "train",
            direction="minimize",
        ),
    ]
    baseline_tags = {}
    if loss_kind == "mse":
        metrics.append(_metric("val_mse", best_mse, "validation", direction="minimize"))
    if loss_kind == "infonce":
        metrics.append(_metric("val_infonce", best_infonce, "validation", direction="minimize"))
        metrics.append(_metric("val_retrieval_top1", best_top1, "validation"))
    if loss_kind in {"mse", "infonce"}:
        baseline_cosine, baseline_mse = _mean_teacher_baseline(
            groups["train"], groups["validation"], target, data_root, tree_dims, device, dims
        )
        baseline_tags = {
            "mean_baseline_val_cosine": f"{baseline_cosine:.6f}",
            "mean_baseline_val_mse": f"{baseline_mse:.6f}",
        }
    writer.add_metrics(metrics)
    writer.update(
        tags={
            "best_validation_cosine": f"{best_cosine:.6f}",
            "best_validation_mse": f"{best_mse:.6f}" if loss_kind == "mse" else "-",
            "best_validation_infonce": (f"{best_infonce:.6f}" if loss_kind == "infonce" else "-"),
            "best_validation_retrieval_top1": (
                f"{best_top1:.6f}" if loss_kind == "infonce" else "-"
            ),
            "best_epoch": str(best[1]),
            "projector_published": str(published),
            **baseline_tags,
        }
    )
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=round(time.time() - started, 1)
    )
    loss_label = (
        "symmetric InfoNCE" if loss_kind == "infonce" else "MSE" if loss_kind == "mse" else "cosine"
    )
    selection_label = (
        "val InfoNCE"
        if loss_kind == "infonce"
        else "val MSE"
        if loss_kind == "mse"
        else "val cosine"
    )
    summary = (
        f"T11P Prithvi distill {target}: {dims} dims, 14x14 tokens, {loss_label} loss, "
        f"best {selection_label} {best[0]:.6f} at epoch {best[1]}"
    )
    if tags.get("smoke") == "true":
        summary += " [SMOKE]"
    writer.complete(summary)
    emit(request.run_id, "distill.done", "projector saved")
    return 0


def _train_wide_b_stage(
    request,
    writer,
    emit,
    protocol_id,
    rows,
    groups,
    data_root,
    device,
    means,
    stds,
    checkpoint,
    epochs,
    lr,
    backbone_lr,
    batch_size,
    accumulation,
    workers,
    limit,
    tags,
    encoder,
) -> int:
    """Wide-head B stage: last four blocks + a head on the 1024-d tokens.

    Kept separate from the admitted 128-d pair. Both arms build the model (and
    head) before the optional teacher projector, and the sample order is an
    explicit per-epoch permutation.
    """
    target = WIDE_B_STAGE_PROTOCOLS[protocol_id]
    blocks, head_dims, _ = B_STAGE_VARIANTS[protocol_id]
    unfrozen = _unfreeze_last_blocks(encoder)
    if unfrozen == 0:
        raise T11PFailure("B stage unfroze zero encoder parameters")
    model = PrithviT11P(encoder, train_encoder=True, projector=None, dims=head_dims).to(device)
    head = model.head
    teacher_projector = build_projector(B_STAGE_DIMS).to(device) if target is not None else None
    backbone_parameters = [p for p in encoder.parameters() if p.requires_grad]
    head_parameters = list(head.parameters())
    if teacher_projector is not None:
        head_parameters += list(teacher_projector.parameters())
    total_parameters = sum(p.numel() for p in model.parameters())
    if teacher_projector is not None:
        total_parameters += sum(p.numel() for p in teacher_projector.parameters())
    optimiser = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": head_parameters, "lr": lr},
        ],
        weight_decay=1e-4,
    )
    writer.update_training(
        head_id=f"prithvi_t11p_b_stage_last{blocks}_{head_dims}_wide",
        trainable_parameters=sum(p.numel() for p in backbone_parameters + head_parameters),
        total_parameters=total_parameters,
    )
    writer.update(
        tags={
            **tags,
            "b_stage_blocks": str(blocks),
            "b_stage_head_dims": str(head_dims),
            "b_stage_head_projector": "False",
            "b_stage_teacher_projector": str(teacher_projector is not None),
            "b_stage_trainable_backbone_parameters": str(unfrozen),
            "b_stage_backbone_lr": f"{backbone_lr:g}",
            "b_stage_head_lr": f"{lr:g}",
            "b_stage_teacher_weight": str(B_STAGE_TEACHER_WEIGHT if target else 0.0),
            "b_stage_distill_target": target or "none",
            "b_stage_batch": f"{batch_size} x {accumulation} accumulation",
            "batch_order": "explicit per-epoch permutation seeded by 1000*seed+epoch",
        }
    )
    tree_dims = {}
    if target is not None:
        tree, dims = next(
            (tree, dims) for name, (t, tree, dims) in DISTILL_PROTOCOLS.items() if t == target
        )
        tree_dims = {target: (tree, dims)}
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    dataset = PastisS2(data_root, groups["train"])
    val_loader = DataLoader(
        PastisS2(data_root, groups["validation"]),
        batch_size=max(batch_size, 1),
        shuffle=False,
        num_workers=workers,
    )
    test_loader = DataLoader(
        PastisS2(data_root, groups["test"]),
        batch_size=max(batch_size, 1),
        shuffle=False,
        num_workers=workers,
    )

    def stack_items(items):
        # The training loop builds its own micro-batches, so it must apply the
        # same per-band statistics as ``_evaluate``; skipping this trained the
        # wide head on raw DN and then evaluated on normalized input.
        return (
            normalize_images(
                torch.stack([item["image"] for item in items]).to(device), means, stds
            ),
            torch.stack([item["label"] for item in items]).to(device),
        )

    started = time.time()
    for epoch in range(epochs):
        model.train()
        if teacher_projector is not None:
            teacher_projector.train()
        order = torch.randperm(
            len(dataset),
            generator=torch.Generator().manual_seed(1_000 * request.seed + epoch),
        ).tolist()
        total, steps, pending, batches = 0.0, 0, 0, 0
        optimiser.zero_grad(set_to_none=True)
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            items = [dataset[index] for index in chosen]
            images, labels = stack_items(items)
            if int((labels != IGNORE_INDEX).sum()) == 0:
                continue
            features = model.encode(images)
            logits = head(features)
            logits = torch.nn.functional.interpolate(
                logits, size=(LABEL_SIZE, LABEL_SIZE), mode="bilinear", align_corners=False
            )
            loss = criterion(logits.float(), labels)
            if target is not None:
                chosen_rows = [groups["train"][index] for index in chosen]
                targets, supports = [], []
                for row in chosen_rows:
                    grid, support = teacher_tokens(target, row, data_root, tree_dims)
                    targets.append(grid)
                    supports.append(support)
                loss = loss + B_STAGE_TEACHER_WEIGHT * (
                    1.0
                    - _cosine(
                        teacher_projector(features.float()),
                        torch.stack(targets).to(device),
                        torch.stack(supports).to(device),
                    )
                )
            (loss / accumulation).backward()
            pending += 1
            batches += 1
            total += float(loss.detach())
            if pending == accumulation:
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
                pending = 0
                steps += 1
                if limit and steps >= limit:
                    break
        if pending:
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            steps += 1
        mean_loss = total / max(batches, 1)
        writer.add_curve("loss", "train", epoch + 1, mean_loss)
        emit(
            request.run_id,
            "b_stage.epoch",
            f"epoch {epoch + 1}/{epochs}: loss {mean_loss:.4f}, {steps} steps",
            progress=(epoch + 1) / epochs,
        )
    if teacher_projector is not None:
        teacher_projector.eval()
    validation, _ = _evaluate(model, val_loader, means, stds, device)
    writer.add_curve("mean_iou", "validation", epochs, validation["mean_iou"])
    test, group_json = _evaluate(model, test_loader, means, stds, device)
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
    writer.add_confusion(group_json, NUM_CLASSES)
    writer.update(
        tags={
            "validation_samples_evaluated": str(len(groups["validation"])),
            "validation_manifest_samples": str(len(groups["validation"])),
            "validation_expected_samples": tags["validation_expected_samples"],
            "test_samples_evaluated": str(len(groups["test"])),
            "test_manifest_samples": str(len(groups["test"])),
            "test_expected_samples": tags["test_expected_samples"],
        }
    )
    wall = time.time() - started
    state = {
        "state_dict": model.state_dict(),
        "backbone_blocks": len(getattr(encoder, "blocks", [])),
        "trained_blocks": blocks,
        "dims": head_dims,
        "head_projector": False,
        "teacher_projector": teacher_projector is not None,
        "num_classes": NUM_CLASSES,
        "bands": BANDS,
        "frames": MAX_FRAMES,
        "grid": GRID,
        "normalization": {"mean": tags["normalization_mean"], "std": tags["normalization_std"]},
        "seed": request.seed,
        "protocol": protocol_id,
        "teacher": target or "none",
        "encoder_checkpoint": str(checkpoint),
        "encoder_checkpoint_sha256": writer.record.model_checkpoint_sha256,
    }
    if teacher_projector is not None:
        state["state_dict"] |= {
            f"teacher_projector.{name}": value.detach().cpu()
            for name, value in teacher_projector.state_dict().items()
        }
    _save_atomic(Path(request.runs_root) / request.run_id / "model.pt", state)
    writer.add_artifact("checkpoint", "model.pt", "wide-head B-stage blocks, teacher E and head")
    writer.update_training(epochs_completed=epochs, wall_time_seconds=round(wall, 1))
    summary = (
        f"T11P Prithvi wide {target or 'labels-only'}: test mIoU {test['mean_iou']:.4f}, "
        f"last {blocks} blocks + {head_dims}-d head, {epochs} epochs"
    )
    if tags.get("smoke") == "true":
        summary += " [SMOKE]"
    elif epochs != OFFICIAL_EPOCHS:
        summary += " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(request.run_id, "b_stage.metrics", summary)
    return 0


def _train_b_stage(
    request,
    writer,
    emit,
    protocol_id,
    rows,
    groups,
    data_root,
    device,
    means,
    stds,
    checkpoint,
    epochs,
    lr,
    backbone_lr,
    batch_size,
    accumulation,
    workers,
    limit,
    tags,
    encoder,
) -> int:
    target = B_STAGE_PROTOCOLS[protocol_id]
    unfrozen = _unfreeze_last_blocks(encoder)
    if unfrozen == 0:
        raise T11PFailure("B stage unfroze zero encoder parameters")
    projector = build_projector(B_STAGE_DIMS).to(device)
    model = PrithviT11P(encoder, train_encoder=True, projector=projector, dims=B_STAGE_DIMS).to(
        device
    )
    head = model.head
    backbone_parameters = [p for p in encoder.parameters() if p.requires_grad]
    head_parameters = list(projector.parameters()) + list(head.parameters())
    optimiser = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": head_parameters, "lr": lr},
        ],
        weight_decay=1e-4,
    )
    writer.update_training(
        head_id="prithvi_t11p_ft_last4",
        trainable_parameters=sum(p.numel() for p in backbone_parameters + head_parameters),
        total_parameters=sum(p.numel() for p in model.parameters()),
    )
    writer.update(
        tags={
            **tags,
            "b_stage_blocks": str(B_STAGE_BLOCKS),
            "b_stage_trainable_backbone_parameters": str(unfrozen),
            "b_stage_backbone_lr": f"{backbone_lr:g}",
            "b_stage_head_lr": f"{lr:g}",
            "b_stage_teacher_weight": str(B_STAGE_TEACHER_WEIGHT if target else 0.0),
            "b_stage_distill_target": target or "none",
            "b_stage_batch": f"{batch_size} x {accumulation} accumulation",
        }
    )
    tree_dims = {}
    if target is not None:
        tree, dims = next(
            (tree, dims) for name, (t, tree, dims) in DISTILL_PROTOCOLS.items() if t == target
        )
        tree_dims = {target: (tree, dims)}
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    train_loader = DataLoader(
        PastisS2(data_root, groups["train"]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )
    val_loader = DataLoader(
        PastisS2(data_root, groups["validation"]),
        batch_size=max(batch_size, 1),
        shuffle=False,
        num_workers=workers,
    )
    test_loader = DataLoader(
        PastisS2(data_root, groups["test"]),
        batch_size=max(batch_size, 1),
        shuffle=False,
        num_workers=workers,
    )
    started = time.time()
    for epoch in range(epochs):
        model.train()
        total, steps, seen = 0.0, 0, 0
        optimiser.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            images = (batch["image"].to(device) - means) / stds
            labels = batch["label"].to(device)
            if int((labels != IGNORE_INDEX).sum()) == 0:
                continue
            features = model.encode(images)
            projected = projector(features)
            logits = head(projected)
            logits = torch.nn.functional.interpolate(
                logits, size=(LABEL_SIZE, LABEL_SIZE), mode="bilinear", align_corners=False
            )
            loss = criterion(logits, labels)
            if target is not None:
                targets, supports = [], []
                for row in _batch_rows(batch, groups["train"]):
                    grid, support = teacher_tokens(target, row, data_root, tree_dims)
                    targets.append(grid)
                    supports.append(support)
                loss = loss + B_STAGE_TEACHER_WEIGHT * (
                    1.0
                    - _cosine(
                        projected,
                        torch.stack(targets).to(device),
                        torch.stack(supports).to(device),
                    )
                )
            (loss / accumulation).backward()
            if (step + 1) % accumulation == 0:
                optimiser.step()
                optimiser.zero_grad(set_to_none=True)
            total += float(loss.detach())
            steps += 1
            seen += images.shape[0]
            if limit and steps >= limit:
                break
        if steps % accumulation:
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
        writer.add_curve("loss", "train", epoch, total / max(steps, 1))
        emit(
            request.run_id,
            "b_stage.epoch",
            f"epoch {epoch + 1}/{epochs}: loss {total / max(steps, 1):.4f}, {steps} steps",
            progress=(epoch + 1) / epochs,
        )
    validation, _ = _evaluate(model, val_loader, means, stds, device)
    writer.add_curve("mean_iou", "validation", epochs, validation["mean_iou"])
    test, group_json = _evaluate(model, test_loader, means, stds, device)
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
    writer.add_confusion(group_json, NUM_CLASSES)
    writer.update(
        tags={
            "validation_samples_evaluated": str(len(groups["validation"])),
            "validation_manifest_samples": str(len(groups["validation"])),
            "validation_expected_samples": tags["validation_expected_samples"],
            "test_samples_evaluated": str(len(groups["test"])),
            "test_manifest_samples": str(len(groups["test"])),
            "test_expected_samples": tags["test_expected_samples"],
        }
    )
    wall = time.time() - started
    state = {
        "state_dict": model.state_dict(),
        "backbone_blocks": len(getattr(encoder, "blocks", [])),
        "trained_blocks": B_STAGE_BLOCKS,
        "dims": B_STAGE_DIMS,
        "num_classes": NUM_CLASSES,
        "bands": BANDS,
        "frames": MAX_FRAMES,
        "grid": GRID,
        "normalization": {"mean": tags["normalization_mean"], "std": tags["normalization_std"]},
        "seed": request.seed,
        "protocol": protocol_id,
        "teacher": target or "none",
        "encoder_checkpoint": str(checkpoint),
        "encoder_checkpoint_sha256": writer.record.model_checkpoint_sha256,
    }
    _save_atomic(Path(request.runs_root) / request.run_id / "model.pt", state)
    writer.add_artifact("checkpoint", "model.pt", "trained B-stage model")
    writer.update_training(epochs_completed=epochs, wall_time_seconds=round(wall, 1))
    summary = (
        f"T11P Prithvi {target or 'labels-only'}: test mIoU {test['mean_iou']:.4f}, "
        f"last {B_STAGE_BLOCKS} blocks + E + head, {epochs} epochs"
    )
    if tags.get("smoke") == "true":
        summary += " [SMOKE]"
    elif epochs != OFFICIAL_EPOCHS:
        summary += " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(request.run_id, "b_stage.metrics", summary)
    return 0


def _batch_rows(batch: dict, rows: list[dict]) -> list[dict]:
    """Manifest rows for a batch, looked up by sample id rather than position."""
    by_id = {row["sample_id"]: row for row in rows}
    return [by_id[group] for group in batch["group"]]
