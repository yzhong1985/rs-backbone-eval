"""T11P (PASTIS-HD six-band) for the AnySat worker.

Third student on the six-band matched pair, after Clay and Prithvi. AnySat's
public Sentinel-2 entrance always consumes ten bands (B2,B3,B4,B5,B6,B7,B8,
B8A,B11,B12), while T11P provides six (B02/B03/B04/B8A/B11/B12). A small
explicit ``SpectralAdapter`` maps the six onto the ten; zero padding is
forbidden by the evaluation design, so the four missing bands are synthesized
and the adapter's parameter count is reported separately.

Readout protocol (same definition as the Clay/Prithvi T11P tracks, so the
three backbones differ only in the representation):

* ``raw``   - dense S2 tokens at the stride-8 16x16 grid, 1x1-conv head;
* ``s128``  - the same tokens plus a trainable E(1536->128) trained jointly
  with the head on labels;
* ``o128``  - a frozen E distilled from OlmoEarth v1.3 128-dim embeddings,
  plus the frozen adapter that produced the distilled features;
* B stage  - the last three of AnySat's six spatial blocks + final norm, the
  adapter, E and head, with and without the teacher cosine on E. Three of six
  blocks (50% depth) carry ~21.3M parameters, matching the ~16% trainable
  share of Clay/Prithvi's last-four-of-24 recipe; the budget is declared here
  and fixed in advance, not chosen from test scores.

Caveats written into every run: AnySat's pretraining corpus includes
PASTIS-HD, so these PASTIS scores are not a fully unseen task for this
backbone. PASTIS S2 is not the exact HLS product Clay/Prithvi use; the
manifest has no coordinates for AnySat's position metadata; the teacher is a
full-year multimodal product while the student sees four 2019 S2 dates.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rs_backbone_eval.contracts import MetricRecord, WorkerRequest
from rs_backbone_eval.data.datasets.pastis_hd import select_2019_frames
from rs_backbone_eval.data.manifest import audit_task_manifest
from rs_backbone_eval.evaluation import to_metric_records
from rs_backbone_eval.evaluation.groups import GroupedSegmentation
from rs_backbone_eval.evaluation.segmentation import multiclass_metrics
from rs_backbone_eval.repro import random_init_requested
from rs_backbone_eval.timeutil import days_since_reference
from rs_backbone_eval.worker_runtime import RunWriter

from .spectral_adapter import SpectralAdapter
from .t2 import build_backbone
from .t11 import _dense_grid

TASK_ID = "T11P"
BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
S2_INDICES = [0, 1, 2, 7, 8, 9]
MAX_FRAMES = 4
CHIP = 128
LABEL_SIZE = 128
GRID = CHIP // 8  # 80 m token spacing -> 16x16 on the 128 px patch
PATCH_M = 80
DENSE_DIM = 2 * 768
EMBED_DIM = DENSE_DIM
NUM_CLASSES = 20
IGNORE_INDEX = 19
VOID_CLASS = 19
GSD_M = 10.0
OFFICIAL_EPOCHS = 8
CACHE_VERSION = "2026-11-12.t11p-anysat-projector"
B_STAGE_BLOCKS = 3
B_STAGE_BLOCKS_TOTAL = 6
B_STAGE_DIMS = 128
B_STAGE_TEACHER_WEIGHT = 1.0

PROBE_PROTOCOLS = {
    "anysat_t11p_probe_raw": ("raw", None),
    "anysat_t11p_probe_s128": ("supervised", 128),
    "anysat_t11p_probe_o128": ("distilled", "oe128"),
    "anysat_t11p_probe_pca128": ("pca128", 128),
    "anysat_t11p_probe_o128_mse": ("distilled", "oe128mse"),
    "anysat_t11p_probe_o128_nce": ("distilled", "oe128nce"),
}
DISTILL_PROTOCOLS = {
    "anysat_t11p_distill_o128": ("oe128", "olmoearth-2019-128", 128),
    "anysat_t11p_distill_o128_mse": ("oe128mse", "olmoearth-2019-128", 128),
    "anysat_t11p_distill_o128_nce": ("oe128nce", "olmoearth-2019-128", 128),
}
DISTILL_LOSS = {
    "anysat_t11p_distill_o128_mse": "mse",
    "anysat_t11p_distill_o128_nce": "infonce",
}
B_STAGE_PROTOCOLS = {
    "anysat_t11p_ft_last3": None,
    "anysat_t11p_ft_last3_distill": "oe128",
    # Wide-head pair: the label path reads the 1536-d dense tokens directly
    # instead of the 128-d projector. The teacher arm keeps a separate 128-d
    # teacher projector for the cosine; it does not feed the label head.
    "anysat_t11p_ft_last3_wide": None,
    "anysat_t11p_ft_last3_wide_distill": "oe128",
}
# protocol -> (unfrozen blocks, head input width, does a projector feed the head?)
B_STAGE_VARIANTS = {
    "anysat_t11p_ft_last3": (B_STAGE_BLOCKS, B_STAGE_DIMS, True),
    "anysat_t11p_ft_last3_distill": (B_STAGE_BLOCKS, B_STAGE_DIMS, True),
    "anysat_t11p_ft_last3_wide": (B_STAGE_BLOCKS, EMBED_DIM, False),
    "anysat_t11p_ft_last3_wide_distill": (B_STAGE_BLOCKS, EMBED_DIM, False),
}


class AnySatT11PFailure(Exception):
    """A gate turns into a failed run with a readable reason."""


def _minimum_train_rows() -> int:
    """1200 full-budget rows, or any size in a declared label-fraction panel."""
    return 1 if os.environ.get("RS_EVAL_LABEL_FRACTION") else 1200


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


def read_frames(row: dict, data_root: Path) -> tuple[np.ndarray, list[int], list[int]]:
    """(T,6,128,128) raw DN for the four 2019 frames, plus dates and indices."""
    aux = row.get("aux_paths") or {}
    stack_path = aux.get("s2_path")
    if not stack_path:
        raise AnySatT11PFailure(f"sample {row['sample_id']}: aux_paths has no s2_path")
    dates_source = aux.get("dates_source") or str(Path(row["image_paths"][0]).with_suffix(".json"))
    sidecar = data_root / dates_source
    if not sidecar.is_file():
        raise AnySatT11PFailure(f"sample {row['sample_id']}: sidecar {sidecar.name} missing")
    dates = [int(date) for date in json.loads(sidecar.read_text(encoding="utf-8"))["dates_s2"]]
    frames = select_2019_frames(dates, MAX_FRAMES)
    stack = np.load(data_root / stack_path, mmap_mode="r")
    if not 0 <= max(S2_INDICES) < stack.shape[1]:
        raise AnySatT11PFailure(f"sample {row['sample_id']}: S2 stack has {stack.shape[1]} bands")
    pixels = np.asarray(stack[frames][:, S2_INDICES], dtype=np.float32)
    return pixels, [int(dates[index]) for index in frames], frames


class PastisS2(Dataset):
    """One PASTIS patch: four normalised-later six-band frames, a label and dates."""

    def __init__(self, data_root: Path, rows: list[dict]) -> None:
        self.data_root = data_root
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        pixels, dates, _ = read_frames(row, self.data_root)
        return {
            "image": torch.from_numpy(pixels),
            "label": torch.from_numpy(_labels(row, self.data_root)),
            "dates": torch.tensor(
                [days_since_reference(date) for date in dates], dtype=torch.float32
            ),
            "group": row["group_id"],
        }


def _tile_sample(rows: list[dict], limit: int) -> list[dict]:
    split = sorted((row for row in rows if row["split"] == "train"), key=lambda r: r["sample_id"])
    by_tile: dict[str, list[dict]] = {}
    for row in split:
        by_tile.setdefault((row.get("aux_paths") or {}).get("tile", ""), []).append(row)
    chosen: list[dict] = []
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


STATS_PATCHES = 24


def _norm_cache_key(manifest_digest: str, rows: list[dict]) -> str:
    """Cache key for the normalisation statistics of one exact train subset.

    The manifest digest alone is not enough: a ``TRAIN_LIMIT`` run computes its
    statistics from a prefix of the train fold, and reusing that cache in a
    full run silently changes the formal input normalisation (review
    2026-11-12, P2).
    """
    subset = hashlib.sha256(
        ",".join(sorted(row["sample_id"] for row in rows)).encode("utf-8")
    ).hexdigest()[:16]
    return f"anysat_t11p_norm_{manifest_digest[:16]}-{subset}-p{STATS_PATCHES}-v2"


def channel_statistics(
    rows: list[dict], data_root: Path, cache_dir: Path, manifest_digest: str
) -> tuple[list[float], list[float]]:
    """Per-band mean/std of the raw DN over a fixed tile-balanced train sample."""
    key = f"{_norm_cache_key(manifest_digest, rows)}.json"
    cache = cache_dir / key
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return list(payload["mean"]), list(payload["std"])
    total = np.zeros(len(BANDS), dtype=np.float64)
    squares = np.zeros(len(BANDS), dtype=np.float64)
    seen = 0
    for row in _tile_sample(rows, STATS_PATCHES):
        pixels, _, _ = read_frames(row, data_root)
        total += pixels.sum(axis=(0, 2, 3), dtype=np.float64)
        squares += np.square(pixels, dtype=np.float64).sum(axis=(0, 2, 3))
        seen += pixels.shape[0] * pixels.shape[2] * pixels.shape[3]
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
                "bands": BANDS,
                "frames": MAX_FRAMES,
                "manifest_sha256": manifest_digest,
                "train_rows": len(rows),
                "cache_key": key,
            }
        ),
        encoding="utf-8",
    )
    temporary.replace(cache)
    return [float(value) for value in mean], [float(value) for value in std]


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


def _spatial_blocks(backbone) -> nn.ModuleList:
    encoder = backbone.model.spatial_encoder
    blocks = getattr(encoder, "predictor_blocks", None)
    if blocks is None:
        raise AnySatT11PFailure("AnySat spatial encoder does not expose predictor_blocks")
    return blocks


def _unfreeze_last_blocks(backbone, count: int = B_STAGE_BLOCKS) -> int:
    encoder = backbone.model.spatial_encoder
    blocks = _spatial_blocks(backbone)
    if count > len(blocks):
        raise AnySatT11PFailure(f"cannot unfreeze {count} of {len(blocks)} blocks")
    for block in blocks[-count:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    norm = getattr(encoder, "predictor_norm", None)
    if norm is None:
        raise AnySatT11PFailure("AnySat spatial encoder does not expose predictor_norm")
    for parameter in norm.parameters():
        parameter.requires_grad = True
    return sum(parameter.numel() for parameter in blocks[-count:].parameters()) + sum(
        parameter.numel() for parameter in norm.parameters()
    )


class AnySatT11P(nn.Module):
    """Frozen (or partly unfrozen) AnySat + spectral adapter + E + token head."""

    def __init__(
        self,
        backbone,
        adapter: SpectralAdapter,
        *,
        projector: nn.Module | None,
        dims: int,
        blocks: int = 0,
        freeze_adapter: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.projector = projector
        self.head = build_head(dims)
        self.blocks = blocks
        self.freeze_adapter = freeze_adapter
        self.adapter_trainable = not freeze_adapter
        if freeze_adapter:
            for parameter in self.adapter.parameters():
                parameter.requires_grad = False
        self.needs_grad = self.adapter_trainable or blocks > 0

    def train(self, mode: bool = True):
        super().train(mode)
        # Everything except the explicitly unfrozen blocks stays deterministic.
        self.backbone.eval()
        if mode:
            # Frozen modules stay in eval mode; only trainable ones switch
            # (review 2026-11-12, P2).
            self.adapter.train(self.adapter_trainable)
            self.head.train(any(parameter.requires_grad for parameter in self.head.parameters()))
            if self.projector is not None:
                self.projector.train(
                    any(parameter.requires_grad for parameter in self.projector.parameters())
                )
            if self.blocks:
                for block in _spatial_blocks(self.backbone)[-self.blocks :]:
                    block.train()
                self.backbone.model.spatial_encoder.predictor_norm.train()
        return self

    def encode(self, images: torch.Tensor, dates: torch.Tensor) -> torch.Tensor:
        """(B,T,6,128,128) normalised -> (B,1536,16,16) dense tokens."""
        adapted = self.adapter(images)
        context = torch.enable_grad() if self.needs_grad else torch.no_grad()
        with context:
            dense = self.backbone(
                {"s2": adapted, "s2_dates": dates},
                patch_size=PATCH_M,
                output="dense",
                output_modality="s2",
            )
        return _dense_grid(dense)

    def forward(self, images: torch.Tensor, dates: torch.Tensor) -> torch.Tensor:
        features = self.encode(images, dates)
        if self.projector is not None:
            features = self.projector(features)
        logits = self.head(features)
        return torch.nn.functional.interpolate(
            logits, size=(LABEL_SIZE, LABEL_SIZE), mode="bilinear", align_corners=False
        )

    def forward_with_projection(
        self, images: torch.Tensor, dates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(images, dates)
        projected = self.projector(features) if self.projector is not None else features
        logits = self.head(projected)
        logits = torch.nn.functional.interpolate(
            logits, size=(LABEL_SIZE, LABEL_SIZE), mode="bilinear", align_corners=False
        )
        return logits, projected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_root() -> Path:
    base = os.environ.get("RS_EVAL_CACHE_ROOT")
    return Path(base) if base else Path.home() / "rs-eval-cache"


def _budget_signature(epochs: int, train_rows: int, limit: int | None, smoke: bool) -> str:
    """Identity of the training budget, so a diagnostic run cannot share a
    formal adapter/projector path (review 2026-11-12, P1)."""
    return f"e{epochs}-n{train_rows}-l{limit or 0}-s{int(smoke)}"


def _fingerprint(
    checkpoint_sha256: str,
    manifest_sha256: str,
    target: str,
    seed: int,
    *,
    init: str,
    normalization: str,
    epochs: int,
    budget: str,
) -> str:
    payload = {
        "version": CACHE_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "task": TASK_ID,
        "bands": BANDS,
        "frames": MAX_FRAMES,
        "grid": GRID,
        "target": target,
        "seed": seed,
        "init": init,
        "normalization": normalization,
        "epochs": epochs,
        "budget": budget,
        "adapter": "learned 6->10 spectral adapter, hidden 32",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _projector_dir(fingerprint: str, seed: int) -> Path:
    return _cache_root() / "pastis-t11p-anysat" / fingerprint / f"seed{seed}"


def _projector_path(target: str, seed: int, fingerprint: str) -> Path:
    return _projector_dir(fingerprint, seed) / f"{target}.pt"


def _adapter_path(seed: int, fingerprint: str) -> Path:
    return _projector_dir(fingerprint, seed) / "adapter_raw.pt"


def _probe_payload(
    model,
    projector,
    *,
    dims: int,
    protocol_id: str,
    kind: str,
    seed: int,
    adapter_fingerprint: str,
    adapter_source: str,
    normalization: dict[str, str],
    checkpoint_sha256: str,
    manifest_sha256: str,
) -> dict:
    """Self-contained probe checkpoint: head plus its optional projector."""
    state = {
        f"head.{name}": value.detach().cpu() for name, value in model.head.state_dict().items()
    }
    if projector is not None:
        state |= {
            f"projector.{name}": value.detach().cpu()
            for name, value in projector.state_dict().items()
        }
    return {
        "state_dict": state,
        "head_dims": dims,
        "projector_dims": dims if projector is not None else None,
        "projector_trainable": kind == "supervised",
        "bands": BANDS,
        "frames": MAX_FRAMES,
        "grid": GRID,
        "num_classes": NUM_CLASSES,
        "normalization": normalization,
        "adapter_fingerprint": adapter_fingerprint,
        "adapter_source": adapter_source,
        "seed": seed,
        "protocol": protocol_id,
        "readout": kind,
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "teacher_used_at_inference": False,
    }


def _save_atomic(path: Path, payload: dict) -> None:
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
    """Teacher pooled from the 128 px grid to AnySat's 16x16 token grid."""
    import torch.nn.functional as F

    tree, dims = tree_dims[target]
    path = data_root / "derived" / tree / "features" / "2019" / f"{row['sample_id']}.npz"
    if not path.is_file():
        raise AnySatT11PFailure(f"sample {row['sample_id']}: teacher payload {path} missing")
    with np.load(path) as archive:
        raw = archive["features"]
        valid = archive["valid"] if "valid" in archive.files else np.ones(raw.shape[1:], bool)
    if raw.shape[0] != dims:
        raise AnySatT11PFailure(
            f"sample {row['sample_id']}: teacher {target} has {raw.shape[0]} dims, expected {dims}"
        )
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

    unit_prediction = F.normalize(prediction.float(), dim=1)
    unit_target = F.normalize(target.float(), dim=1)
    cosine = (unit_prediction * unit_target).sum(dim=1)
    if bool(mask.any()):
        return cosine[mask].mean()
    return cosine.mean()


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over observed tokens, with no length normalisation."""
    error = ((prediction.float() - target.float()) ** 2).mean(dim=1)
    if bool(mask.any()):
        return error[mask.bool()].mean()
    return error.mean()


def _loss_improved(score: float, best: float, loss_kind: str) -> bool:
    """Epoch selection: MSE and InfoNCE minimise, cosine maximises."""
    return score < best if loss_kind in {"mse", "infonce"} else score > best


def _mean_teacher_baseline(
    train_teacher: list[tuple[torch.Tensor, torch.Tensor]],
    val_teacher: list[tuple[torch.Tensor, torch.Tensor]],
    dims: int,
    device: str,
) -> tuple[float, float]:
    """Cosine/MSE of predicting the train-set mean teacher vector."""
    total = torch.zeros(dims, device=device)
    count = 0
    for grid, support in train_teacher:
        values = grid.to(device)
        mask = support.to(device).bool()
        total += values[:, mask].sum(dim=1)
        count += int(mask.sum())
    mean_vector = total / max(count, 1)
    cosine_sum, mse_sum, tokens = 0.0, 0.0, 0
    for grid, support in val_teacher:
        target_tensor = grid.unsqueeze(0).to(device)
        support_tensor = support.unsqueeze(0).to(device)
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


def _metric(name: str, value: float, split: str, direction: str = "maximize") -> MetricRecord:
    return MetricRecord(
        name=name,
        direction=direction,
        value=round(float(value), 6),
        split=split,
        is_primary=False,
    )


def _evaluate(
    model: AnySatT11P,
    loader: DataLoader,
    means: torch.Tensor,
    stds: torch.Tensor,
    device: str,
) -> tuple[dict, dict[str, list[list[int]]]]:
    collector = GroupedSegmentation(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    model.eval()
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"),
    ):
        for batch in loader:
            images = (batch["image"].to(device) - means) / stds
            logits = model(images, batch["dates"].to(device))
            collector.update(
                list(batch["group"]), batch["label"].numpy(), logits.argmax(1).cpu().numpy()
            )
    return multiclass_metrics(collector.global_matrix), collector.group_confusion()


def _epoch_order(size: int, seed: int, epoch: int) -> list[int]:
    """Deterministic per-epoch permutation shared by both arms of a pair.

    The global RNG is not used: initialising an extra module in one arm must not
    change the other arm's sample order (review 2026-11-12, P1).
    """
    return torch.randperm(
        size,
        generator=torch.Generator().manual_seed(1_000 * seed + epoch),
    ).tolist()


def _stack_items(items: list[dict], device: str) -> dict:
    return {
        "image": torch.stack([item["image"] for item in items]).to(device),
        "label": torch.stack([item["label"] for item in items]).to(device),
        "dates": torch.stack([item["dates"] for item in items]).to(device),
    }


def _collate(rows: list[dict], data_root: Path, device: str) -> dict:
    dataset = PastisS2(data_root, rows)
    return _stack_items([dataset[index] for index in range(len(rows))], device)


def _label_tensor(rows: list[dict], data_root: Path) -> torch.Tensor:
    return torch.stack([torch.from_numpy(_labels(row, data_root)) for row in rows])


def _frozen_features(
    model: AnySatT11P,
    rows: list[dict],
    data_root: Path,
    means: torch.Tensor,
    stds: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Stacked float16 CPU tokens for a frozen adapter + backbone.

    Only used when the adapter and backbone are both frozen (S128/O128/distill);
    the raw probe and the B stage must keep the graph alive and train online.
    """
    model.eval()
    parts: list[torch.Tensor] = []
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"),
    ):
        for start in range(0, len(rows), 1):
            batch = _collate(rows[start : start + 1], data_root, device)
            images = (batch["image"] - means) / stds
            features = model.encode(images, batch["dates"])
            parts.append(features.to(torch.float16).cpu())
    if not parts:
        return torch.empty((0, EMBED_DIM, GRID, GRID), dtype=torch.float16)
    return torch.cat(parts)


def _evaluate_tokens(
    model: AnySatT11P,
    features: torch.Tensor,
    labels: torch.Tensor,
    groups: list[str],
    device: str,
    *,
    chunk: int = 16,
) -> tuple[dict, dict[str, list[list[int]]]]:
    collector = GroupedSegmentation(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    model.eval()
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"),
    ):
        for start in range(0, features.shape[0], chunk):
            tokens = features[start : start + chunk].to(device).float()
            if model.projector is not None:
                tokens = model.projector(tokens)
            logits = model.head(tokens)
            logits = torch.nn.functional.interpolate(
                logits, size=(LABEL_SIZE, LABEL_SIZE), mode="bilinear", align_corners=False
            )
            collector.update(
                groups[start : start + chunk],
                labels[start : start + chunk].numpy(),
                logits.argmax(1).cpu().numpy(),
            )
    return multiclass_metrics(collector.global_matrix), collector.group_confusion()


def _train_t11p(
    request: WorkerRequest,
    writer: RunWriter,
    emit,
    *,
    checkpoint: Path,
    protocol_id: str,
    task_spec,
) -> int:
    """Dispatcher for the probe, projector-distillation and B-stage protocols."""
    device = os.environ.get("ANYSAT_WORKER_DEVICE") or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    manifest = Path(request.manifest_path)
    data_root = Path(request.data_root)
    audit = audit_task_manifest(manifest, task_spec)
    strict_errors = audit.error_count - len(audit.leakage)
    if strict_errors:
        raise AnySatT11PFailure(f"manifest audit failed: {audit.all_errors()[:5]}")
    rows = _rows(manifest)
    groups = _groups(rows)
    train_rows = groups["train"]
    eval_limit = _int_env("ANYSAT_T11P_EVAL_LIMIT", 0)
    if eval_limit:
        groups["validation"] = groups["validation"][:eval_limit]
        groups["test"] = groups["test"][:eval_limit]
    if len(train_rows) < _minimum_train_rows():
        raise AnySatT11PFailure(
            f"train fold has {len(train_rows)} rows; T11P formal runs need the full "
            "1,455-row fold unless RS_EVAL_LABEL_FRACTION declares a few-shot panel"
        )
    train_limit = _int_env("ANYSAT_T11P_TRAIN_LIMIT", 0)
    if train_limit:
        groups["train"] = groups["train"][:train_limit]
        train_rows = groups["train"]
    from rs_backbone_eval.repro import seed_everything

    seed_everything(request.seed)

    from rs_backbone_eval.data.manifest import manifest_sha256

    manifest_digest = manifest_sha256(manifest)
    cache_dir = Path(request.cache_root) / "anysat_t11p"
    mean, std = channel_statistics(train_rows, data_root, cache_dir, manifest_digest)
    means = torch.tensor(mean, device=device).view(1, 1, -1, 1, 1)
    stds = torch.tensor(std, device=device).view(1, 1, -1, 1, 1)
    epochs = _int_env("ANYSAT_T11P_EPOCHS", OFFICIAL_EPOCHS)
    batch_size = _int_env("ANYSAT_T11P_BATCH", 1)
    accumulation = _int_env("ANYSAT_T11P_ACCUM", 2)
    lr = _float_env("ANYSAT_T11P_LR", 3e-3)
    backbone_lr = _float_env("ANYSAT_T11P_BACKBONE_LR", 1e-4)
    workers = _int_env("ANYSAT_T11P_NUM_WORKERS", 4)
    limit = _int_env("ANYSAT_T11P_LIMIT_BATCHES", 0) or None
    partial_eval = os.environ.get("ANYSAT_T11P_ALLOW_PARTIAL_EVAL") == "1"
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
                raise AnySatT11PFailure(
                    f"{split} split has {len(groups[split])} rows; T11P formal runs "
                    f"need {expected} (set ANYSAT_T11P_ALLOW_PARTIAL_EVAL=1 only for smoke)"
                )
    init = "random" if random_init_requested() else "pretrained"
    normalization = hashlib.sha256(
        json.dumps([mean, std], separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]

    def eval_loader(rows_split: list[dict]) -> DataLoader:
        return DataLoader(
            PastisS2(data_root, rows_split),
            batch_size=1,
            shuffle=False,
            num_workers=workers,
        )

    val_loader = eval_loader(groups["validation"])
    test_loader = eval_loader(groups["test"])
    backbone = build_backbone(checkpoint).to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    writer.update_input(
        preprocessing_id="pastis_s2_sixband_train_stats+learned_spectral_adapter",
        bands=list(BANDS),
        time_steps=MAX_FRAMES,
        gsd_m=GSD_M,
    )
    common_tags = {
        "student_input": "S2 B02/B03/B04/B8A/B11/B12 raw DN, 4 frames from 2019",
        "band_semantics": "B8A~HLS NIR_NARROW, B11~SWIR_1, B12~SWIR_2",
        "input_adapter": "learned 6->10 spectral adapter (hidden 32), no zero padding",
        "adapter_parameters": str(sum(p.numel() for p in SpectralAdapter(BANDS).parameters())),
        "synthesized_anysat_bands": ",".join(SpectralAdapter(BANDS).synthesized_bands),
        "normalization": "PASTIS train-split per-band mean/std (raw DN)",
        "normalization_mean": ",".join(f"{value:.2f}" for value in mean),
        "normalization_std": ",".join(f"{value:.2f}" for value in std),
        "token_grid": f"{GRID}x{GRID}",
        "token_spacing_m": str(PATCH_M),
        "coords": "zero-filled (PASTIS manifest has no coordinates)",
        "teacher_used_at_inference": "no",
        "label_mapping": "0..19 kept; void 19 ignored",
        "effective_batch_patches": str(batch_size * accumulation),
        "micro_batch": str(batch_size),
        "grad_accum": str(accumulation),
        "device": device,
        "host": (
            torch.cuda.get_device_name(0).lower().replace(" ", "-") if device == "cuda" else "cpu"
        ),
        "torch": torch.__version__,
        "anysat_src": os.environ.get("ANYSAT_SRC", "unset"),
        "pretraining_overlap": "AnySat pretraining corpus includes PASTIS-HD",
        "train_samples": str(len(train_rows)),
        "validation_samples": str(len(groups["validation"])),
        "test_samples": str(len(groups["test"])),
        "init": init,
        "normalization_digest": normalization,
        "test_expected_samples": str(OFFICIAL_TEST_SAMPLES[TASK_ID]),
        "validation_expected_samples": str(OFFICIAL_VALIDATION_SAMPLES[TASK_ID]),
    }
    common_tags["budget_signature"] = _budget_signature(epochs, len(groups["train"]), limit, smoke)
    if smoke:
        common_tags["smoke"] = "true"
        common_tags["panel"] = "diagnostic"
    writer.update(tags=common_tags)

    if protocol_id in DISTILL_PROTOCOLS:
        return _train_distill(
            request,
            writer,
            emit,
            protocol_id,
            groups,
            data_root,
            device,
            means,
            stds,
            epochs,
            lr,
            batch_size,
            accumulation,
            limit,
            common_tags,
            backbone,
        )
    if protocol_id in PROBE_PROTOCOLS:
        return _train_probe(
            request,
            writer,
            emit,
            protocol_id,
            groups,
            data_root,
            device,
            means,
            stds,
            epochs,
            lr,
            batch_size,
            accumulation,
            limit,
            common_tags,
            val_loader,
            test_loader,
            backbone,
        )
    if protocol_id in B_STAGE_PROTOCOLS:
        return _train_b_stage(
            request,
            writer,
            emit,
            protocol_id,
            groups,
            data_root,
            device,
            means,
            stds,
            epochs,
            lr,
            backbone_lr,
            batch_size,
            accumulation,
            workers,
            limit,
            common_tags,
            backbone,
        )
    raise AnySatT11PFailure(f"AnySat T11P has no protocol {protocol_id!r}")


def _train_probe(
    request,
    writer,
    emit,
    protocol_id,
    groups,
    data_root,
    device,
    means,
    stds,
    epochs,
    lr,
    batch_size,
    accumulation,
    limit,
    tags,
    val_loader,
    test_loader,
    backbone,
) -> int:
    kind, target = PROBE_PROTOCOLS[protocol_id]
    train_rows = groups["train"]
    budget = _budget_signature(epochs, len(train_rows), limit, tags.get("smoke") == "true")
    adapter = SpectralAdapter(BANDS).to(device)
    # The adapter is a shared, label-supervised input mapping: the raw probe
    # trains it, and the S128/O128 readouts load that exact state so the
    # three probes differ only in E and the head (review note 2026-11-12).
    adapter_fingerprint = _fingerprint(
        writer.record.model_checkpoint_sha256 or "",
        writer.record.dataset.manifest_sha256,
        "raw",
        request.seed,
        init=tags["init"],
        normalization=tags["normalization_digest"],
        epochs=epochs,
        budget=budget,
    )
    if kind != "raw":
        adapter_path = _adapter_path(request.seed, adapter_fingerprint)
        if not adapter_path.is_file():
            raise AnySatT11PFailure(
                f"no label-supervised 6->10 adapter for seed {request.seed} at "
                f"{adapter_path}; run anysat_t11p_probe_raw first"
            )
        adapter_payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
        adapter.load_state_dict(adapter_payload["state_dict"])
        for parameter in adapter.parameters():
            parameter.requires_grad = False
        adapter.eval()
        writer.update(
            tags={
                "input_adapter_source": str(adapter_path),
                "input_adapter_sha256": _sha256(adapter_path),
                "input_adapter_source_run": str(adapter_payload.get("source_run_id", "")),
                "input_adapter_source_epochs": str(adapter_payload.get("epochs", "")),
            }
        )
    projector = None
    dims = EMBED_DIM
    if kind == "pca128":
        dims = 128
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
                epochs=epochs,
                budget=budget,
            )
            path = _projector_path(target, request.seed, fingerprint)
            if not path.is_file():
                raise AnySatT11PFailure(
                    f"no distilled projector for {target} seed {request.seed} at {path}; "
                    f"run {DISTILL_PROTOCOLS_BY_TARGET[target]} first"
                )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            projector.load_state_dict(payload["state_dict"])
            for parameter in projector.parameters():
                parameter.requires_grad = False
            projector.eval()
            writer.update(
                tags={
                    "transfer_projector": str(path),
                    "transfer_projector_sha256": _sha256(path),
                    "transfer_projector_source_run": str(payload.get("source_run_id", "")),
                }
            )
    model = AnySatT11P(
        backbone,
        adapter,
        projector=projector,
        dims=dims,
        blocks=0,
        freeze_adapter=kind != "raw",
    ).to(device)
    parameters = list(model.head.parameters())
    if kind == "supervised":
        parameters += list(projector.parameters())
    elif kind == "raw":
        parameters += list(model.adapter.parameters())
    writer.update_training(
        head_id=f"anysat_t11p_probe_{kind}_{dims}",
        trainable_parameters=sum(p.numel() for p in parameters),
        total_parameters=sum(p.numel() for p in model.parameters()),
    )
    writer.update(
        tags={
            **tags,
            "probe_readout": kind,
            "probe_dims": str(dims),
            "adapter_trainable": str(kind == "raw").lower(),
        }
    )
    train_rows = groups["train"]
    emit(
        request.run_id,
        "probe.prepare",
        f"T11P AnySat {kind} readout {dims}, {len(train_rows)} train / "
        f"{len(groups['validation'])} val / {len(groups['test'])} test",
    )
    optimiser = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    started = time.time()

    # The S128/O128 readouts share the frozen adapter + backbone, so their
    # tokens are computed once and reused across epochs (review 2026-11-12,
    # P3). The raw probe trains the adapter and must stay online.
    frozen = kind != "raw"
    if frozen:
        train_features = _frozen_features(model, train_rows, data_root, means, stds, device)
        train_labels = _label_tensor(train_rows, data_root)
        val_features = _frozen_features(model, groups["validation"], data_root, means, stds, device)
        val_labels = _label_tensor(groups["validation"], data_root)
        test_features = _frozen_features(model, groups["test"], data_root, means, stds, device)
        test_labels = _label_tensor(groups["test"], data_root)
        writer.update(
            tags={
                "frozen_feature_shape": "x".join(str(dim) for dim in train_features.shape[1:]),
                "frozen_feature_dtype": "float16",
                "frozen_feature_rows": str(train_features.shape[0]),
            }
        )

    if kind == "pca128":
        from rs_backbone_eval.pca import fit_token_pca, project_token_pca

        train_array = train_features.float().numpy()
        mean_pca, components_pca, explained = fit_token_pca(train_array, dims)
        train_features = torch.from_numpy(project_token_pca(train_array, mean_pca, components_pca))
        val_features = torch.from_numpy(
            project_token_pca(val_features.float().numpy(), mean_pca, components_pca)
        )
        test_features = torch.from_numpy(
            project_token_pca(test_features.float().numpy(), mean_pca, components_pca)
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
                "pca_input": "shared adapter + frozen backbone tokens",
            }
        )

    for epoch in range(epochs):
        model.train()
        order = _epoch_order(len(train_rows), request.seed, epoch)
        total, steps, pending, batches = 0.0, 0, 0, 0
        optimiser.zero_grad(set_to_none=True)
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            if frozen:
                features = train_features[chosen].to(device).float()
                labels = train_labels[chosen].to(device)
                if int((labels != IGNORE_INDEX).sum()) == 0:
                    continue
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
                ):
                    tokens = model.projector(features) if model.projector is not None else features
                    logits = model.head(tokens)
                    logits = torch.nn.functional.interpolate(
                        logits,
                        size=(LABEL_SIZE, LABEL_SIZE),
                        mode="bilinear",
                        align_corners=False,
                    )
            else:
                batch = _collate([train_rows[index] for index in chosen], data_root, device)
                images = (batch["image"] - means) / stds
                labels = batch["label"]
                if int((labels != IGNORE_INDEX).sum()) == 0:
                    continue
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
                ):
                    logits = model(images, batch["dates"])
            loss = criterion(logits.float(), labels)
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
            "probe.epoch",
            f"epoch {epoch + 1}/{epochs} loss={mean_loss:.4f}",
            progress=(epoch + 1) / epochs,
        )
    if kind == "raw":
        payload = {
            "state_dict": adapter.state_dict(),
            "fingerprint": adapter_fingerprint,
            "seed": request.seed,
            "epochs": epochs,
            "source_run_id": request.run_id,
            "bands": BANDS,
            "init": tags["init"],
            "normalization_digest": tags["normalization_digest"],
            "manifest_sha256": writer.record.dataset.manifest_sha256,
            "checkpoint_sha256": writer.record.model_checkpoint_sha256,
            "teacher_used": False,
        }
        local = Path(request.runs_root) / request.run_id / "adapter.pt"
        _save_atomic(local, payload)
        published = _adapter_path(request.seed, adapter_fingerprint)
        _save_atomic(published, payload)
        writer.add_artifact("checkpoint", "adapter.pt", "label-supervised 6->10 adapter")
        writer.update(
            tags={
                "adapter_published": str(published),
                "artifact_scope": "diagnostic" if tags.get("smoke") == "true" else "formal",
            }
        )

    # A probe is a completed experiment only if its task head can be reloaded.
    # Save the head (and the S128/supervised or O128/frozen projector) with
    # the normalisation and adapter provenance (review 2026-11-12, P1).
    probe_payload = _probe_payload(
        model,
        projector,
        dims=dims,
        protocol_id=protocol_id,
        kind=kind,
        seed=request.seed,
        adapter_fingerprint=adapter_fingerprint,
        adapter_source=str(_adapter_path(request.seed, adapter_fingerprint)),
        normalization={
            "mean": tags["normalization_mean"],
            "std": tags["normalization_std"],
        },
        checkpoint_sha256=writer.record.model_checkpoint_sha256 or "",
        manifest_sha256=writer.record.dataset.manifest_sha256,
    )
    local = Path(request.runs_root) / request.run_id / "head.pt"
    _save_atomic(local, probe_payload)
    writer.add_artifact("checkpoint", "head.pt", "probe task head (+ projector)")
    writer.update(tags={"head_artifact": "head.pt"})
    if projector is not None:
        projector_payload = {
            **{key: value for key, value in probe_payload.items() if key != "state_dict"},
            "state_dict": {
                name: value.detach().cpu() for name, value in projector.state_dict().items()
            },
            "trainable": kind == "supervised",
        }
        projector_local = Path(request.runs_root) / request.run_id / "projector.pt"
        _save_atomic(projector_local, projector_payload)
        writer.add_artifact("checkpoint", "projector.pt", "probe projector E")
        writer.update(tags={"projector_artifact": "projector.pt"})

    if frozen:
        val_groups = [row["group_id"] for row in groups["validation"]]
        test_groups = [row["group_id"] for row in groups["test"]]
        validation, _ = _evaluate_tokens(model, val_features, val_labels, val_groups, device)
        test, group_json = _evaluate_tokens(model, test_features, test_labels, test_groups, device)
    else:
        validation, _ = _evaluate(model, val_loader, means, stds, device)
        test, group_json = _evaluate(model, test_loader, means, stds, device)
    writer.add_curve("mean_iou", "validation", epochs, validation["mean_iou"])
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
    writer.update_training(epochs_completed=epochs, wall_time_seconds=round(wall, 1))
    summary = (
        f"T11P AnySat {kind}/{dims}: test mIoU {test['mean_iou']:.4f}, "
        f"{epochs} epochs, frozen backbone"
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
    groups,
    data_root,
    device,
    means,
    stds,
    epochs,
    lr,
    batch_size,
    accumulation,
    limit,
    tags,
    backbone,
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
    train_rows = groups["train"]
    budget = _budget_signature(epochs, len(train_rows), limit, tags.get("smoke") == "true")
    adapter = SpectralAdapter(BANDS).to(device)
    adapter_fingerprint = _fingerprint(
        writer.record.model_checkpoint_sha256 or "",
        writer.record.dataset.manifest_sha256,
        "raw",
        request.seed,
        init=tags["init"],
        normalization=tags["normalization_digest"],
        epochs=epochs,
        budget=budget,
    )
    adapter_path = _adapter_path(request.seed, adapter_fingerprint)
    if not adapter_path.is_file():
        raise AnySatT11PFailure(
            f"no label-supervised 6->10 adapter for seed {request.seed} at "
            f"{adapter_path}; run anysat_t11p_probe_raw first"
        )
    adapter_payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(adapter_payload["state_dict"])
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    adapter.eval()
    model = AnySatT11P(
        backbone, adapter, projector=None, dims=EMBED_DIM, blocks=0, freeze_adapter=True
    ).to(device)
    projector = build_projector(dims).to(device)
    writer.update_training(
        head_id=f"anysat_t11p_projector_{dims}",
        trainable_parameters=sum(p.numel() for p in projector.parameters()),
        total_parameters=sum(p.numel() for p in projector.parameters())
        + sum(p.numel() for p in adapter.parameters()),
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
                else "mean((E(adapter(time-mean F)) - teacher)^2) over observed tokens, "
                "no L2 normalisation"
                if loss_kind == "mse"
                else "mean(1 - cos(E(adapter(time-mean F)), teacher))"
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
            "distill_aggregation": "adaptive average 128px teacher -> 16x16 tokens",
            "distill_trains_adapter": "false (adapter comes from the raw probe)",
            "input_adapter_source": str(adapter_path),
            "input_adapter_sha256": _sha256(adapter_path),
            "frozen_features": "true",
        }
    )
    optimiser = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=1e-4)
    minimize = loss_kind in {"mse", "infonce"}
    best = (float("inf") if minimize else -1.0, 0, None)
    best_cosine = float("nan")
    best_mse = float("nan")
    best_infonce = float("nan")
    best_top1 = float("nan")
    started = time.time()

    # Both the adapter + backbone and the teacher are fixed, so encode and
    # teacher pooling happen once instead of every epoch (review 2026-11-12,
    # P3).
    train_features = _frozen_features(model, groups["train"], data_root, means, stds, device)
    val_features = _frozen_features(model, groups["validation"], data_root, means, stds, device)
    train_teacher = [teacher_tokens(target, row, data_root, tree_dims) for row in groups["train"]]
    val_teacher = [
        teacher_tokens(target, row, data_root, tree_dims) for row in groups["validation"]
    ]

    for epoch in range(epochs):
        model.eval()
        projector.train()
        order = _epoch_order(len(groups["train"]), request.seed, epoch)
        total, steps, pending, batches = 0.0, 0, 0, 0
        optimiser.zero_grad(set_to_none=True)
        if loss_kind == "infonce":
            patches = patches_per_step(loss_kind)
            for step, start in enumerate(range(0, len(order), patches)):
                chosen = order[start : start + patches]
                features = train_features[chosen].to(device).float()
                predictions = projector(features)
                targets = torch.stack([train_teacher[index][0] for index in chosen]).to(device)
                supports = torch.stack([train_teacher[index][1] for index in chosen]).to(device)
                generator = torch.Generator().manual_seed(
                    1_000_000 * request.seed + 1_000 * epoch + step
                )
                student, teacher = sample_token_pairs(
                    predictions, targets, supports, INFONCE_TOKENS, generator
                )
                loss = info_nce(student, teacher)
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                total += float(loss.detach())
                batches += 1
                steps += 1
                if limit and steps >= limit:
                    break
        else:
            for start in range(0, len(order), batch_size):
                chosen = order[start : start + batch_size]
                features = train_features[chosen].to(device).float()
                predictions = projector(features)
                targets = torch.stack([train_teacher[index][0] for index in chosen]).to(device)
                supports = torch.stack([train_teacher[index][1] for index in chosen]).to(device)
                if loss_kind == "mse":
                    loss = _masked_mse(predictions, targets, supports)
                else:
                    loss = 1.0 - _cosine(predictions, targets, supports)
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
        projector.eval()
        with torch.inference_mode():
            cosine_total, mse_total = 0.0, 0.0
            for start in range(0, val_features.shape[0], 1):
                predictions = projector(val_features[start : start + 1].to(device).float())
                grid, support = val_teacher[start]
                target_tensor = grid.unsqueeze(0).to(device)
                support_tensor = support.unsqueeze(0).to(device)
                cosine_total += float(_cosine(predictions, target_tensor, support_tensor))
                mse_total += float(_masked_mse(predictions, target_tensor, support_tensor))
            validation_cosine = cosine_total / max(val_features.shape[0], 1)
            validation_mse = mse_total / max(val_features.shape[0], 1)
            validation_infonce = float("nan")
            validation_top1 = float("nan")
            if loss_kind == "infonce":
                infonce_total = 0.0
                top1_correct, top1_total, chunks = 0, 0, 0
                patches = patches_per_step(loss_kind)
                for start in range(0, val_features.shape[0], patches):
                    predictions = projector(
                        val_features[start : start + patches].to(device).float()
                    )
                    target_tensor = torch.stack(
                        [
                            val_teacher[i][0]
                            for i in range(start, min(start + patches, len(val_teacher)))
                        ]
                    ).to(device)
                    support_tensor = torch.stack(
                        [
                            val_teacher[i][1]
                            for i in range(start, min(start + patches, len(val_teacher)))
                        ]
                    ).to(device)
                    generator = torch.Generator().manual_seed(10_000_000 + start)
                    student, teacher = sample_token_pairs(
                        predictions, target_tensor, support_tensor, INFONCE_TOKENS, generator
                    )
                    infonce_total += float(info_nce(student, teacher))
                    correct, total_tokens = retrieval_top1(student, teacher)
                    top1_correct += correct
                    top1_total += total_tokens
                    chunks += 1
                validation_infonce = infonce_total / max(chunks, 1)
                validation_top1 = top1_correct / max(top1_total, 1)
        mean_loss = total / max(batches, 1)
        writer.add_curve("distill_loss", "train", epoch + 1, mean_loss)
        writer.add_curve("cosine", "validation", epoch + 1, validation_cosine)
        if loss_kind == "infonce":
            writer.add_curve("infonce", "validation", epoch + 1, validation_infonce)
            writer.add_curve("retrieval_top1", "validation", epoch + 1, validation_top1)
        if loss_kind == "mse":
            writer.add_curve("mse", "validation", epoch + 1, validation_mse)
        emit(
            request.run_id,
            "distill.epoch",
            f"epoch {epoch + 1}/{epochs}: loss {mean_loss:.4f}, "
            f"val cosine {validation_cosine:.4f}, val MSE {validation_mse:.5f}",
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
        epochs=epochs,
        budget=budget,
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
        "adapter_description": "SpectralAdapter(6->10), frozen from the raw probe",
        "adapter_fingerprint": adapter_fingerprint,
        "teacher_used_at_inference": False,
        "distill_loss": loss_kind,
        "distill_selection": "val_mse" if loss_kind == "mse" else "val_cosine",
    }
    payload = {"state_dict": projector.state_dict(), "adapter": adapter.state_dict(), **meta}
    local = Path(request.runs_root) / request.run_id / "projector.pt"
    _save_atomic(local, payload)
    published = _projector_path(target, request.seed, fingerprint)
    _save_atomic(published, payload)
    writer.add_artifact("checkpoint", "projector.pt", "distilled E + shared 6->10 adapter")
    metrics = [
        _metric("val_cosine", best_cosine, "validation"),
        _metric("distill_train_loss", mean_loss, "train", direction="minimize"),
    ]
    baseline_tags = {}
    if loss_kind == "mse":
        metrics.append(_metric("val_mse", best_mse, "validation", direction="minimize"))
    if loss_kind == "infonce":
        metrics.append(_metric("val_infonce", best_infonce, "validation", direction="minimize"))
        metrics.append(_metric("val_retrieval_top1", best_top1, "validation"))
    if loss_kind in {"mse", "infonce"}:
        baseline_cosine, baseline_mse = _mean_teacher_baseline(
            train_teacher, val_teacher, dims, device
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
            "artifact_scope": "diagnostic" if tags.get("smoke") == "true" else "formal",
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
        f"T11P AnySat distill {target}: {dims} dims, 16x16 tokens, {loss_label} loss, "
        f"best {selection_label} {best[0]:.6f} at epoch {best[1]}"
    )
    if tags.get("smoke") == "true":
        summary += " [SMOKE]"
    writer.complete(summary)
    emit(request.run_id, "distill.done", "projector saved; shared adapter reused")
    return 0


def _train_b_stage(
    request,
    writer,
    emit,
    protocol_id,
    groups,
    data_root,
    device,
    means,
    stds,
    epochs,
    lr,
    backbone_lr,
    batch_size,
    accumulation,
    workers,
    limit,
    tags,
    backbone,
) -> int:
    target = B_STAGE_PROTOCOLS[protocol_id]
    blocks, head_dims, use_head_projector = B_STAGE_VARIANTS[protocol_id]
    budget = _budget_signature(epochs, len(groups["train"]), limit, tags.get("smoke") == "true")
    unfrozen = _unfreeze_last_blocks(backbone, blocks)
    if unfrozen == 0:
        raise AnySatT11PFailure("B stage unfroze zero backbone parameters")
    adapter = SpectralAdapter(BANDS).to(device)
    adapter_fingerprint = _fingerprint(
        writer.record.model_checkpoint_sha256 or "",
        writer.record.dataset.manifest_sha256,
        "raw",
        request.seed,
        init=tags["init"],
        normalization=tags["normalization_digest"],
        epochs=epochs,
        budget=budget,
    )
    adapter_path = _adapter_path(request.seed, adapter_fingerprint)
    if not adapter_path.is_file():
        raise AnySatT11PFailure(
            f"no label-supervised 6->10 adapter for seed {request.seed} at "
            f"{adapter_path}; run anysat_t11p_probe_raw first"
        )
    adapter_payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    adapter.load_state_dict(adapter_payload["state_dict"])

    # Build the model (and therefore the classification head) before any
    # optional projector: initialising a teacher projector in one arm must not
    # change the other arm's head init (review 2026-11-12, P1).
    model = AnySatT11P(
        backbone,
        adapter,
        projector=None,
        dims=head_dims,
        blocks=blocks,
    ).to(device)
    head_projector = build_projector(B_STAGE_DIMS).to(device) if use_head_projector else None
    teacher_projector = None
    if target is not None:
        teacher_projector = (
            head_projector if use_head_projector else build_projector(B_STAGE_DIMS).to(device)
        )
    model.projector = head_projector
    model.needs_grad = model.adapter_trainable or blocks > 0

    backbone_parameters = [p for p in backbone.parameters() if p.requires_grad]
    head_parameters = list(adapter.parameters()) + list(model.head.parameters())
    seen = {id(parameter) for parameter in head_parameters}
    for module in (head_projector, teacher_projector):
        if module is None:
            continue
        for parameter in module.parameters():
            if id(parameter) not in seen:
                head_parameters.append(parameter)
                seen.add(id(parameter))
    total_parameters = sum(p.numel() for p in model.parameters())
    if teacher_projector is not None and teacher_projector is not head_projector:
        total_parameters += sum(p.numel() for p in teacher_projector.parameters())
    optimiser = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": head_parameters, "lr": lr},
        ],
        weight_decay=1e-4,
    )
    writer.update_training(
        head_id=(
            f"anysat_t11p_b_stage_last{blocks}_{head_dims}"
            f"{'' if use_head_projector else '_headraw'}"
        ),
        trainable_parameters=sum(p.numel() for p in backbone_parameters + head_parameters),
        total_parameters=total_parameters,
    )
    writer.update(
        tags={
            **tags,
            "b_stage": "backbone adaptation",
            "b_stage_blocks": str(blocks),
            "b_stage_blocks_total": str(B_STAGE_BLOCKS_TOTAL),
            "b_stage_depth_fraction": f"{blocks / B_STAGE_BLOCKS_TOTAL:.4f}",
            "b_stage_head_dims": str(head_dims),
            "b_stage_head_projector": str(use_head_projector),
            "b_stage_teacher_projector": str(
                teacher_projector is not None and teacher_projector is not head_projector
            ),
            "b_stage_trainable_backbone_parameters": str(unfrozen),
            "b_stage_backbone_lr": f"{backbone_lr:g}",
            "b_stage_head_lr": f"{lr:g}",
            "b_stage_teacher_weight": str(B_STAGE_TEACHER_WEIGHT if target else 0.0),
            "b_stage_distill_target": target or "none",
            "b_stage_batch": f"{batch_size} x {accumulation} accumulation",
            "input_adapter_source": str(adapter_path),
            "input_adapter_sha256": _sha256(adapter_path),
            "adapter_init": "label-supervised raw probe",
            "adapter_trainable": "true",
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
        batch_size=1,
        shuffle=False,
        num_workers=workers,
    )
    test_loader = DataLoader(
        PastisS2(data_root, groups["test"]),
        batch_size=1,
        shuffle=False,
        num_workers=workers,
    )
    started = time.time()
    for epoch in range(epochs):
        model.train()
        if teacher_projector is not None and teacher_projector is not head_projector:
            teacher_projector.train(
                any(parameter.requires_grad for parameter in teacher_projector.parameters())
            )
        order = _epoch_order(len(dataset), request.seed, epoch)
        total, steps, pending, batches = 0.0, 0, 0, 0
        optimiser.zero_grad(set_to_none=True)
        for start in range(0, len(order), batch_size):
            chosen = order[start : start + batch_size]
            chosen_rows = [groups["train"][index] for index in chosen]
            batch = _stack_items([dataset[index] for index in chosen], device)
            images = (batch["image"] - means) / stds
            labels = batch["label"]
            if int((labels != IGNORE_INDEX).sum()) == 0:
                continue
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits, readout = model.forward_with_projection(images, batch["dates"])
            loss = criterion(logits.float(), labels)
            if target is not None:
                projected = readout if use_head_projector else teacher_projector(readout)
                targets, supports = [], []
                for row in chosen_rows:
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
    if teacher_projector is not None and teacher_projector is not head_projector:
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
    trainable_state = {
        f"adapter.{name}": value.detach().cpu() for name, value in adapter.state_dict().items()
    }
    if head_projector is not None:
        trainable_state |= {
            f"head_projector.{name}": value.detach().cpu()
            for name, value in head_projector.state_dict().items()
        }
    if teacher_projector is not None and teacher_projector is not head_projector:
        trainable_state |= {
            f"teacher_projector.{name}": value.detach().cpu()
            for name, value in teacher_projector.state_dict().items()
        }
    trainable_state |= {
        f"head.{name}": value.detach().cpu() for name, value in model.head.state_dict().items()
    }
    for name, parameter in backbone.named_parameters():
        if parameter.requires_grad:
            trainable_state[f"backbone.{name}"] = parameter.detach().cpu()
    state = {
        "state_dict": trainable_state,
        "backbone_blocks_total": B_STAGE_BLOCKS_TOTAL,
        "trained_blocks": blocks,
        "dims": head_dims,
        "head_projector": use_head_projector,
        "teacher_projector": teacher_projector is not None
        and teacher_projector is not head_projector,
        "num_classes": NUM_CLASSES,
        "bands": BANDS,
        "frames": MAX_FRAMES,
        "grid": GRID,
        "normalization": {"mean": tags["normalization_mean"], "std": tags["normalization_std"]},
        "seed": request.seed,
        "protocol": protocol_id,
        "teacher": target or "none",
        "anysat_checkpoint_sha256": writer.record.model_checkpoint_sha256 or "",
        "adapter_fingerprint": adapter_fingerprint,
        "adapter_source": str(adapter_path),
        "pretraining_overlap": "AnySat pretraining corpus includes PASTIS-HD",
    }
    _save_atomic(Path(request.runs_root) / request.run_id / "model.pt", state)
    writer.add_artifact("checkpoint", "model.pt", "AnySat T11P B-stage adapted blocks, E and head")
    writer.update_training(epochs_completed=epochs, wall_time_seconds=round(wall, 1))
    readout = "128-d E + head" if use_head_projector else f"{head_dims}-d wide head"
    summary = (
        f"T11P AnySat {target or 'labels-only'}: test mIoU {test['mean_iou']:.4f}, "
        f"last {blocks}/{B_STAGE_BLOCKS_TOTAL} blocks + {readout}, {epochs} epochs"
    )
    if tags.get("smoke") == "true":
        summary += " [SMOKE]"
    elif epochs != OFFICIAL_EPOCHS:
        summary += " [REDUCED-BUDGET]"
    writer.complete(summary)
    emit(request.run_id, "b_stage.metrics", summary)
    return 0


def train_t11p(
    request: WorkerRequest,
    writer: RunWriter,
    emit,
    *,
    checkpoint: Path,
    protocol_id: str,
    task_spec,
) -> int:
    """Entry point used by the worker runner; wraps failures into failed runs."""
    return _train_t11p(
        request,
        writer,
        emit,
        checkpoint=checkpoint,
        protocol_id=protocol_id,
        task_spec=task_spec,
    )
