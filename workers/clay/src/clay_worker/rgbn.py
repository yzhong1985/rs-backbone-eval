"""Clay v1.5 raw-S2 student: PASTIS-HD distillation and probes (T11D/T11P).

T11D is the four-band RGB+NIR input (20 frames); T11P is the six-band matched
pair with Prithvi (4 frames). Both use ``BAND_PROFILES`` below.

Scope (handoff 2026-11-11, P0/P1):

* student input = S2 B02/B03/B04/B08 at 10 m, **2019 acquisition dates only**,
  at most 20 frames evenly spaced by date (the archive also carries Sep-Dec
  2018, which this track does not use);
* teacher = AEF 2019 (64 dims) or OlmoEarth v1.3 2019 (64 or 128 dims), both on
  the same 128x128 grid as the S2 patch, decoded with
  ``sign(v)*(|v|/127.5)^2`` and aggregated onto Clay's stride-8 token support
  (8x8 pixels = 80 m) before the cosine loss. Aggregation is a mean over the
  pixels the product actually observes; the token grid is never treated as a
  new 80 m observation of the surface;
* Clay's ``SegmentEncoder`` is frozen. The trainable module ``E`` is a per-token
  MLP (1024 -> hidden -> 64/128), and the downstream probe trains the same
  1x1-convolution MLP head on frozen F or F+E. Token logits are bilinearly
  upsampled to the 128x128 label grid for the loss and the metric; the 80 m
  token grid is a readout support, not a claim of 10 m observation;
* the teacher is used at training time only. The probe and any future transfer
  run read the S2 frames and the frozen student, never AEF/OlmoEarth.

The stride-8 token cache is fingerprinted by checkpoint, task/bands, frame
rule, grid, preprocessing, manifest and code version; a cache that does not
match is recomputed rather than reused (FBP review 2026-11-11, P1).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rs_backbone_eval.contracts import MetricRecord

TASK_ID = "T11D"


class BandProfile:
    """One registered S2 input budget for this student.

    T11D is the four-band Clay-native subset at 20 frames; T11P is the
    six-band Prithvi-native subset at 4 frames, so the two models share an
    input budget in the matched pair. PASTIS-HD S2 order is
    B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 (README + the documentation PDF).
    """

    __slots__ = ("bands", "clay_bands", "s2_indices", "max_frames", "preprocessing", "description")

    def __init__(
        self,
        bands: tuple[str, ...],
        clay_bands: tuple[str, ...],
        s2_indices: tuple[int, ...],
        max_frames: int,
        preprocessing: str,
        description: str,
    ) -> None:
        self.bands = bands
        self.clay_bands = clay_bands
        self.s2_indices = s2_indices
        self.max_frames = max_frames
        self.preprocessing = preprocessing
        self.description = description


BAND_PROFILES = {
    "T11D": BandProfile(
        bands=("B02", "B03", "B04", "B08"),
        clay_bands=("blue", "green", "red", "nir"),
        s2_indices=(0, 1, 2, 6),
        max_frames=20,
        preprocessing="clay_sentinel-2-l2a_stats_4band_raw_dn",
        description="S2 B02/B03/B04/B08 raw DN, 2019 dates only, <=20 frames",
    ),
    "T11P": BandProfile(
        bands=("B02", "B03", "B04", "B8A", "B11", "B12"),
        clay_bands=("blue", "green", "red", "nir08", "swir16", "swir22"),
        s2_indices=(0, 1, 2, 7, 8, 9),
        max_frames=4,
        preprocessing="clay_sentinel-2-l2a_stats_6band_raw_dn",
        description="S2 B02/B03/B04/B8A/B11/B12 raw DN, 2019 dates only, 4 frames",
    ),
}
BAND_NAMES = list(BAND_PROFILES[TASK_ID].bands)
CLAY_BANDS = list(BAND_PROFILES[TASK_ID].clay_bands)
S2_INDICES = list(BAND_PROFILES[TASK_ID].s2_indices)
GRID = 128
PATCH = 8
TOKENS = GRID // PATCH
GSD_M = 10.0
MAX_FRAMES = BAND_PROFILES[TASK_ID].max_frames
VOID_CLASS = 19
CACHE_VERSION = "2026-11-11a.rgbn-token-cache"
# protocol_id -> (target name, teacher tree, teacher dims)
DISTILL_PROTOCOLS = {
    "clay_rgbn_distill_a64": ("aef64", "aef-2019", 64),
    "clay_rgbn_distill_o64": ("oe64", "olmoearth-2019", 64),
    "clay_rgbn_distill_o128": ("oe128", "olmoearth-2019-128", 128),
    # Six-band matched-pair track (T11P); the output path is keyed by the
    # task-dependent cache fingerprint, so the two tracks cannot collide.
    "clay_s2six_distill_o128": ("oe128", "olmoearth-2019-128", 128),
    # MSE-vs-cosine control (T11D, diagnostic): separate target names so the
    # published projectors never overwrite the cosine group's files.
    "clay_rgbn_distill_o64_mse": ("oe64mse", "olmoearth-2019", 64),
    "clay_rgbn_distill_a64_mse": ("aef64mse", "aef-2019", 64),
    # Six-band matched-pair MSE control (T11P, OE128).
    "clay_s2six_distill_o128_mse": ("oe128mse", "olmoearth-2019-128", 128),
    # Contrastive (symmetric InfoNCE) control (T11D).
    "clay_rgbn_distill_o64_nce": ("oe64nce", "olmoearth-2019", 64),
    "clay_rgbn_distill_a64_nce": ("aef64nce", "aef-2019", 64),
    # Six-band replication of the positive T11D InfoNCE result.
    "clay_s2six_distill_o128_nce": ("oe128nce", "olmoearth-2019-128", 128),
}
# probe protocol -> how the readout is built
#   raw    : temporal-mean F, no projection
#   pca64  : train-only PCA of F tokens to 64 dims
#   s64/128: E trained jointly with the head on labels (task adaptation)
#   a64/o64/o128: frozen E distilled from that teacher
PROBE_PROTOCOLS = {
    "clay_rgbn_probe_raw": ("raw", None),
    "clay_rgbn_probe_pca64": ("pca64", None),
    "clay_rgbn_probe_s64": ("supervised", 64),
    "clay_rgbn_probe_s128": ("supervised", 128),
    "clay_rgbn_probe_a64": ("distilled", "aef64"),
    "clay_rgbn_probe_o64": ("distilled", "oe64"),
    "clay_rgbn_probe_o128": ("distilled", "oe128"),
    # T11P: same three readouts as the four-band track, on the six-band cache.
    "clay_s2six_probe_raw": ("raw", None),
    "clay_s2six_probe_s128": ("supervised", 128),
    "clay_s2six_probe_o128": ("distilled", "oe128"),
    "clay_rgbn_probe_o64_mse": ("distilled", "oe64mse"),
    "clay_rgbn_probe_a64_mse": ("distilled", "aef64mse"),
    "clay_rgbn_probe_o64_nce": ("distilled", "oe64nce"),
    "clay_rgbn_probe_a64_nce": ("distilled", "aef64nce"),
    "clay_s2six_probe_o128_nce": ("distilled", "oe128nce"),
    "clay_s2six_probe_o128_mse": ("distilled", "oe128mse"),
    "clay_s2six_probe_pca128": ("pca128", None),
    # 20-epoch frozen-readout budget control (diagnostic).
    "clay_rgbn_probe_raw_e20": ("raw", None),
    "clay_rgbn_probe_s64_e20": ("supervised", 64),
    "clay_rgbn_probe_o64_e20": ("distilled", "oe64"),
}
TARGET_TREE_DIMS = {name: (tree, dims) for name, tree, dims in DISTILL_PROTOCOLS.values()}
# Protocols that train with masked MSE instead of the cosine loss. Everything
# not listed here keeps the original cosine loss and epoch-selection rule.
DISTILL_LOSS = {
    "clay_rgbn_distill_o64_mse": "mse",
    "clay_rgbn_distill_a64_mse": "mse",
    "clay_s2six_distill_o128_mse": "mse",
    "clay_rgbn_distill_o64_nce": "infonce",
    "clay_rgbn_distill_a64_nce": "infonce",
    "clay_s2six_distill_o128_nce": "infonce",
}
INFONCE_TEMPERATURE = 0.07
INFONCE_TOKENS = 1024
INFONCE_PATCHES = 8

# B-stage (backbone adaptation): identical trainable modules in both arms — the
# last N transformer blocks plus the final norm, one shared projector E(->128)
# and one head. The only difference is the auxiliary teacher cosine on E.
# Declared here, not in an environment variable.
B_STAGE_PROTOCOLS = {
    "clay_rgbn_ft_last4": None,
    "clay_rgbn_ft_last4_distill": "oe128",
    # Locked-recipe formal variants: same modules, same budget, primary panel.
    "clay_rgbn_ft_last4_v1": None,
    "clay_rgbn_ft_last4_distill_v1": "oe128",
    # Six-band matched pair (T11P): same modules and budget as the v1 pair.
    "clay_s2six_ft_last4": None,
    "clay_s2six_ft_last4_distill": "oe128",
    # Wide-head T11P arm: the 128-d projector is removed from the label path,
    # following the three-seed T11D result (+0.030 test / +0.025 validation).
    # No teacher counterpart yet: the teacher loss in this worker reads the
    # projector, so a fair wide-head teacher arm needs a separate 128-d
    # teacher projector (follow-up).
    "clay_s2six_ft_last4_raw1024": None,
    "clay_s2six_ft_last4_raw1024_distill": "oe128",
    # Adaptation-depth ablations (diagnostic, seed 17 only): does more
    # unfrozen capacity help, and does the 128-d projector cost anything?
    "clay_rgbn_ft_last8": None,
    "clay_rgbn_ft_last8_distill": "oe128",
    "clay_rgbn_ft_last4_raw1024": None,
    # Budget-sensitivity controls (diagnostic): 80 epoch 10% label arm and its
    # teacher counterpart, same 128-d readout as the original 10% experiment.
    "clay_rgbn_ft_last4_e80": None,
    "clay_rgbn_ft_last4_distill_e80": "oe128",
}
# protocol -> (unfrozen blocks, readout width, projector?)
B_STAGE_VARIANTS = {
    "clay_rgbn_ft_last4": (4, 128, True),
    "clay_rgbn_ft_last4_distill": (4, 128, True),
    "clay_rgbn_ft_last4_v1": (4, 128, True),
    "clay_rgbn_ft_last4_distill_v1": (4, 128, True),
    "clay_s2six_ft_last4": (4, 128, True),
    "clay_s2six_ft_last4_distill": (4, 128, True),
    "clay_s2six_ft_last4_raw1024": (4, 1024, False),
    "clay_s2six_ft_last4_raw1024_distill": (4, 1024, False),
    "clay_rgbn_ft_last8": (8, 128, True),
    "clay_rgbn_ft_last8_distill": (8, 128, True),
    "clay_rgbn_ft_last4_raw1024": (4, 1024, False),
    "clay_rgbn_ft_last4_e80": (4, 128, True),
    "clay_rgbn_ft_last4_distill_e80": (4, 128, True),
}
B_STAGE_BLOCKS = 4
B_STAGE_TEACHER_WEIGHT = 1.0
# Wide-head arms use a separate training function so the admitted 128-d pair
# keeps its original RNG/accumulation path.
WIDE_B_STAGE_PROTOCOLS = {
    "clay_s2six_ft_last4_raw1024": None,
    "clay_s2six_ft_last4_raw1024_distill": "oe128",
}
B_STAGE_FRAME_CHUNK = 5
B_STAGE_BACKBONE_LR = 1e-4
B_STAGE_DIMS = 128
B_STAGE_ACCUMULATION = 2


def accumulation_chunks(order: list[int], accumulation: int):
    """Split a sample order into optimizer-step groups (last group may be short)."""
    for start in range(0, len(order), accumulation):
        yield order[start : start + accumulation]


class RgbnFailure(Exception):
    """Gates convert to a failed run with a readable reason."""


# --------------------------------------------------------------------------
# Pure helpers (no torch, exercised by CPU-only tests)
# --------------------------------------------------------------------------


def profile_for(task_id: str) -> BandProfile:
    try:
        return BAND_PROFILES[task_id]
    except KeyError as exc:
        raise RgbnFailure(
            f"rgbn module implements {sorted(BAND_PROFILES)}, got {task_id!r}"
        ) from exc


def select_frames(dates: Sequence[int], max_frames: int = MAX_FRAMES) -> list[int]:
    """2019-only frame indices (shared with the manifest builder, one rule)."""
    from rs_backbone_eval.data.datasets.pastis_hd import select_2019_frames

    try:
        return select_2019_frames([int(date) for date in dates], max_frames)
    except ValueError as exc:  # the builder raises its own error type
        raise RgbnFailure(str(exc)) from exc


def decode_teacher(raw: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Raw int8 embedding -> declared value encoding, invalid pixels zeroed.

    Both products declare ``sign(v)*(|v|/127.5)^2``; AEF stores the int8 value
    as float16 and OlmoEarth stores it as int8, so the arithmetic is identical.
    """
    features = np.asarray(raw, dtype=np.float32)
    features = np.sign(features) * (features / 127.5) ** 2
    mask = np.asarray(valid, dtype=bool)
    if mask.shape != features.shape[1:]:
        raise RgbnFailure(f"teacher valid mask {mask.shape} != grid {features.shape[1:]}")
    features[:, ~mask] = 0.0
    return features


def aggregate_tokens(features: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean over each stride-8 block's observed pixels -> (D,16,16) and support."""
    channels = features.shape[0]
    values = features.reshape(channels, TOKENS, PATCH, TOKENS, PATCH)
    mask = np.asarray(valid, dtype=bool).reshape(1, TOKENS, PATCH, TOKENS, PATCH)
    total = (values * mask).sum(axis=(2, 4), dtype=np.float64)
    count = mask.sum(axis=(2, 4))[0]  # drop the channel axis -> (16,16)
    return (total / np.maximum(count, 1.0)).astype(np.float32), count.astype(np.int32)


def majority_labels(labels: np.ndarray, ignore: int = VOID_CLASS) -> np.ndarray:
    """(128,128) labels -> (16,16) majority class per stride-8 block.

    A block whose pixels are all void keeps the ignore value; ties fall back to
    the smallest class id so the aggregation is deterministic.
    """
    blocks = labels.reshape(TOKENS, PATCH, TOKENS, PATCH).transpose(0, 2, 1, 3)
    flat = blocks.reshape(TOKENS, TOKENS, PATCH * PATCH)
    result = np.full((TOKENS, TOKENS), ignore, dtype=np.int64)
    pure = 0
    for row in range(TOKENS):
        for col in range(TOKENS):
            values = flat[row, col]
            values = values[values != ignore]
            if values.size == 0:
                continue
            counts = np.bincount(values, minlength=int(values.max()) + 1)
            result[row, col] = int(np.argmax(counts))
            pure += int(counts.max() == values.size)
    return result, pure / (TOKENS * TOKENS)


def cache_fingerprint(
    *,
    checkpoint_sha256: str,
    manifest_sha256: str,
    preprocessing: str,
    code_version: str,
    task_id: str = TASK_ID,
    bands: Sequence[str] = tuple(BAND_NAMES),
    frame_rule: str = "2019-only-linspace",
    max_frames: int = MAX_FRAMES,
    grid: int = GRID,
    patch_size: int = PATCH,
    init: str = "pretrained",
) -> str:
    payload = {
        "version": CACHE_VERSION,
        "checkpoint_sha256": checkpoint_sha256,
        "task_id": task_id,
        "bands": list(bands),
        "frame_rule": frame_rule,
        "max_frames": max_frames,
        "grid": grid,
        "patch_size": patch_size,
        "preprocessing": preprocessing,
        "manifest_sha256": manifest_sha256,
        "code_version": code_version,
    }
    # Keep the pretrained fingerprint unchanged so the deployed caches stay
    # valid; a random-init run gets a distinct namespace (review 2026-11-12,
    # P1). Clay's per-band normalisation is the fixed sensor metadata, so it
    # does not enter the key.
    if init != "pretrained":
        payload["init"] = init
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def code_version() -> str:
    """Version of this extraction rule (not the git commit; see run.json)."""
    return CACHE_VERSION


def cache_root() -> Path:
    """Feature cache root; ``RS_EVAL_RGBN_CACHE`` wins, then the shared cache."""
    explicit = os.environ.get("RS_EVAL_RGBN_CACHE")
    if explicit:
        return Path(explicit)
    shared = os.environ.get("RS_EVAL_CACHE_ROOT")
    base = Path(shared) if shared else Path.home() / "rs-eval-cache"
    return base / "pastis-rgbn"


def save_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    temporary.replace(path)


def cache_entry_current(
    meta_path: Path, expected: dict[str, Any], *, max_frames: int = MAX_FRAMES
) -> bool:
    """Whether a per-patch cache entry matches the fingerprint and frame list.

    The heavy fingerprint is encoded in the directory name; the per-patch meta
    repeats it so a moved/renamed entry cannot be reused, and the ``.npy``
    header is opened so a truncated write is treated as missing (FBP review
    2026-11-11, P1).
    """
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("version") != expected.get("version", CACHE_VERSION):
        return False
    if expected.get("fingerprint") and meta.get("fingerprint") != expected["fingerprint"]:
        return False
    frames = meta.get("frames") or []
    if not 1 <= len(frames) <= max_frames:
        return False
    if len(meta.get("dates") or []) != len(frames):
        return False
    features_path = meta_path.with_suffix(".npy")
    if not features_path.is_file():
        return False
    try:
        array = np.load(features_path, mmap_mode="r")
    except (OSError, ValueError):
        return False
    expected_shape = (len(frames), 1024, TOKENS, TOKENS)
    return tuple(array.shape) == expected_shape and array.dtype == np.float16


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def read_rows(manifest_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_frames(
    row: dict, data_root: Path, profile: BandProfile | None = None
) -> tuple[np.ndarray, list[int], list[int]]:
    """(T,C,128,128) raw DN for the selected 2019 frames, plus dates/indices."""
    profile = profile or BAND_PROFILES[TASK_ID]
    aux = row.get("aux_paths") or {}
    s2_rel = aux.get("s2_path")
    if not s2_rel:
        raise RgbnFailure(f"sample {row['sample_id']}: aux_paths has no s2_path")
    # T11D names the AEF sidecar explicitly: image_paths[0] is the S2 stack, so
    # the acquisition dates do not live next to the primary input.
    dates_source = aux.get("dates_source") or str(Path(row["image_paths"][0]).with_suffix(".json"))
    sidecar = data_root / dates_source
    if not sidecar.is_file():
        raise RgbnFailure(f"sample {row['sample_id']}: sidecar {sidecar.name} missing")
    dates = json.loads(sidecar.read_text(encoding="utf-8")).get("dates_s2") or []
    stack = np.load(data_root / s2_rel, mmap_mode="r")
    if stack.shape[0] != len(dates):
        raise RgbnFailure(
            f"sample {row['sample_id']}: S2 stack T={stack.shape[0]} vs sidecar {len(dates)}"
        )
    frames = select_frames(dates, profile.max_frames)
    pixels = np.asarray(stack[frames][:, profile.s2_indices], dtype=np.float32)
    return pixels, [int(dates[index]) for index in frames], frames


def teacher_tokens(target: str, row: dict, data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    """Aggregated teacher on the token grid and the per-token support mask."""
    if target not in TARGET_TREE_DIMS:
        raise RgbnFailure(f"unknown teacher target {target!r}")
    tree, dims = TARGET_TREE_DIMS[target]
    sample_id = row["sample_id"]
    path = data_root / "derived" / tree / "features" / "2019" / f"{sample_id}.npz"
    if not path.is_file():
        raise RgbnFailure(f"sample {sample_id}: teacher payload {path} missing")
    with np.load(path) as archive:
        raw = archive["features"]
        valid = archive["valid"] if "valid" in archive.files else np.ones(raw.shape[1:], bool)
    if raw.shape[0] != dims:
        raise RgbnFailure(
            f"sample {sample_id}: teacher {target} has {raw.shape[0]} dims, expected {dims}"
        )
    grid, support = aggregate_tokens(decode_teacher(raw, valid), valid)
    return grid, support > 0


# --------------------------------------------------------------------------
# Model pieces (torch imported lazily so the CPU tests can import the module)
# --------------------------------------------------------------------------


def build_encoder(checkpoint: Path):
    from claymodel.finetune.segment.factory import SegmentEncoder  # upstream checkout

    from .t1 import CLAY_LARGE

    return SegmentEncoder(**CLAY_LARGE, ckpt_path=str(checkpoint))


def frame_tokens(encoder, pixels, waves, *, grad: bool = False) -> Any:
    """(B,4,H,W) raw DN -> (B,1024,16,16) stride-8 tokens with no gradients."""
    import torch

    from .t1 import CLAY_LARGE

    batch, _, height, width = pixels.shape
    datacube = {
        "pixels": pixels,
        "time": torch.zeros(batch, 4, device=pixels.device),
        "latlon": torch.zeros(batch, 4, device=pixels.device),
        "gsd": torch.tensor(GSD_M, device=pixels.device),
        "waves": waves,
        "platform": "sentinel-2-l2a",
    }
    if grad:
        tokens = encoder(datacube)
    else:
        with torch.no_grad():
            tokens = encoder(datacube)
    patch = CLAY_LARGE["patch_size"]
    return tokens.transpose(1, 2).reshape(batch, CLAY_LARGE["dim"], height // patch, width // patch)


def build_projector(in_channels: int, dims: int, hidden: int = 256):
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 1),
        nn.GELU(),
        nn.Conv2d(hidden, dims, 1),
    )


def build_head(in_channels: int, classes: int, hidden: int = 128):
    from torch import nn

    return nn.Sequential(
        nn.Conv2d(in_channels, hidden, 1),
        nn.GroupNorm(8, hidden),
        nn.GELU(),
        nn.Conv2d(hidden, classes, 1),
    )


def cosine_alignment(prediction, target, mask):
    import torch.nn.functional as F

    unit_prediction = F.normalize(prediction, dim=1)
    unit_target = F.normalize(target, dim=1)
    cosine = (unit_prediction * unit_target).sum(dim=1)
    if mask is not None:
        cosine = cosine[mask]
    return cosine.mean()


# --------------------------------------------------------------------------
# Feature cache
# --------------------------------------------------------------------------


class FeatureCache:
    """Stride-8 Clay tokens per patch, fingerprinted and written atomically."""

    def __init__(
        self,
        root: Path,
        fingerprint: str,
        encoder,
        waves,
        mean,
        std,
        *,
        device: str,
        profile: BandProfile | None = None,
    ) -> None:
        self.profile = profile or BAND_PROFILES[TASK_ID]
        self.directory = root / "features" / fingerprint
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.encoder = encoder
        self.waves = waves
        self.mean = mean
        self.std = std
        self.device = device
        self.hits = 0
        self.misses = 0
        self.seconds = 0.0

    def _paths(self, sample_id: str) -> tuple[Path, Path]:
        return (
            self.directory / f"{sample_id}.npy",
            self.directory / f"{sample_id}.json",
        )

    def get(self, row: dict, data_root: Path) -> np.ndarray:
        features_path, meta_path = self._paths(row["sample_id"])
        expected = {"version": CACHE_VERSION, "fingerprint": self.fingerprint}
        if cache_entry_current(meta_path, expected, max_frames=self.profile.max_frames):
            self.hits += 1
            return np.load(features_path, mmap_mode="r")
        self.misses += 1
        pixels, dates, frames = read_frames(row, data_root, self.profile)
        import torch

        started = time.time()
        channels = len(self.profile.bands)
        mean = torch.as_tensor(self.mean, device=self.device).view(1, channels, 1, 1)
        std = torch.as_tensor(self.std, device=self.device).view(1, channels, 1, 1)
        batch = torch.from_numpy(pixels).to(self.device)
        batch = (batch - mean) / std
        tokens = frame_tokens(self.encoder, batch, self.waves).to(torch.float16).cpu().numpy()
        self.seconds += time.time() - started
        save_npy_atomic(features_path, tokens)
        meta = {
            **expected,
            "version": CACHE_VERSION,
            "frames": frames,
            "dates": dates,
            "shape": list(tokens.shape),
            "dtype": "float16",
        }
        save_atomic(meta_path, json.dumps(meta, sort_keys=True).encode("utf-8"))
        return np.load(features_path, mmap_mode="r")


# --------------------------------------------------------------------------
# Run actions
# --------------------------------------------------------------------------


def _paths_and_specs(request):
    from rs_backbone_eval.registry import ProtocolSpec, TaskSpec, load_yaml_spec

    task_spec: TaskSpec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
    protocol_spec: ProtocolSpec = load_yaml_spec(Path(request.protocol_config_path), ProtocolSpec)
    return task_spec, protocol_spec


def _load_encoder(checkpoint: Path, device: str, profile: BandProfile):
    from .clay_probe import add_upstream_to_path, load_sensor_metadata
    from .t1 import CLAY_LARGE  # noqa: F401 - imported for the side-effect of validation

    metadata = load_sensor_metadata(add_upstream_to_path(), "sentinel-2-l2a")
    encoder = build_encoder(checkpoint).to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    means = np.asarray([metadata["bands"]["mean"][band] for band in profile.clay_bands], np.float32)
    stds = np.asarray([metadata["bands"]["std"][band] for band in profile.clay_bands], np.float32)
    waves = np.asarray(
        [metadata["bands"]["wavelength"][band] for band in profile.clay_bands], np.float32
    )
    import torch

    return encoder, torch.from_numpy(means), torch.from_numpy(stds), torch.from_numpy(waves)


def _prepare(request, writer, checkpoint: Path, emit):
    """Shared setup: specs, device, encoder, fingerprint and the feature cache."""
    import torch

    task_spec, protocol_spec = _paths_and_specs(request)
    profile = profile_for(task_spec.task_id)
    if list(task_spec.bands) != list(profile.bands):
        raise RgbnFailure(
            f"task {task_spec.task_id} declares {list(task_spec.bands)}, "
            f"rgbn profile has {list(profile.bands)}"
        )
    if protocol_spec.protocol_id not in (DISTILL_PROTOCOLS | PROBE_PROTOCOLS | B_STAGE_PROTOCOLS):
        raise RgbnFailure(f"rgbn module has no protocol {protocol_spec.protocol_id!r}")
    manifest_path = Path(request.manifest_path)
    rows = read_rows(manifest_path)
    from rs_backbone_eval.data.manifest import audit_task_manifest, manifest_sha256

    audit = audit_task_manifest(manifest_path, task_spec)
    strict_errors = audit.error_count - len(audit.leakage)
    if strict_errors:
        raise RgbnFailure(f"manifest audit failed: {audit.all_errors()[:5]}")
    manifest_digest = manifest_sha256(manifest_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, mean, std, waves = _load_encoder(checkpoint, device, profile)
    from rs_backbone_eval.repro import seed_everything

    seed_everything(request.seed)
    from clay_worker.clay_probe import add_upstream_to_path, upstream_commit
    from rs_backbone_eval.repro import random_init_requested

    fingerprint = cache_fingerprint(
        checkpoint_sha256=writer.record.model_checkpoint_sha256 or "",
        manifest_sha256=manifest_digest,
        preprocessing=profile.preprocessing,
        code_version=f"{CACHE_VERSION}:{upstream_commit(add_upstream_to_path())[:12]}",
        task_id=task_spec.task_id,
        bands=profile.bands,
        max_frames=profile.max_frames,
        init="random" if random_init_requested() else "pretrained",
    )
    cache = FeatureCache(
        cache_root(),
        fingerprint,
        encoder,
        waves.to(device),
        mean,
        std,
        device=device,
        profile=profile,
    )
    writer.update_input(
        preprocessing_id=profile.preprocessing,
        bands=list(profile.bands),
        time_steps=profile.max_frames,
        gsd_m=GSD_M,
    )
    writer.update(
        tags={
            "student_input": profile.description,
            "frame_rule": "2019-only-linspace",
            "feature_fingerprint": fingerprint,
            "feature_cache": str(cache.directory),
            "token_grid": f"{TOKENS}x{TOKENS}",
            "token_stride_m": str(PATCH * GSD_M),
            **(
                {"smoke": "true", "panel": "diagnostic"}
                if (
                    os.environ.get("CLAY_WORKER_EVAL_LIMIT")
                    or os.environ.get("CLAY_WORKER_TRAIN_LIMIT")
                    or os.environ.get("CLAY_WORKER_ALLOW_PARTIAL_EVAL") == "1"
                )
                else {}
            ),
            "teacher_aggregation": "mean over observed pixels of each 8x8 block",
            "device": device,
            "host": (
                torch.cuda.get_device_name(0).lower().replace(" ", "-")
                if device == "cuda"
                else "cpu"
            ),
            "torch": torch.__version__,
            "encoder_parameters": str(sum(p.numel() for p in encoder.parameters())),
            "encoder_trainable": str(
                sum(p.numel() for p in encoder.parameters() if p.requires_grad)
            ),
        }
    )
    emit(request.run_id, "rgbn.setup", f"{len(rows)} rows, fingerprint {fingerprint}")
    return task_spec, protocol_spec, rows, manifest_digest, device, encoder, mean, std, waves, cache


def probe_action(request, writer, emit) -> int:
    """P0 smoke: one real raw-S2 forward, frozen-gradient and cost report."""
    import torch

    checkpoint = _checkpoint(request, writer)
    task_spec, protocol_spec, rows, manifest_digest, device, encoder, mean, std, waves, cache = (
        _prepare(request, writer, checkpoint, emit)
    )
    start_row = next(row for row in rows if row["split"] == "train")
    profile = profile_for(task_spec.task_id)
    pixels, dates, frames = read_frames(start_row, Path(request.data_root), profile)
    writer.update(
        tags={
            "probe_sample": start_row["sample_id"],
            "probe_frames": str(len(frames)),
            "probe_dates": ",".join(str(date) for date in dates),
        }
    )
    batch = torch.from_numpy(pixels).to(device)
    channels = len(profile.bands)
    batch = (batch - mean.to(device).view(1, channels, 1, 1)) / std.to(device).view(
        1, channels, 1, 1
    )
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    started = time.time()
    tokens = frame_tokens(encoder, batch, waves.to(device))
    if device == "cuda":
        torch.cuda.synchronize()
    latency = time.time() - started
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
    finite = bool(torch.isfinite(tokens).all())
    trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    # Frozen check: a backward through the tokens must leave every encoder
    # parameter without a gradient. The tokens themselves are detached and
    # marked as requiring grad, otherwise the graph has no leaf that needs a
    # gradient and ``backward`` raises instead of proving the freeze.
    tokens.detach().requires_grad_(True).sum().backward()
    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    report = {
        "task": task_spec.task_id,
        "protocol": protocol_spec.protocol_id,
        "sample": start_row["sample_id"],
        "device": device,
        "input_shape": list(batch.shape),
        "frames": len(frames),
        "dates": dates,
        "token_shape": list(tokens.shape),
        "token_grid": [TOKENS, TOKENS],
        "stride_m": PATCH * GSD_M,
        "latency_s": round(latency, 3),
        "latency_per_frame_s": round(latency / max(len(frames), 1), 4),
        "peak_gpu_mb": round(peak_mb, 1),
        "finite": finite,
        "encoder_trainable_parameters": trainable,
        "encoder_grad_tensors": len(grads),
        "feature_fingerprint": cache.fingerprint,
        "manifest_sha256": manifest_digest,
    }
    (Path(request.runs_root) / request.run_id / "probe_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    writer.add_artifact("report", "probe_report.json", "RGB+NIR Clay probe report")
    writer.update(
        tags={
            "token_shape": "x".join(str(dim) for dim in tokens.shape),
            "latency_s": f"{latency:.3f}",
            "peak_gpu_mb": f"{peak_mb:.0f}",
        }
    )
    if not finite:
        raise RgbnFailure("non-finite tokens in the raw-S2 forward")
    if trainable or grads:
        raise RgbnFailure(
            f"encoder is not frozen (trainable={trainable}, grad tensors={len(grads)})"
        )
    writer.complete(
        f"{len(profile.bands)}-band Clay forward on {start_row['sample_id']}: "
        f"{len(frames)} frames -> tokens {tuple(tokens.shape)}, {latency:.2f}s, "
        f"peak {peak_mb:.0f} MB, frozen"
    )
    emit(request.run_id, "rgbn.probe", "probe passed", progress=1.0)
    return 0


def _checkpoint(request, writer) -> Path:
    from rs_backbone_eval.registry import ModelSpec, load_yaml_spec

    model_spec: ModelSpec = load_yaml_spec(Path(request.model_config_path), ModelSpec)
    checkpoint = os.environ.get(model_spec.checkpoint_env)
    if not checkpoint or not Path(checkpoint).is_file():
        raise RgbnFailure(f"checkpoint missing via {model_spec.checkpoint_env}: {checkpoint}")
    writer.attach_checkpoint(Path(checkpoint))
    return Path(checkpoint)


def _minimum_train_rows() -> int:
    """1400 full-budget rows, or any row count in a declared label-fraction panel.

    The gate exists so a shrunken manifest cannot masquerade as a formal run;
    the few-shot panel declares ``RS_EVAL_LABEL_FRACTION`` (recorded in run.json)
    and therefore keeps its smaller train fold.
    """
    return 1 if os.environ.get("RS_EVAL_LABEL_FRACTION") else 1400


def _groups(rows: list[dict]) -> dict[str, list[dict]]:
    groups = {
        split: sorted((row for row in rows if row["split"] == split), key=lambda r: r["sample_id"])
        for split in ("train", "validation", "test")
    }
    # Smoke-only cap on the evaluation folds; formal runs never set it. The
    # train fold stays whole so the manifest-size gate still applies.
    limit = int(os.environ.get("CLAY_WORKER_EVAL_LIMIT", "0") or 0)
    if limit:
        groups["validation"] = groups["validation"][:limit]
        groups["test"] = groups["test"][:limit]
    return groups


def _projector_path(target: str, seed: int, fingerprint: str) -> Path:
    return cache_root() / "projector" / target / f"seed{seed}" / f"{fingerprint}.pt"


def _publish_projector(model, target: str, seed: int, fingerprint: str, meta: dict) -> Path:
    import io

    import torch

    path = _projector_path(target, seed, fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    torch.save({"state_dict": model.state_dict(), **meta}, buffer)
    save_atomic(path, buffer.getvalue())
    return path


def _load_projector(model, target: str, seed: int, fingerprint: str):
    import torch

    path = _projector_path(target, seed, fingerprint)
    if not path.is_file():
        raise RgbnFailure(
            f"no distilled projector for {target} seed {seed} at {path}; "
            f"run {DISTILL_PROTOCOLS_BY_TARGET[target]} first"
        )
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    return path, payload


DISTILL_PROTOCOLS_BY_TARGET = {
    target: protocol for protocol, (target, _, _) in DISTILL_PROTOCOLS.items()
}


def train_action(request, writer, emit) -> int:
    checkpoint = _checkpoint(request, writer)
    prepared = _prepare(request, writer, checkpoint, emit)
    task_spec, protocol_spec, rows, manifest_digest, device, encoder, mean, std, waves, cache = (
        prepared
    )
    if protocol_spec.protocol_id in WIDE_B_STAGE_PROTOCOLS:
        return _train_wide_b_stage(
            request,
            writer,
            emit,
            protocol_spec,
            task_spec,
            rows,
            device,
            encoder,
            mean,
            std,
            waves,
            checkpoint,
        )
    if protocol_spec.protocol_id in B_STAGE_PROTOCOLS:
        return _train_b_stage(
            request,
            writer,
            emit,
            protocol_spec,
            task_spec,
            rows,
            device,
            encoder,
            mean,
            std,
            waves,
            cache,
            checkpoint,
        )
    if protocol_spec.protocol_id in DISTILL_PROTOCOLS:
        target = DISTILL_PROTOCOLS[protocol_spec.protocol_id][0]
        return _train_distill(
            request,
            writer,
            emit,
            protocol_spec,
            rows,
            device,
            encoder,
            mean,
            std,
            waves,
            cache,
            task_spec,
            target,
            checkpoint,
        )
    return _train_probe(
        request,
        writer,
        emit,
        protocol_spec,
        task_spec,
        rows,
        device,
        encoder,
        mean,
        std,
        waves,
        cache,
        checkpoint,
    )


def _mean_tokens(features: np.ndarray, device: str):
    """Temporal mean of the stride-8 tokens -> (1024,16,16) float32 tensor."""
    import torch

    array = np.asarray(features, dtype=np.float32).mean(axis=0)
    return torch.from_numpy(array).to(device)


def _train_distill(
    request,
    writer,
    emit,
    protocol_spec,
    rows,
    device,
    encoder,
    mean,
    std,
    waves,
    cache,
    task_spec,
    target: str,
    checkpoint: Path,
) -> int:
    import torch

    tree, dims = TARGET_TREE_DIMS[target]
    loss_kind = DISTILL_LOSS.get(protocol_spec.protocol_id, "cosine")
    groups = _groups(rows)
    if len(groups["train"]) < _minimum_train_rows():
        raise RgbnFailure(
            f"train fold has {len(groups['train'])} rows; the protocol needs "
            f">= {_minimum_train_rows()}"
        )
    data_root = Path(request.data_root)
    projector = build_projector(1024, dims).to(device)
    learning_rate = protocol_spec.learning_rate or 3e-3
    epochs = protocol_spec.epochs or 8
    optimiser = torch.optim.AdamW(projector.parameters(), lr=learning_rate, weight_decay=1e-4)
    writer.update_training(
        head_id=f"rgbn_projector_1024_256_{dims}",
        trainable_parameters=sum(p.numel() for p in projector.parameters()),
        total_parameters=sum(p.numel() for p in projector.parameters()),
    )
    writer.update(
        tags={
            "distill_target": target,
            "distill_teacher_tree": tree,
            "distill_dims": str(dims),
            "distill_epochs": str(epochs),
            "distill_lr": str(learning_rate),
            "distill_batch_patches": str(_patches_per_step(loss_kind)),
            "optimizer_steps_per_epoch": str(
                len(range(0, len(groups["train"]), _patches_per_step(loss_kind)))
            ),
            "distill_loss": (
                "symmetric InfoNCE over sampled observed tokens"
                if loss_kind == "infonce"
                else "mean((z - teacher)^2) over observed tokens, no L2 normalisation"
                if loss_kind == "mse"
                else "mean(1 - cos(z, teacher)) over observed tokens"
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
                    "infonce_patches_per_step": str(INFONCE_PATCHES),
                    "false_negative_handling": "none",
                }
                if loss_kind == "infonce"
                else {}
            ),
            "distill_aggregation": "temporal mean of F before E",
            "teacher_used_at_inference": "no",
            "train_samples": str(len(groups["train"])),
            "validation_samples": str(len(groups["validation"])),
            "test_samples": str(len(groups["test"])),
        }
    )
    started = time.time()
    minimize = loss_kind in {"mse", "infonce"}
    best = (float("inf") if minimize else -1.0, 0, None)
    best_cosine = float("nan")
    best_mse = float("nan")
    best_infonce = float("nan")
    best_top1 = float("nan")

    def epoch_loss(order: list[int], epoch: int, train: bool) -> tuple[float, int, int]:
        projector.train(train)
        total, steps, seen = 0.0, 0, 0
        patches = _patches_per_step(loss_kind)
        for step, start in enumerate(range(0, len(order), patches)):
            chosen = [groups["train"][index] for index in order[start : start + patches]]
            stacked = torch.stack(
                [_mean_tokens(cache.get(row, data_root), device) for row in chosen]
            )
            predictions = projector(stacked)
            targets, supports = [], []
            for row in chosen:
                grid, support = teacher_tokens(target, row, data_root)
                targets.append(torch.from_numpy(grid))
                supports.append(torch.from_numpy(support))
            target_tensor = torch.stack(targets).to(device)
            support_tensor = torch.stack(supports).to(device)
            if loss_kind == "infonce":
                generator = torch.Generator().manual_seed(
                    1_000_000 * request.seed + 1_000 * epoch + step
                )
                student, teacher = _sample_token_pairs(
                    predictions, target_tensor, support_tensor, INFONCE_TOKENS, generator
                )
                loss = _info_nce(student, teacher)
                seen += int(student.shape[0])
            elif loss_kind == "mse":
                loss = _masked_mse(predictions, target_tensor, support_tensor)
                seen += int(support_tensor.sum())
            else:
                loss = 1.0 - _cosine(predictions, target_tensor, support_tensor)
                seen += int(support_tensor.sum())
            if train:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
            total += float(loss.detach())
            steps += 1
        return total / max(steps, 1), steps, seen

    @torch.no_grad()
    def evaluate(split: str) -> tuple[float, float, float, float]:
        """Return (cosine, MSE, InfoNCE, retrieval top-1); NaN when unused."""
        projector.eval()
        cosine_total, mse_total, infonce_total = 0.0, 0.0, 0.0
        top1, nce_tokens, steps = 0, 0, 0
        rows_split = groups[split]
        patches = _patches_per_step(loss_kind)
        for start in range(0, len(rows_split), patches):
            chosen = rows_split[start : start + patches]
            stacked = torch.stack(
                [_mean_tokens(cache.get(row, data_root), device) for row in chosen]
            )
            predictions = projector(stacked)
            targets, supports = [], []
            for row in chosen:
                grid, support = teacher_tokens(target, row, data_root)
                targets.append(torch.from_numpy(grid))
                supports.append(torch.from_numpy(support))
            target_tensor = torch.stack(targets).to(device)
            support_tensor = torch.stack(supports).to(device)
            cosine_total += float(_cosine(predictions, target_tensor, support_tensor))
            mse_total += float(_masked_mse(predictions, target_tensor, support_tensor))
            if loss_kind == "infonce":
                generator = torch.Generator().manual_seed(10_000_000 + start)
                student, teacher = _sample_token_pairs(
                    predictions, target_tensor, support_tensor, INFONCE_TOKENS, generator
                )
                infonce_total += float(_info_nce(student, teacher))
                unit_student = torch.nn.functional.normalize(student.float(), dim=1)
                unit_teacher = torch.nn.functional.normalize(teacher.float(), dim=1)
                matches = (unit_student @ unit_teacher.transpose(0, 1)).argmax(dim=1)
                top1 += int(
                    (matches == torch.arange(student.shape[0], device=student.device)).sum()
                )
                nce_tokens += int(student.shape[0])
            steps += 1
        count = max(steps, 1)
        if loss_kind != "infonce":
            return cosine_total / count, mse_total / count, float("nan"), float("nan")
        return (
            cosine_total / count,
            mse_total / count,
            infonce_total / count,
            top1 / max(nce_tokens, 1),
        )

    @torch.no_grad()
    def mean_teacher_baseline() -> tuple[float, float]:
        """Cosine/MSE of predicting the train-set mean teacher vector."""
        total = torch.zeros(dims, device=device)
        count = 0
        for row in groups["train"]:
            grid, support = teacher_tokens(target, row, data_root)
            values = torch.from_numpy(grid).to(device)
            mask = torch.from_numpy(support).to(device).bool()
            total += values[:, mask].sum(dim=1)
            count += int(mask.sum())
        mean_vector = total / max(count, 1)
        cosine_sum, mse_sum, tokens = 0.0, 0.0, 0
        rows_split = groups["validation"]
        for start in range(0, len(rows_split), 2):
            chosen = rows_split[start : start + 2]
            targets, supports = [], []
            for row in chosen:
                grid, support = teacher_tokens(target, row, data_root)
                targets.append(torch.from_numpy(grid))
                supports.append(torch.from_numpy(support))
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

    history: list[float] = []
    for epoch in range(epochs):
        order = torch.randperm(
            len(groups["train"]),
            generator=torch.Generator().manual_seed(1_000 * request.seed + epoch),
        ).tolist()
        loss, steps, seen = epoch_loss(order, epoch, True)
        history.append(round(loss, 4))
        validation_cosine, validation_mse, validation_infonce, validation_top1 = evaluate(
            "validation"
        )
        writer.add_curve("distill_loss", "train", epoch + 1, loss)
        writer.add_curve("cosine", "validation", epoch + 1, validation_cosine)
        if loss_kind == "mse":
            writer.add_curve("mse", "validation", epoch + 1, validation_mse)
        if loss_kind == "infonce":
            writer.add_curve("infonce", "validation", epoch + 1, validation_infonce)
            writer.add_curve("retrieval_top1", "validation", epoch + 1, validation_top1)
        emit(
            request.run_id,
            "distill.epoch",
            f"epoch {epoch}: loss {loss:.4f}, val cosine {validation_cosine:.4f}, "
            f"val MSE {validation_mse:.5f}, val InfoNCE {validation_infonce:.4f}, "
            f"top1 {validation_top1:.4f}, tokens {seen}",
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
                {k: v.clone() for k, v in projector.state_dict().items()},
            )
            best_cosine = validation_cosine
            best_mse = validation_mse
            best_infonce = validation_infonce
            best_top1 = validation_top1

    if best[2] is not None:
        projector.load_state_dict(best[2])
    fingerprint = writer.record.tags["feature_fingerprint"]
    meta = {
        "target": target,
        "dims": dims,
        "hidden": 256,
        "in_channels": 1024,
        "seed": request.seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "feature_fingerprint": fingerprint,
        "manifest_sha256": writer.record.dataset.manifest_sha256,
        "checkpoint_sha256": writer.record.model_checkpoint_sha256,
        "source_run_id": request.run_id,
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
    (Path(request.runs_root) / request.run_id / "projector.pt").write_bytes(b"")
    torch.save(
        {"state_dict": projector.state_dict(), **meta},
        Path(request.runs_root) / request.run_id / "projector.pt",
    )
    published = _publish_projector(projector, target, request.seed, fingerprint, meta)
    writer.add_artifact("checkpoint", "projector.pt", "distilled projector E")
    metrics = [
        _metric("val_cosine", best_cosine, "validation"),
        _metric(
            "distill_train_loss",
            history[-1] if history else float("nan"),
            "train",
            direction="minimize",
        ),
    ]
    if loss_kind == "mse":
        metrics.append(_metric("val_mse", best_mse, "validation", direction="minimize"))
    if loss_kind == "infonce":
        metrics.append(_metric("val_infonce", best_infonce, "validation", direction="minimize"))
        metrics.append(_metric("val_retrieval_top1", best_top1, "validation"))
    writer.add_metrics(metrics)
    baseline_tags = {}
    if loss_kind in {"mse", "infonce"}:
        baseline_cosine, baseline_mse = mean_teacher_baseline()
        baseline_tags = {
            "mean_baseline_val_cosine": f"{baseline_cosine:.6f}",
            "mean_baseline_val_mse": f"{baseline_mse:.6f}",
        }
    writer.update(
        tags={
            "best_validation_cosine": f"{best_cosine:.6f}",
            "best_validation_mse": f"{best_mse:.6f}" if loss_kind == "mse" else "-",
            "best_validation_infonce": (f"{best_infonce:.6f}" if loss_kind == "infonce" else "-"),
            "best_validation_retrieval_top1": (
                f"{best_top1:.6f}" if loss_kind == "infonce" else "-"
            ),
            "best_epoch": str(best[1]),
            **baseline_tags,
            "projector_published": str(published),
            "feature_cache_hits": str(cache.hits),
            "feature_cache_misses": str(cache.misses),
            "feature_cache_seconds": f"{cache.seconds:.1f}",
            "distill_wall_time_seconds": f"{time.time() - started:.1f}",
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
    writer.complete(
        f"{task_spec.task_id} distill {target}: {dims} dims, temporal-mean F, {loss_label} loss, "
        f"best {selection_label} {best[0]:.6f} at epoch {best[1]}; "
        f"teacher not used at inference"
    )
    return 0


def _metric(name: str, value: float, split: str, direction: str = "maximize"):
    return MetricRecord(
        name=name,
        direction=direction,
        value=round(float(value), 6),
        split=split,
        is_primary=False,
    )


def _cosine(prediction, target, mask):
    import torch.nn.functional as F

    unit_prediction = F.normalize(prediction, dim=1)
    unit_target = F.normalize(target, dim=1)
    cosine = (unit_prediction * unit_target).sum(dim=1)
    return cosine[mask.bool()].mean()


def _masked_mse(prediction, target, mask):
    """Mean squared error over observed tokens, with no length normalisation."""
    error = ((prediction - target) ** 2).mean(dim=1)
    return error[mask.bool()].mean()


def _loss_improved(score: float, best: float, loss_kind: str) -> bool:
    """Epoch selection: MSE/InfoNCE minimise, cosine maximises."""
    return score < best if loss_kind in {"mse", "infonce"} else score > best


def _patches_per_step(loss_kind: str) -> int:
    """InfoNCE needs more negatives per step; the other losses stay at 2."""
    return INFONCE_PATCHES if loss_kind == "infonce" else 2


def _info_nce(student, teacher, temperature: float = INFONCE_TEMPERATURE):
    """Symmetric InfoNCE over matched (student, teacher) token pairs."""
    import torch
    import torch.nn.functional as F

    z = F.normalize(student.float(), dim=1)
    t = F.normalize(teacher.float(), dim=1)
    logits = z @ t.transpose(0, 1) / temperature
    labels = torch.arange(z.shape[0], device=z.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def _sample_token_pairs(prediction, target, mask, count: int, generator):
    """Flatten observed tokens and sample at most ``count`` matched pairs."""
    import torch

    _, dims, _, _ = prediction.shape
    flat_prediction = prediction.permute(0, 2, 3, 1).reshape(-1, dims)
    flat_target = target.permute(0, 2, 3, 1).reshape(-1, dims)
    indices = mask.bool().reshape(-1).nonzero(as_tuple=False).squeeze(1)
    if indices.numel() > count:
        pick = torch.randperm(indices.numel(), generator=generator)[:count].to(indices.device)
        indices = indices[pick]
    return flat_prediction[indices], flat_target[indices]


def _train_probe(
    request,
    writer,
    emit,
    protocol_spec,
    task_spec,
    rows,
    device,
    encoder,
    mean,
    std,
    waves,
    cache,
    checkpoint: Path,
) -> int:
    import torch
    import torch.nn.functional as F

    kind, target = PROBE_PROTOCOLS[protocol_spec.protocol_id]
    ignore = task_spec.ignore_index if task_spec.ignore_index is not None else VOID_CLASS
    groups = _groups(rows)
    data_root = Path(request.data_root)
    width = (
        1024
        if kind in {"raw", "supervised"}
        else 128
        if kind in {"pca128"}
        else 64
        if kind in {"pca64"}
        else TARGET_TREE_DIMS[target][1]
    )
    if kind == "supervised":
        width = target
    projector = None
    if kind == "distilled":
        projector = build_projector(1024, TARGET_TREE_DIMS[target][1]).to(device)
        path, payload = _load_projector(projector, target, request.seed, cache.fingerprint)
        for parameter in projector.parameters():
            parameter.requires_grad = False
        projector.eval()
        writer.update(tags={"projector_source_path": str(path)})
    elif kind == "supervised":
        projector = build_projector(1024, width).to(device)
        for parameter in projector.parameters():
            parameter.requires_grad = True
    head = build_head(width, task_spec.num_outputs).to(device)
    pca = None
    if kind in {"pca64", "pca128"}:
        pca, explained = _fit_pca(groups["train"], cache, data_root, device, components=width)
        writer.update(
            tags={"pca_components": str(width), "pca_explained_variance": f"{explained:.4f}"}
        )
        run_dir = Path(request.runs_root) / request.run_id
        np.save(run_dir / "pca_mean.npy", pca["mean"])
        np.save(run_dir / "pca_components.npy", pca["components"])
        writer.add_artifact("table", "pca_mean.npy", "train-only PCA mean")
        writer.add_artifact("table", "pca_components.npy", "train-only PCA basis")

    params = list(head.parameters())
    if projector is not None and kind == "supervised":
        params += list(projector.parameters())
    learning_rate = protocol_spec.learning_rate or 3e-3
    epochs = protocol_spec.epochs or 8
    optimiser = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-4)
    use_schedule = protocol_spec.lr_schedule == "warmup1-cosine"
    steps_per_epoch = len(range(0, len(groups["train"]), 2))
    scheduler = None
    if use_schedule:
        from rs_backbone_eval.schedule import build_scheduler

        scheduler = build_scheduler(optimiser, steps_per_epoch * epochs, steps_per_epoch)
    writer.update_training(
        head_id=f"rgbn_probe_{kind}_{width}",
        trainable_parameters=sum(p.numel() for p in params),
        total_parameters=sum(p.numel() for p in params),
    )
    writer.update(
        tags={
            "probe_kind": kind,
            "probe_target": str(target) if target is not None else "-",
            "probe_width": str(width),
            "probe_epochs": str(epochs),
            "probe_lr": str(learning_rate),
            "lr_schedule": protocol_spec.lr_schedule or "constant",
            "schedule_total_steps": str(steps_per_epoch * epochs),
            "schedule_warmup_steps": str(steps_per_epoch if use_schedule else 0),
            "label_grid": "10 m (128x128); token logits bilinearly upsampled",
            "token_grid_label_purity": "recorded per split",
            "teacher_used_at_inference": "no" if kind != "supervised" else "labels only",
            "train_samples": str(len(groups["train"])),
            "validation_samples": str(len(groups["validation"])),
            "test_samples": str(len(groups["test"])),
        }
    )
    if kind == "distilled":
        writer.update(tags={"projector_source_run": str(payload.get("source_run_id", ""))})

    started = time.time()

    def readout(stacked):
        if projector is not None:
            return projector(stacked)
        if pca is not None:
            flat = stacked.flatten(2).transpose(1, 2)  # (B,256,1024)
            centred = flat - torch.from_numpy(pca["mean"]).to(device)
            projected = centred @ torch.from_numpy(pca["components"]).to(device).T
            return projected.transpose(1, 2).reshape(-1, width, TOKENS, TOKENS)
        return stacked

    def batch_loss(chosen):
        """Token-grid logits, bilinearly upsampled to 10 m, against the 10 m labels.

        The readout lives on the stride-8 grid; the loss and the metric stay on
        the original 128x128 label grid so the numbers share the T11 track's
        evaluation grid. That does not claim the student observes 10 m detail:
        the upsampled logits are smooth by construction, and the token-grid IoU
        is recorded separately as a diagnostic.
        """
        stacked = torch.stack([_mean_tokens(cache.get(row, data_root), device) for row in chosen])
        labels = torch.stack(
            [torch.from_numpy(_labels(row, data_root, ignore)).to(device).long() for row in chosen]
        )
        logits = head(readout(stacked))
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        return F.cross_entropy(logits, labels, ignore_index=ignore)

    for epoch in range(epochs):
        order = torch.randperm(
            len(groups["train"]),
            generator=torch.Generator().manual_seed(1_000 * request.seed + epoch),
        ).tolist()
        head.train()
        if projector is not None and kind == "supervised":
            projector.train()
        total, steps = 0.0, 0
        for start in range(0, len(order), 2):
            chosen = [groups["train"][index] for index in order[start : start + 2]]
            loss = batch_loss(chosen)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach())
            steps += 1
        writer.add_curve("loss", "train", epoch + 1, total / max(steps, 1))
        emit(
            request.run_id,
            "probe.epoch",
            f"epoch {epoch}: loss {total / max(steps, 1):.4f}",
            progress=(epoch + 1) / epochs,
        )

    for split in ("validation", "test"):
        head.eval()
        if projector is not None:
            projector.eval()
        confusion = np.zeros((task_spec.num_outputs, task_spec.num_outputs), dtype=np.int64)
        purity, evaluated = [], 0
        with torch.no_grad():
            for start in range(0, len(groups[split]), 2):
                chosen = groups[split][start : start + 2]
                evaluated += len(chosen)
                stacked = torch.stack(
                    [_mean_tokens(cache.get(row, data_root), device) for row in chosen]
                )
                labels, pure = zip(
                    *[majority_labels(_labels(row, data_root, ignore)) for row in chosen],
                    strict=True,
                )
                purity.extend(pure)
                target_labels = (
                    torch.from_numpy(np.stack([_labels(row, data_root, ignore) for row in chosen]))
                    .to(device)
                    .long()
                )
                logits = head(readout(stacked))
                logits = F.interpolate(
                    logits, size=target_labels.shape[-2:], mode="bilinear", align_corners=False
                )
                prediction = logits.argmax(dim=1)
                mask = target_labels != ignore
                flat = (
                    (target_labels[mask] * task_spec.num_outputs + prediction[mask]).cpu().numpy()
                )
                confusion += np.bincount(flat, minlength=task_spec.num_outputs**2).reshape(
                    task_spec.num_outputs, task_spec.num_outputs
                )
        from rs_backbone_eval.matrix import (
            OFFICIAL_TEST_SAMPLES,
            OFFICIAL_VALIDATION_SAMPLES,
        )

        expected = (
            OFFICIAL_TEST_SAMPLES[task_spec.task_id]
            if split == "test"
            else OFFICIAL_VALIDATION_SAMPLES[task_spec.task_id]
        )
        if evaluated != expected and os.environ.get("CLAY_WORKER_ALLOW_PARTIAL_EVAL") != "1":
            raise RgbnFailure(
                f"{split} fold evaluated on {evaluated} of the official {expected} rows"
            )
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
        writer.update(
            tags={
                f"{split}_samples_evaluated": str(evaluated),
                f"{split}_manifest_samples": str(len(groups[split])),
                f"{split}_expected_samples": str(expected),
                f"{split}_pure_blocks": f"{float(np.mean(purity)):.4f}",
            }
        )

    test = next(metric for metric in writer.record.metrics if metric.is_primary)
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=round(time.time() - started, 1)
    )
    writer.complete(
        f"{task_spec.task_id} probe {kind} {target or ''} width {width}: "
        f"test mIoU {test.value:.4f} on the 10 m label grid "
        f"(token logits bilinearly upsampled); teacher not read at inference"
    )
    return 0


def _labels(row: dict, data_root: Path, ignore: int) -> np.ndarray:
    labels = np.load(data_root / row["label_path"])[0].astype(np.int64)
    return np.where(labels == VOID_CLASS, ignore, labels)


def _fit_pca(
    rows: list[dict], cache, data_root: Path, device: str, components: int = 64
) -> tuple[dict, float]:
    """Train-only PCA of the stride-8 tokens, deterministic sign convention."""

    samples = []
    for index, row in enumerate(rows):
        if index % 2:  # every other patch keeps the fit under a minute
            continue
        tokens = np.asarray(cache.get(row, data_root), dtype=np.float32).mean(axis=0)
        samples.append(tokens.reshape(1024, TOKENS * TOKENS).T.astype(np.float64))
    matrix = np.concatenate(samples, axis=0)
    mean = matrix.mean(axis=0)
    centred = matrix - mean
    covariance = centred.T @ centred / max(len(centred) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:components]
    components = eigenvectors[:, order].T
    # Deterministic sign: the largest-magnitude loading is positive.
    for row in range(components.shape[0]):
        pivot = int(np.argmax(np.abs(components[row])))
        if components[row, pivot] < 0:
            components[row] *= -1
    explained = float(eigenvalues[order].sum() / eigenvalues.sum())
    return {
        "mean": mean.astype(np.float32),
        "components": components.astype(np.float32),
    }, explained


def _unfreeze_last_blocks(encoder, blocks: int = B_STAGE_BLOCKS) -> list[str]:
    """Unfreeze the last ``blocks`` transformer blocks and the final norm."""
    layers = encoder.transformer.layers
    if blocks > len(layers):
        raise RgbnFailure(f"cannot unfreeze {blocks} of {len(layers)} blocks")
    for block in layers[-blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in encoder.transformer.norm.parameters():
        parameter.requires_grad = True
    names = [f"transformer.layers.{index}" for index in range(len(layers) - blocks, len(layers))]
    names.append("transformer.norm")
    return names


def _token_forward(encoder, pixels, waves, *, grad: bool, chunk: int = B_STAGE_FRAME_CHUNK):
    """(T,4,128,128) -> (T,1024,16,16) tokens, chunked over frames for memory."""
    import torch

    parts = []
    for start in range(0, pixels.shape[0], chunk):
        parts.append(frame_tokens(encoder, pixels[start : start + chunk], waves, grad=grad))
    return torch.cat(parts, dim=0)


def _train_wide_b_stage(
    request,
    writer,
    emit,
    protocol_spec,
    task_spec,
    rows,
    device,
    encoder,
    mean,
    std,
    waves,
    checkpoint: Path,
) -> int:
    """Wide-head B stage: adapted blocks + a head on the raw token width.

    Kept separate from ``_train_b_stage`` so the admitted 128-d pair keeps its
    original training path. Both arms build the head before the optional
    teacher projector and use an explicit per-epoch permutation.
    """
    import torch
    import torch.nn.functional as F

    target = WIDE_B_STAGE_PROTOCOLS[protocol_spec.protocol_id]
    blocks, dims, _ = B_STAGE_VARIANTS[protocol_spec.protocol_id]
    profile = profile_for(task_spec.task_id)
    ignore = task_spec.ignore_index if task_spec.ignore_index is not None else VOID_CLASS
    groups = _groups(rows)
    if len(groups["train"]) < _minimum_train_rows():
        raise RgbnFailure(
            f"train fold has {len(groups['train'])} rows; the protocol needs "
            f">= {_minimum_train_rows()}"
        )
    data_root = Path(request.data_root)
    unfrozen = _unfreeze_last_blocks(encoder, blocks)
    encoder.train()
    # Head first: the teacher arm must not shift the head init.
    head = build_head(dims, task_spec.num_outputs).to(device)
    teacher_projector = (
        build_projector(1024, B_STAGE_DIMS).to(device) if target is not None else None
    )
    head_lr = protocol_spec.learning_rate or 3e-3
    backbone_lr = protocol_spec.backbone_learning_rate or B_STAGE_BACKBONE_LR
    epochs = protocol_spec.epochs or 8
    backbone_parameters = [p for p in encoder.parameters() if p.requires_grad]
    head_parameters = list(head.parameters())
    if teacher_projector is not None:
        head_parameters += list(teacher_projector.parameters())
    optimiser = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": head_parameters, "lr": head_lr},
        ],
        weight_decay=1e-4,
    )
    use_schedule = protocol_spec.lr_schedule == "warmup1-cosine"
    steps_per_epoch = len(range(0, len(groups["train"]), 2))
    scheduler = None
    if use_schedule:
        from rs_backbone_eval.schedule import build_scheduler

        scheduler = build_scheduler(optimiser, steps_per_epoch * epochs, steps_per_epoch)
    writer.update_training(
        head_id=f"rgbn_b_stage_last{blocks}_{dims}_wide",
        trainable_parameters=len_total(backbone_parameters) + len_total(head_parameters),
        total_parameters=len_total(list(encoder.parameters())) + len_total(head_parameters),
    )
    writer.update(
        tags={
            "b_stage": "backbone adaptation (wide head)",
            "unfrozen_modules": ",".join(unfrozen),
            "backbone_lr": str(backbone_lr),
            "head_lr": str(head_lr),
            "teacher_weight": str(B_STAGE_TEACHER_WEIGHT if target else 0.0),
            "b_stage_target": target or "-",
            "b_stage_blocks": str(blocks),
            "b_stage_width": str(dims),
            "b_stage_head_projector": "False",
            "b_stage_teacher_projector": str(teacher_projector is not None),
            "b_stage_epochs": str(epochs),
            "lr_schedule": protocol_spec.lr_schedule or "constant",
            "schedule_total_steps": str(steps_per_epoch * epochs),
            "schedule_warmup_steps": str(steps_per_epoch if use_schedule else 0),
            "b_stage_frames": str(profile.max_frames),
            "b_stage_batch": "2 patches per optimizer step (effective 2)",
            "optimizer_steps_per_epoch": str(
                len(range(0, len(groups["train"]), B_STAGE_ACCUMULATION))
            ),
            "batch_order": "explicit per-epoch permutation seeded by 1000*seed+epoch",
            "label_grid": "10 m (128x128); token logits bilinearly upsampled",
            "teacher_used_at_inference": "no",
            "train_samples": str(len(groups["train"])),
            "validation_samples": str(len(groups["validation"])),
            "test_samples": str(len(groups["test"])),
        }
    )
    started = time.time()

    def forward_patch(row: dict, *, grad: bool):
        pixels, _, _ = read_frames(row, data_root, profile)
        tensor = torch.from_numpy(pixels).to(device)
        channels = len(profile.bands)
        tensor = (tensor - mean.to(device).view(1, channels, 1, 1)) / std.to(device).view(
            1, channels, 1, 1
        )
        tokens = _token_forward(encoder, tensor, waves.to(device), grad=grad)
        return tokens.mean(dim=0)

    def label_tensor(row: dict):
        return torch.from_numpy(_labels(row, data_root, ignore)).to(device).long().unsqueeze(0)

    def teacher_loss(features, row: dict):
        if teacher_projector is None:
            return torch.zeros((), device=device)
        grid, support = teacher_tokens(target, row, data_root)
        target_tensor = torch.from_numpy(grid).unsqueeze(0).to(device)
        support_tensor = torch.from_numpy(support).unsqueeze(0).to(device)
        return 1.0 - _cosine(teacher_projector(features), target_tensor, support_tensor)

    accumulation = B_STAGE_ACCUMULATION
    updates_per_epoch = len(range(0, len(groups["train"]), accumulation))
    for epoch in range(epochs):
        order = torch.randperm(
            len(groups["train"]),
            generator=torch.Generator().manual_seed(1_000 * request.seed + epoch),
        ).tolist()
        encoder.train()
        head.train()
        if teacher_projector is not None:
            teacher_projector.train()
        optimiser.zero_grad(set_to_none=True)
        total, steps, skipped = 0.0, 0, 0
        for chunk in accumulation_chunks(order, accumulation):
            loss = torch.zeros((), device=device)
            valid = 0
            for index in chunk:
                row = groups["train"][index]
                labels = label_tensor(row)
                if int((labels != ignore).sum()) == 0:
                    continue
                features = forward_patch(row, grad=True).unsqueeze(0)
                logits = head(features)
                logits = F.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                row_loss = F.cross_entropy(logits, labels, ignore_index=ignore)
                if target is not None:
                    row_loss = row_loss + B_STAGE_TEACHER_WEIGHT * teacher_loss(features, row)
                loss = loss + row_loss
                valid += 1
            if valid == 0:
                skipped += 1
                continue
            # One optimizer step per chunk: the chunk is the effective batch.
            # Nesting a ``pending`` counter over two-patch chunks silently
            # doubled the effective batch (364 instead of 728 updates).
            (loss / valid).backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            steps += 1
            total += float(loss.detach() / valid)
        if steps + skipped != updates_per_epoch:
            raise RgbnFailure(
                f"epoch {epoch + 1} took {steps} optimizer steps (+{skipped} empty chunks) "
                f"for {len(order)} rows at effective batch {accumulation}; "
                f"expected {updates_per_epoch}"
            )
        mean_loss = total / max(steps, 1)
        writer.add_curve("loss", "train", epoch + 1, mean_loss)
        emit(
            request.run_id,
            "b_stage.epoch",
            f"epoch {epoch + 1}/{epochs}: loss {mean_loss:.4f}, {steps} steps",
            progress=(epoch + 1) / epochs,
        )

    for split in ("validation", "test"):
        encoder.eval()
        head.eval()
        if teacher_projector is not None:
            teacher_projector.eval()
        confusion = np.zeros((task_spec.num_outputs, task_spec.num_outputs), dtype=np.int64)
        evaluated = 0
        with torch.no_grad():
            for row in groups[split]:
                evaluated += 1
                features = forward_patch(row, grad=False).unsqueeze(0)
                logits = head(features)
                labels = label_tensor(row)
                logits = F.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                prediction = logits.argmax(dim=1)
                mask = labels != ignore
                flat = (labels[mask] * task_spec.num_outputs + prediction[mask]).cpu().numpy()
                confusion += np.bincount(flat, minlength=task_spec.num_outputs**2).reshape(
                    task_spec.num_outputs, task_spec.num_outputs
                )
        from rs_backbone_eval.matrix import (
            OFFICIAL_TEST_SAMPLES,
            OFFICIAL_VALIDATION_SAMPLES,
        )

        expected = (
            OFFICIAL_TEST_SAMPLES[task_spec.task_id]
            if split == "test"
            else OFFICIAL_VALIDATION_SAMPLES[task_spec.task_id]
        )
        if evaluated != expected and os.environ.get("CLAY_WORKER_ALLOW_PARTIAL_EVAL") != "1":
            raise RgbnFailure(
                f"{split} fold evaluated on {evaluated} of the official {expected} rows"
            )
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
        writer.update(
            tags={
                f"{split}_samples_evaluated": str(evaluated),
                f"{split}_manifest_samples": str(len(groups[split])),
                f"{split}_expected_samples": str(expected),
            }
        )

    test = next(metric for metric in writer.record.metrics if metric.is_primary)
    import torch as _torch

    backbone_state = {
        name: value.detach().cpu()
        for name, value in encoder.named_parameters()
        if value.requires_grad
    }
    state = {
        "state_dict": {f"backbone.{name}": value for name, value in backbone_state.items()}
        | {f"head.{name}": value.cpu() for name, value in head.state_dict().items()},
        "trainable_parameters": sum(int(v.numel()) for v in backbone_state.values())
        + sum(int(p.numel()) for p in head_parameters),
        "unfrozen_modules": unfrozen,
        "target": target,
        "blocks": blocks,
        "width": dims,
        "head_projector": False,
        "teacher_projector": teacher_projector is not None,
        "seed": request.seed,
        "epochs": epochs,
        "teacher_weight": B_STAGE_TEACHER_WEIGHT if target else 0.0,
    }
    if teacher_projector is not None:
        state["state_dict"] |= {
            f"teacher_projector.{name}": value.detach().cpu()
            for name, value in teacher_projector.state_dict().items()
        }
    _torch.save(state, Path(request.runs_root) / request.run_id / "model.pt")
    writer.add_artifact("checkpoint", "model.pt", "wide-head B-stage blocks, teacher E and head")
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=round(time.time() - started, 1)
    )
    writer.complete(
        f"{task_spec.task_id} wide B-stage last{blocks} blocks, width {dims}: "
        f"{target or 'labels only'}, test mIoU {test.value:.4f}; teacher not used at inference"
    )
    return 0


def _train_b_stage(
    request,
    writer,
    emit,
    protocol_spec,
    task_spec,
    rows,
    device,
    encoder,
    mean,
    std,
    waves,
    _cache,
    checkpoint: Path,
) -> int:
    """Backbone adaptation: last-4 blocks + E + head, with/without teacher loss."""
    import torch
    import torch.nn.functional as F

    target = B_STAGE_PROTOCOLS[protocol_spec.protocol_id]
    blocks, dims, use_projector = B_STAGE_VARIANTS[protocol_spec.protocol_id]
    profile = profile_for(task_spec.task_id)
    ignore = task_spec.ignore_index if task_spec.ignore_index is not None else VOID_CLASS
    groups = _groups(rows)
    if len(groups["train"]) < _minimum_train_rows():
        raise RgbnFailure(
            f"train fold has {len(groups['train'])} rows; the protocol needs "
            f">= {_minimum_train_rows()}"
        )
    data_root = Path(request.data_root)
    unfrozen = _unfreeze_last_blocks(encoder, blocks)
    encoder.train()
    head_projector = build_projector(1024, dims).to(device) if use_projector else None
    teacher_projector = None
    if target is not None:
        teacher_projector = (
            head_projector if use_projector else build_projector(1024, B_STAGE_DIMS).to(device)
        )
    head = build_head(dims if use_projector else 1024, task_spec.num_outputs).to(device)
    readout = head_projector if head_projector is not None else (lambda features: features)
    head_lr = protocol_spec.learning_rate or 3e-3
    backbone_lr = protocol_spec.backbone_learning_rate or B_STAGE_BACKBONE_LR
    epochs = protocol_spec.epochs or 8
    backbone_parameters = [p for p in encoder.parameters() if p.requires_grad]
    head_parameters = list(head.parameters())
    seen = {id(parameter) for parameter in head_parameters}
    for module in (head_projector, teacher_projector):
        if module is None:
            continue
        for parameter in module.parameters():
            if id(parameter) not in seen:
                head_parameters.append(parameter)
                seen.add(id(parameter))
    optimiser = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_lr},
            {"params": head_parameters, "lr": head_lr},
        ],
        weight_decay=1e-4,
    )
    use_schedule = protocol_spec.lr_schedule == "warmup1-cosine"
    steps_per_epoch = len(range(0, len(groups["train"]), 2))
    scheduler = None
    if use_schedule:
        from rs_backbone_eval.schedule import build_scheduler

        scheduler = build_scheduler(optimiser, steps_per_epoch * epochs, steps_per_epoch)
    writer.update_training(
        head_id=f"rgbn_b_stage_last{blocks}_{dims}{'' if use_projector else '_raw'}",
        trainable_parameters=len_total(backbone_parameters) + len_total(head_parameters),
        total_parameters=len_total(list(encoder.parameters())) + len_total(head_parameters),
    )
    writer.update(
        tags={
            "b_stage": "backbone adaptation",
            "unfrozen_modules": ",".join(unfrozen),
            "backbone_lr": str(backbone_lr),
            "head_lr": str(head_lr),
            "teacher_weight": str(B_STAGE_TEACHER_WEIGHT if target else 0.0),
            "b_stage_target": target or "-",
            "b_stage_blocks": str(blocks),
            "b_stage_width": str(dims),
            "b_stage_head_projector": str(use_projector),
            "b_stage_teacher_projector": str(
                teacher_projector is not None and teacher_projector is not head_projector
            ),
            "b_stage_epochs": str(epochs),
            "lr_schedule": protocol_spec.lr_schedule or "constant",
            "schedule_total_steps": str(steps_per_epoch * epochs),
            "schedule_warmup_steps": str(steps_per_epoch if use_schedule else 0),
            "b_stage_frames": str(profile.max_frames),
            "b_stage_batch": "1 patch x 2 accumulation (effective 2)",
            "label_grid": "10 m (128x128); token logits bilinearly upsampled",
            "teacher_used_at_inference": "no",
            "train_samples": str(len(groups["train"])),
            "validation_samples": str(len(groups["validation"])),
            "test_samples": str(len(groups["test"])),
        }
    )
    started = time.time()

    def forward_patch(row: dict, *, grad: bool):
        pixels, _, _ = read_frames(row, data_root, profile)
        tensor = torch.from_numpy(pixels).to(device)
        channels = len(profile.bands)
        tensor = (tensor - mean.to(device).view(1, channels, 1, 1)) / std.to(device).view(
            1, channels, 1, 1
        )
        tokens = _token_forward(encoder, tensor, waves.to(device), grad=grad)
        return tokens.mean(dim=0)

    def logits_for(row: dict, *, grad: bool):
        features = forward_patch(row, grad=grad).unsqueeze(0)  # (1,1024,16,16)
        logits = head(readout(features))
        labels = torch.from_numpy(_labels(row, data_root, ignore)).to(device).long()
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        return logits, labels

    def teacher_loss(row: dict, features) -> Any:
        grid, support = teacher_tokens(target, row, data_root)
        target_tensor = torch.from_numpy(grid).unsqueeze(0).to(device)
        support_tensor = torch.from_numpy(support).unsqueeze(0).to(device)
        return 1.0 - _cosine(teacher_projector(features), target_tensor, support_tensor)

    def validation_iou() -> float:
        """Intermediate validation mIoU (recorded, never used for selection)."""
        encoder.eval()
        head.eval()
        if head_projector is not None:
            head_projector.eval()
        confusion = np.zeros((task_spec.num_outputs, task_spec.num_outputs), dtype=np.int64)
        with torch.no_grad():
            for row in groups["validation"]:
                features = forward_patch(row, grad=False).unsqueeze(0)
                logits = head(readout(features))
                labels = (
                    torch.from_numpy(_labels(row, data_root, ignore)).to(device).long().unsqueeze(0)
                )
                logits = F.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                prediction = logits.argmax(dim=1)
                mask = labels != ignore
                flat = (labels[mask] * task_spec.num_outputs + prediction[mask]).cpu().numpy()
                confusion += np.bincount(flat, minlength=task_spec.num_outputs**2).reshape(
                    task_spec.num_outputs, task_spec.num_outputs
                )
        tp = np.diag(confusion)
        fp = confusion.sum(axis=0) - tp
        fn = confusion.sum(axis=1) - tp
        present = confusion.sum(axis=1) > 0
        iou = np.where(present, tp / np.maximum(tp + fp + fn, 1), np.nan)
        return float(np.nanmean(iou))

    accumulation = 2
    for epoch in range(epochs):
        order = torch.randperm(
            len(groups["train"]),
            generator=torch.Generator().manual_seed(1_000 * request.seed + epoch),
        ).tolist()
        encoder.train()
        if head_projector is not None:
            head_projector.train()
        if teacher_projector is not None and teacher_projector is not head_projector:
            teacher_projector.train()
        head.train()
        optimiser.zero_grad(set_to_none=True)
        total, steps = 0.0, 0
        for start in range(0, len(order), accumulation):
            chunk = [groups["train"][index] for index in order[start : start + accumulation]]
            loss = torch.zeros((), device=device)
            for row in chunk:
                # One backbone pass per patch: the label loss and the teacher
                # cosine both read the same token tensor.
                features = forward_patch(row, grad=True).unsqueeze(0)
                logits = head(readout(features))
                labels = (
                    torch.from_numpy(_labels(row, data_root, ignore)).to(device).long().unsqueeze(0)
                )
                logits = F.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = loss + F.cross_entropy(logits, labels, ignore_index=ignore) / len(chunk)
                if target is not None:
                    loss = loss + (
                        B_STAGE_TEACHER_WEIGHT * teacher_loss(row, features) / len(chunk)
                    )
            loss.backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            total += float(loss.detach())
            steps += 1
        writer.add_curve("loss", "train", epoch + 1, total / max(steps, 1))
        if scheduler is not None:
            interval = 4 if epochs <= 32 else 10
            if (epoch + 1) % interval == 0:
                value = validation_iou()
                writer.add_curve("mean_iou", "validation", epoch + 1, value)
                emit(
                    request.run_id,
                    "b_stage.validation",
                    f"epoch {epoch + 1}/{epochs}: validation mIoU {value:.4f}",
                    progress=(epoch + 1) / epochs,
                )
        emit(
            request.run_id,
            "b_stage.epoch",
            f"epoch {epoch}: loss {total / max(steps, 1):.4f}, {steps} steps",
            progress=(epoch + 1) / epochs,
        )

    # Evaluation with the frozen-readout protocol: the backbone and E are in
    # eval mode, the same head, the same 10 m labels.
    for split in ("validation", "test"):
        encoder.eval()
        if head_projector is not None:
            head_projector.eval()
        if teacher_projector is not None and teacher_projector is not head_projector:
            teacher_projector.eval()
        head.eval()
        confusion = np.zeros((task_spec.num_outputs, task_spec.num_outputs), dtype=np.int64)
        evaluated = 0
        with torch.no_grad():
            for row in groups[split]:
                evaluated += 1
                logits, labels = logits_for(row, grad=False)
                prediction = logits.argmax(dim=1)[0]
                mask = labels != ignore
                flat = (labels[mask] * task_spec.num_outputs + prediction[mask]).cpu().numpy()
                confusion += np.bincount(flat, minlength=task_spec.num_outputs**2).reshape(
                    task_spec.num_outputs, task_spec.num_outputs
                )
        from rs_backbone_eval.matrix import (
            OFFICIAL_TEST_SAMPLES,
            OFFICIAL_VALIDATION_SAMPLES,
        )

        expected = (
            OFFICIAL_TEST_SAMPLES[task_spec.task_id]
            if split == "test"
            else OFFICIAL_VALIDATION_SAMPLES[task_spec.task_id]
        )
        if evaluated != expected and os.environ.get("CLAY_WORKER_ALLOW_PARTIAL_EVAL") != "1":
            raise RgbnFailure(
                f"{split} fold evaluated on {evaluated} of the official {expected} rows"
            )
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
        writer.update(
            tags={
                f"{split}_samples_evaluated": str(evaluated),
                f"{split}_manifest_samples": str(len(groups[split])),
                f"{split}_expected_samples": str(expected),
            }
        )

    test = next(metric for metric in writer.record.metrics if metric.is_primary)
    import torch as _torch

    # The adapted blocks are part of the result: a checkpoint that only holds
    # E and the head cannot be re-evaluated or transferred (review 2026-11-11,
    # P1). Save every parameter this run trained, namespaced.
    backbone_state = {
        name: value.detach().cpu()
        for name, value in encoder.named_parameters()
        if value.requires_grad
    }
    projector_state = {}
    if head_projector is not None:
        projector_state |= {
            f"head_projector.{name}": value.detach().cpu()
            for name, value in head_projector.state_dict().items()
        }
    if teacher_projector is not None and teacher_projector is not head_projector:
        projector_state |= {
            f"teacher_projector.{name}": value.detach().cpu()
            for name, value in teacher_projector.state_dict().items()
        }
    _torch.save(
        {
            "state_dict": {f"backbone.{name}": value for name, value in backbone_state.items()}
            | projector_state
            | {f"head.{name}": value.cpu() for name, value in head.state_dict().items()},
            "trainable_parameters": sum(int(v.numel()) for v in backbone_state.values())
            + sum(int(p.numel()) for p in head_parameters),
            "unfrozen_modules": unfrozen,
            "target": target,
            "blocks": blocks,
            "width": dims,
            "head_projector": use_projector,
            "teacher_projector": teacher_projector is not None
            and teacher_projector is not head_projector,
            "seed": request.seed,
            "epochs": epochs,
            "teacher_weight": B_STAGE_TEACHER_WEIGHT if target else 0.0,
        },
        Path(request.runs_root) / request.run_id / "model.pt",
    )
    writer.add_artifact("checkpoint", "model.pt", "B-stage adapted blocks, projector and head")
    writer.update_training(
        epochs_completed=epochs, wall_time_seconds=round(time.time() - started, 1)
    )
    readout_label = "128-d E + head" if use_projector else f"{dims}-d wide head"
    writer.complete(
        f"{task_spec.task_id} B-stage last{blocks} blocks, {readout_label}: "
        f"{target or 'labels only'}, "
        f"test mIoU {test.value:.4f}; teacher not used at inference"
    )
    return 0


def len_total(parameters) -> int:
    return int(sum(p.numel() for p in parameters))
