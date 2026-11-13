"""Clay T11D fusion diagnostic: the head reads ``F`` and ``E_teacher(F)``.

Handoff: ``docs/evaluation/高维特征与教师投影拼接_agent执行交接_2026-11-13.md``.

The source ``F`` is the temporal-mean stride-8 token grid of the already-adapted
``clay_rgbn_ft_last4_raw1024`` checkpoint for the *same* seed. Three arms share
one head initialisation and one F cache:

* ``raw``     : ``head(F)``
* ``rand``    : ``head([F, N(E_init(F))])``
* ``teacher`` : ``head([F, N(E_best(F))])``

Stage A trains ``E`` on the train split with a cosine loss against the
OlmoEarth v1.3 2019 128-d teacher; stage B freezes ``F`` and ``E`` and trains
one head per arm. The fit run publishes ``fusion_init.pt`` (head base + E_init)
and ``projector_best.pt``; the three head runs load them instead of rebuilding
from RNG, so the arms cannot drift apart.

Everything here is diagnostic and never enters the formal matrix.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from .rgbn import (
    BAND_PROFILES,
    GSD_M,
    TOKENS,
    RgbnFailure,
    _checkpoint,
    _groups,
    _labels,
    _minimum_train_rows,
    accumulation_chunks,
    build_encoder,
    build_head,
    build_projector,
    cache_root,
    cosine_alignment,
    frame_tokens,
    read_frames,
    read_rows,
    save_atomic,
    save_npy_atomic,
    teacher_tokens,
)

SOURCE_MODEL = "clay_v1_5"
SOURCE_PROTOCOL = "clay_rgbn_ft_last4_raw1024"
FIT_PROTOCOL = "clay_rgbn_fusion_o128_fit"
RAW_PROTOCOL = "clay_rgbn_fusion_raw"
RAND_PROTOCOL = "clay_rgbn_fusion_rand_o128"
TEACHER_PROTOCOL = "clay_rgbn_fusion_teacher_o128"
FUSION_PROTOCOLS = {FIT_PROTOCOL, RAW_PROTOCOL, RAND_PROTOCOL, TEACHER_PROTOCOL}
TEACHER_TARGET = "oe128"
TEACHER_DIMS = 128
PROJECTOR_HIDDEN = 256
HEAD_HIDDEN = 128
EPOCHS = 8
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 1e-4
ACCUMULATION = 2
TOKEN_DIM = 1024
CACHE_VERSION = "2026-11-13a.fusion-f"
REQUIRED_SOURCE_FIXES = ("ffdc6db", "7377fb6")
HEAD_SEED_OFFSET = 0x11EAD
PROJECTOR_SEED_OFFSET = 0x9F17


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_run_id(seed: int) -> str:
    return f"{SOURCE_MODEL}-T11D-{SOURCE_PROTOCOL}-train-{seed}"


def fit_run_id(seed: int) -> str:
    return f"{SOURCE_MODEL}-T11D-{FIT_PROTOCOL}-train-{seed}"


def fit_parent_id(request) -> str:
    """Parent fit run for this head; smoke runs set it explicitly."""
    return os.environ.get("CLAY_WORKER_FUSION_FIT_RUN") or fit_run_id(request.seed)


def _fusion_groups(rows):
    groups = _groups(rows)
    limit = int(os.environ.get("CLAY_WORKER_TRAIN_LIMIT", "0") or 0)
    if limit:
        groups["train"] = groups["train"][:limit]
    return groups


def _fusion_minimum_rows() -> int:
    if os.environ.get("CLAY_WORKER_TRAIN_LIMIT"):
        return 1
    return _minimum_train_rows()


def _source_lock(request, seed: int, original_checkpoint: Path) -> dict:
    """Resolve the adapted raw1024 checkpoint from the real run ledger."""
    run_dir = Path(request.runs_root) / source_run_id(seed)
    record_path = run_dir / "run.json"
    if not record_path.is_file():
        raise RgbnFailure(f"source run missing: {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    training = record.get("training") or {}
    tags = record.get("tags") or {}
    if record.get("status") != "completed":
        raise RgbnFailure(f"source run {source_run_id(seed)} is {record.get('status')!r}")
    if record.get("protocol_id") != SOURCE_PROTOCOL:
        raise RgbnFailure(f"source run protocol is {record.get('protocol_id')!r}")
    if int(record.get("seed", -1)) != seed:
        raise RgbnFailure(f"source run seed {record.get('seed')} != requested {seed}")
    if int(training.get("epochs_completed") or 0) != EPOCHS:
        raise RgbnFailure(
            f"source run completed {training.get('epochs_completed')} epochs, expected {EPOCHS}"
        )
    if tags.get("panel") != "diagnostic":
        raise RgbnFailure("source run is not on the diagnostic panel")
    if tags.get("init", "pretrained") != "pretrained":
        raise RgbnFailure(f"source run init is {tags.get('init')!r}")
    if tags.get("label_fraction", "1.0") != "1.0":
        raise RgbnFailure(f"source run label_fraction is {tags.get('label_fraction')!r}")
    model_pt = run_dir / "model.pt"
    if not model_pt.is_file():
        raise RgbnFailure(f"source run has no adapted model.pt: {model_pt}")
    lock = {
        "source_run_id": source_run_id(seed),
        "source_git_commit": record.get("git_commit"),
        "source_protocol": SOURCE_PROTOCOL,
        "seed": seed,
        "epochs_completed": int(training["epochs_completed"]),
        "manifest_sha256": (record.get("dataset") or {}).get("manifest_sha256"),
        "original_checkpoint": str(original_checkpoint),
        "original_checkpoint_sha256": _sha256_file(original_checkpoint),
        "adapted_model_pt": str(model_pt),
        "adapted_model_sha256": _sha256_file(model_pt),
        "required_fixes": list(REQUIRED_SOURCE_FIXES),
    }
    out = cache_root() / "fusion" / "source_lock" / f"seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    return lock


def _load_adapted_encoder(original_checkpoint: Path, model_pt: Path, device: str):
    """Original Clay checkpoint + the source run's last-4-block adaptation."""
    import torch

    from clay_worker.clay_probe import add_upstream_to_path

    add_upstream_to_path()
    encoder = build_encoder(original_checkpoint).to(device).eval()
    payload = torch.load(model_pt, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") or {}
    backbone = {
        key[len("backbone.") :]: value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    adapted_prefixes = (
        "transformer.layers.20.",
        "transformer.layers.21.",
        "transformer.layers.22.",
        "transformer.layers.23.",
        "transformer.norm.",
    )
    encoder_parameters = {name for name, _ in encoder.named_parameters()}
    unknown = sorted(key for key in backbone if key not in encoder_parameters)
    outside = sorted(key for key in backbone if not key.startswith(adapted_prefixes))
    expected = {name for name in encoder_parameters if name.startswith(adapted_prefixes)}
    missing = sorted(expected - set(backbone))
    if unknown or outside or missing:
        raise RgbnFailure(
            "source adaptation does not match the last-4-block recipe: "
            f"unknown={unknown[:3]}, outside={outside[:3]}, missing={missing[:3]}"
        )
    result = encoder.load_state_dict(backbone, strict=False)
    if result.unexpected_keys:
        raise RgbnFailure(f"unexpected adapted keys: {result.unexpected_keys[:3]}")
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    return encoder, payload


def _fingerprint(
    manifest_sha256: str,
    source_sha256: str,
    original_sha256: str,
    *,
    task_id: str,
    profile: object,
    upstream_commit: str,
) -> str:
    payload = {
        "version": CACHE_VERSION,
        "task": task_id,
        "manifest": manifest_sha256,
        "source_model": source_sha256,
        "original_checkpoint": original_sha256,
        "bands": list(profile.bands),
        "max_frames": profile.max_frames,
        "preprocessing": profile.preprocessing,
        "grid": TOKENS,
        "reduction": "temporal-mean float32",
        "upstream": upstream_commit,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


class FusionCache:
    """Temporal-mean F, ``(1024, 16, 16)`` float32, fingerprinted."""

    def __init__(self, root: Path, fingerprint: str, encoder, mean, std, waves, device, profile):
        self.directory = root / "fusion" / "features" / fingerprint
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fingerprint = fingerprint
        self.encoder = encoder
        self.mean = mean
        self.std = std
        self.waves = waves
        self.device = device
        self.profile = profile
        self.hits = 0
        self.misses = 0

    def _paths(self, sample_id: str) -> tuple[Path, Path]:
        return self.directory / f"{sample_id}.npy", self.directory / f"{sample_id}.json"

    def _current(self, meta_path: Path) -> bool:
        if not meta_path.is_file():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return (
            meta.get("version") == CACHE_VERSION
            and meta.get("fingerprint") == self.fingerprint
            and meta.get("shape") == [TOKEN_DIM, TOKENS, TOKENS]
            and meta.get("dtype") == "float32"
        )

    def get(self, row: dict, data_root: Path) -> np.ndarray:
        import torch

        features_path, meta_path = self._paths(row["sample_id"])
        if self._current(meta_path):
            self.hits += 1
            return np.load(features_path)
        self.misses += 1
        pixels, dates, frames = read_frames(row, data_root, self.profile)
        channels = len(self.profile.bands)
        tensor = torch.from_numpy(pixels).to(self.device)
        tensor = (tensor - self.mean.to(self.device).view(1, channels, 1, 1)) / self.std.to(
            self.device
        ).view(1, channels, 1, 1)
        tokens = frame_tokens(self.encoder, tensor, self.waves.to(self.device))
        features = tokens.to(torch.float32).mean(dim=0).cpu().numpy()
        if features.shape != (TOKEN_DIM, TOKENS, TOKENS):
            raise RgbnFailure(f"F shape {features.shape} != (1024, 16, 16)")
        save_npy_atomic(features_path, features)
        meta = {
            "version": CACHE_VERSION,
            "fingerprint": self.fingerprint,
            "frames": frames,
            "dates": dates,
            "shape": list(features.shape),
            "dtype": "float32",
        }
        save_atomic(meta_path, json.dumps(meta, sort_keys=True).encode())
        return features


def _normalize_branch(z):
    import torch  # noqa: F401 - the module stays importable without torch

    return float(z.shape[1]) ** 0.5 * z / z.norm(dim=1, keepdim=True).clamp_min(1e-6)


def _head_input(features, projector):
    if projector is None:
        return features
    import torch

    return torch.cat([features, _normalize_branch(projector(features))], dim=1)


def _expand_head(base, wide):
    import torch

    with torch.no_grad():
        wide[0].weight.zero_()
        wide[0].weight[:, : base[0].in_channels].copy_(base[0].weight)
        wide[0].bias.copy_(base[0].bias)
        for index in range(1, len(base)):
            wide[index].load_state_dict(base[index].state_dict())
    return wide


def _permutation(length: int, seed: int, epoch: int) -> list[int]:
    import torch

    generator = torch.Generator().manual_seed(1_000 * seed + epoch)
    return torch.randperm(length, generator=generator).tolist()


def _validate_common(
    request,
    writer,
    task_spec,
    original_checkpoint: Path,
):
    import torch

    if task_spec.task_id != "T11D":
        raise RgbnFailure(f"fusion protocols only implement T11D, got {task_spec.task_id}")
    profile = BAND_PROFILES["T11D"]
    if list(task_spec.bands) != list(profile.bands):
        raise RgbnFailure(
            f"task declares {list(task_spec.bands)}, fusion profile has {list(profile.bands)}"
        )
    lock = _source_lock(request, request.seed, original_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder, payload = _load_adapted_encoder(
        original_checkpoint, Path(lock["adapted_model_pt"]), device
    )
    from clay_worker.clay_probe import add_upstream_to_path, load_sensor_metadata, upstream_commit

    source_root = add_upstream_to_path()
    metadata = load_sensor_metadata(source_root, "sentinel-2-l2a")
    mean = np.asarray([metadata["bands"]["mean"][band] for band in profile.clay_bands], np.float32)
    std = np.asarray([metadata["bands"]["std"][band] for band in profile.clay_bands], np.float32)
    waves = np.asarray(
        [metadata["bands"]["wavelength"][band] for band in profile.clay_bands], np.float32
    )
    fingerprint = _fingerprint(
        lock["manifest_sha256"] or "",
        lock["adapted_model_sha256"],
        lock["original_checkpoint_sha256"],
        task_id=task_spec.task_id,
        profile=profile,
        upstream_commit=upstream_commit(source_root)[:12],
    )
    cache = FusionCache(
        cache_root(),
        fingerprint,
        encoder,
        torch.from_numpy(mean),
        torch.from_numpy(std),
        torch.from_numpy(waves),
        device,
        profile,
    )
    writer.update_input(
        preprocessing_id=profile.preprocessing,
        bands=list(profile.bands),
        time_steps=profile.max_frames,
        gsd_m=GSD_M,
    )
    writer.update(
        tags={
            "fusion_role": "fit" if request.run_id == fit_run_id(request.seed) else "head",
            "source_run_id": lock["source_run_id"],
            "source_git_commit": lock["source_git_commit"],
            "source_protocol": SOURCE_PROTOCOL,
            "source_model_sha256": lock["adapted_model_sha256"],
            "original_checkpoint_sha256": lock["original_checkpoint_sha256"],
            "f_cache_fingerprint": fingerprint,
            "f_cache": str(cache.directory),
            "teacher": TEACHER_TARGET,
            "teacher_dims": str(TEACHER_DIMS),
            "token_grid": f"{TOKENS}x{TOKENS}",
            "token_stride_m": str(8 * GSD_M),
            "fusion_branch_normalization": "sqrt(128)*z/max(||z||_2,1e-6) over channels",
            "fusion_channel_order": "F(1024) then N(E(F))(128)",
            "panel": "diagnostic",
            "device": device,
        }
    )
    return profile, lock, encoder, mean, std, waves, device, cache, fingerprint


def _train_fit(request, writer, emit, task_spec, rows, cache, data_root, device, fingerprint, lock):
    import torch

    groups = _fusion_groups(rows)
    if len(groups["train"]) < _fusion_minimum_rows():
        raise RgbnFailure(
            f"train fold has {len(groups['train'])} rows; the protocol needs "
            f">= {_fusion_minimum_rows()}"
        )
    ignore = task_spec.ignore_index if task_spec.ignore_index is not None else 19
    run_dir = Path(request.runs_root) / request.run_id

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(PROJECTOR_SEED_OFFSET + request.seed)
        projector_init = build_projector(TOKEN_DIM, TEACHER_DIMS, PROJECTOR_HIDDEN).to(device)
        torch.manual_seed(HEAD_SEED_OFFSET + request.seed)
        head_base = build_head(TOKEN_DIM, task_spec.num_outputs, HEAD_HIDDEN).to(device)
    init_state = copy.deepcopy(projector_init.state_dict())
    head_state = copy.deepcopy(head_base.state_dict())

    projector = build_projector(TOKEN_DIM, TEACHER_DIMS, PROJECTOR_HIDDEN).to(device)
    projector.load_state_dict(init_state)
    optimiser = torch.optim.AdamW(
        projector.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    epochs = EPOCHS

    def f_tensor(row):
        return torch.from_numpy(cache.get(row, data_root)).unsqueeze(0).to(device)

    def teacher_pair(row):
        grid, support = teacher_tokens(TEACHER_TARGET, row, data_root)
        return (
            torch.from_numpy(grid).unsqueeze(0).to(device),
            torch.from_numpy(support).unsqueeze(0).to(device),
        )

    def evaluate(rows_):
        projector.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for row in rows_:
                features = f_tensor(row)
                target, support = teacher_pair(row)
                total += float(cosine_alignment(projector(features), target, support))
                count += 1
        projector.train()
        return total / max(count, 1)

    best_score = float("-inf")
    best_epoch = 0
    best_state = copy.deepcopy(init_state)
    started = time.time()
    updates_per_epoch = len(range(0, len(groups["train"]), ACCUMULATION))
    for epoch in range(epochs):
        order = _permutation(len(groups["train"]), request.seed, epoch)
        projector.train()
        optimiser.zero_grad(set_to_none=True)
        total, steps, skipped = 0.0, 0, 0
        for chunk in accumulation_chunks(order, ACCUMULATION):
            loss = torch.zeros((), device=device)
            valid = 0
            for index in chunk:
                row = groups["train"][index]
                if int((_labels(row, data_root, ignore) != ignore).sum()) == 0:
                    continue
                features = f_tensor(row)
                target, support = teacher_pair(row)
                loss = loss + (1.0 - cosine_alignment(projector(features), target, support))
                valid += 1
            if valid == 0:
                skipped += 1
                continue
            (loss / valid).backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            steps += 1
            total += float(loss.detach() / valid)
        if steps + skipped != updates_per_epoch:
            raise RgbnFailure(
                f"epoch {epoch + 1} took {steps} steps (+{skipped} empty) "
                f"for {len(order)} rows at effective batch {ACCUMULATION}"
            )
        validation = evaluate(groups["validation"])
        writer.add_curve("teacher_cosine", "validation", epoch + 1, validation)
        emit(
            request.run_id,
            "fusion.fit.epoch",
            f"epoch {epoch + 1}/{epochs}: loss {total / max(steps, 1):.4f}, "
            f"validation cosine {validation:.4f}",
            progress=(epoch + 1) / epochs,
        )
        if validation > best_score:
            best_score = validation
            best_epoch = epoch + 1
            best_state = copy.deepcopy(projector.state_dict())

    torch.save(
        {
            "state_dict": init_state,
            "shape": [TOKEN_DIM, TEACHER_DIMS],
            "hidden": PROJECTOR_HIDDEN,
            "seed": request.seed,
            "source_run_id": lock["source_run_id"],
            "f_cache_fingerprint": fingerprint,
        },
        run_dir / "projector_init.pt",
    )
    torch.save(
        {
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_validation_cosine": best_score,
            "shape": [TOKEN_DIM, TEACHER_DIMS],
            "hidden": PROJECTOR_HIDDEN,
            "seed": request.seed,
            "source_run_id": lock["source_run_id"],
            "f_cache_fingerprint": fingerprint,
        },
        run_dir / "projector_best.pt",
    )
    torch.save(
        {
            "head": head_state,
            "projector_init": init_state,
            "classes": task_spec.num_outputs,
            "head_hidden": HEAD_HIDDEN,
            "seed": request.seed,
            "source_run_id": lock["source_run_id"],
            "source_model_sha256": lock["adapted_model_sha256"],
            "f_cache_fingerprint": fingerprint,
        },
        run_dir / "fusion_init.pt",
    )
    report = {
        "best_epoch": best_epoch,
        "best_validation_cosine": best_score,
        "epochs": epochs,
        "train_rows": len(groups["train"]),
        "validation_rows": len(groups["validation"]),
        "test_rows": len(groups["test"]),
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
    }
    (run_dir / "fit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in ("projector_init.pt", "projector_best.pt", "fusion_init.pt", "fit_report.json"):
        writer.add_artifact("checkpoint" if name.endswith(".pt") else "report", name, name)
    writer.update_training(
        head_id="fusion_projector_oe128",
        epochs_completed=epochs,
        wall_time_seconds=round(time.time() - started, 1),
    )
    writer.complete(
        f"fusion fit: OE128 projector, best validation cosine {best_score:.4f} "
        f"at epoch {best_epoch}"
    )
    return 0


def _load_fit_artifacts(request, lock, fingerprint, task_spec, device):
    import torch

    parent = fit_parent_id(request)
    fit_dir = Path(request.runs_root) / parent
    record_path = fit_dir / "run.json"
    if not record_path.is_file():
        raise RgbnFailure(f"fusion fit run missing: {record_path}; run {FIT_PROTOCOL} first")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise RgbnFailure(f"fusion fit run {parent} is {record.get('status')!r}")
    init = torch.load(fit_dir / "fusion_init.pt", map_location="cpu", weights_only=False)
    if init.get("seed") != request.seed:
        raise RgbnFailure("fusion_init.pt seed mismatch")
    if init.get("source_run_id") != lock["source_run_id"]:
        raise RgbnFailure("fusion_init.pt source run mismatch")
    if init.get("f_cache_fingerprint") != fingerprint:
        raise RgbnFailure("fusion_init.pt F-cache fingerprint mismatch")
    head_base = build_head(TOKEN_DIM, task_spec.num_outputs, HEAD_HIDDEN).to(device)
    head_base.load_state_dict(init["head"])
    projector_init = build_projector(TOKEN_DIM, TEACHER_DIMS, PROJECTOR_HIDDEN).to(device)
    projector_init.load_state_dict(init["projector_init"])
    return fit_dir, head_base, projector_init


def _evaluate_branch(head, projector, groups, cache, data_root, task_spec, device, ignore):
    import torch
    import torch.nn.functional as F

    results = {}
    for split in ("validation", "test"):
        head.eval()
        confusion = np.zeros((task_spec.num_outputs, task_spec.num_outputs), dtype=np.int64)
        evaluated = 0
        with torch.no_grad():
            for row in groups[split]:
                evaluated += 1
                features = torch.from_numpy(cache.get(row, data_root)).unsqueeze(0).to(device)
                logits = head(_head_input(features, projector))
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
        results[split] = {
            "evaluated": evaluated,
            "mean_iou": float(np.nanmean(iou)),
            "macro_f1": _macro_f1(confusion),
        }
    return results


def _macro_f1(confusion: np.ndarray) -> float:
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    present = confusion.sum(axis=1) > 0
    f1 = np.where(present, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
    return float(np.nanmean(f1))


def _train_head(
    request,
    writer,
    emit,
    task_spec,
    rows,
    cache,
    data_root,
    device,
    fingerprint,
    lock,
    role: str,
):
    import torch
    import torch.nn.functional as F

    groups = _fusion_groups(rows)
    if len(groups["train"]) < _fusion_minimum_rows():
        raise RgbnFailure(
            f"train fold has {len(groups['train'])} rows; the protocol needs "
            f">= {_fusion_minimum_rows()}"
        )
    ignore = task_spec.ignore_index if task_spec.ignore_index is not None else 19
    run_dir = Path(request.runs_root) / request.run_id
    fit_dir, head_base, projector_init = _load_fit_artifacts(
        request, lock, fingerprint, task_spec, device
    )

    if role == "raw":
        head = head_base
        projector = None
    else:
        head = build_head(TOKEN_DIM + TEACHER_DIMS, task_spec.num_outputs, HEAD_HIDDEN).to(device)
        _expand_head(head_base, head)
        if role == "rand":
            projector = projector_init
        else:
            projector = build_projector(TOKEN_DIM, TEACHER_DIMS, PROJECTOR_HIDDEN).to(device)
            best = torch.load(fit_dir / "projector_best.pt", map_location="cpu", weights_only=False)
            if best.get("seed") != request.seed or best.get("f_cache_fingerprint") != fingerprint:
                raise RgbnFailure("projector_best.pt identity mismatch")
            projector.load_state_dict(best["state_dict"])
    for parameter in head.parameters():
        parameter.requires_grad = True
    if projector is not None:
        for parameter in projector.parameters():
            parameter.requires_grad = False

    optimiser = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    epochs = EPOCHS
    started = time.time()
    updates_per_epoch = len(range(0, len(groups["train"]), ACCUMULATION))
    for epoch in range(epochs):
        order = _permutation(len(groups["train"]), request.seed, epoch)
        head.train()
        if projector is not None:
            projector.eval()
        optimiser.zero_grad(set_to_none=True)
        total, steps, skipped = 0.0, 0, 0
        for chunk in accumulation_chunks(order, ACCUMULATION):
            loss = torch.zeros((), device=device)
            valid = 0
            for index in chunk:
                row = groups["train"][index]
                labels = (
                    torch.from_numpy(_labels(row, data_root, ignore)).to(device).long().unsqueeze(0)
                )
                if int((labels != ignore).sum()) == 0:
                    continue
                features = torch.from_numpy(cache.get(row, data_root)).unsqueeze(0).to(device)
                with torch.no_grad():
                    branch = _head_input(features, projector)
                logits = head(branch)
                logits = F.interpolate(
                    logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
                )
                loss = loss + F.cross_entropy(logits, labels, ignore_index=ignore)
                valid += 1
            if valid == 0:
                skipped += 1
                continue
            (loss / valid).backward()
            optimiser.step()
            optimiser.zero_grad(set_to_none=True)
            steps += 1
            total += float(loss.detach() / valid)
        if steps + skipped != updates_per_epoch:
            raise RgbnFailure(
                f"epoch {epoch + 1} took {steps} steps (+{skipped} empty) "
                f"for {len(order)} rows at effective batch {ACCUMULATION}"
            )
        writer.add_curve("loss", "train", epoch + 1, total / max(steps, 1))
        emit(
            request.run_id,
            "fusion.head.epoch",
            f"epoch {epoch + 1}/{epochs}: loss {total / max(steps, 1):.4f}, {steps} steps",
            progress=(epoch + 1) / epochs,
        )

    results = _evaluate_branch(head, projector, groups, cache, data_root, task_spec, device, ignore)
    from rs_backbone_eval.contracts import MetricRecord
    from rs_backbone_eval.matrix import OFFICIAL_TEST_SAMPLES, OFFICIAL_VALIDATION_SAMPLES

    for split in ("validation", "test"):
        expected = (
            OFFICIAL_TEST_SAMPLES[task_spec.task_id]
            if split == "test"
            else OFFICIAL_VALIDATION_SAMPLES[task_spec.task_id]
        )
        if (
            results[split]["evaluated"] != expected
            and os.environ.get("CLAY_WORKER_ALLOW_PARTIAL_EVAL") != "1"
        ):
            raise RgbnFailure(
                f"{split} fold evaluated on {results[split]['evaluated']} of "
                f"the official {expected} rows"
            )
        writer.add_metrics(
            [
                MetricRecord(
                    name="mean_iou",
                    direction=task_spec.primary_metric.direction,
                    value=round(results[split]["mean_iou"], 6),
                    split=split,
                    is_primary=split == "test",
                ),
                MetricRecord(
                    name="macro_f1",
                    direction=task_spec.primary_metric.direction,
                    value=round(results[split]["macro_f1"], 6),
                    split=split,
                    is_primary=False,
                ),
            ]
        )
        writer.update(
            tags={
                f"{split}_samples_evaluated": str(results[split]["evaluated"]),
                f"{split}_manifest_samples": str(len(groups[split])),
                f"{split}_expected_samples": str(expected),
            }
        )

    state = {
        "state_dict": {
            f"head.{key}": value.detach().cpu() for key, value in head.state_dict().items()
        },
        "branch": role,
        "classes": task_spec.num_outputs,
        "head_hidden": HEAD_HIDDEN,
        "input_channels": TOKEN_DIM + (TEACHER_DIMS if projector is not None else 0),
        "channel_order": "F(1024) then N(E(F))(128)" if projector is not None else "F(1024)",
        "normalization": "sqrt(128)*z/max(||z||_2,1e-6) over channels",
        "source_run_id": lock["source_run_id"],
        "source_model_sha256": lock["adapted_model_sha256"],
        "f_cache_fingerprint": fingerprint,
        "seed": request.seed,
        "epochs": epochs,
    }
    if projector is not None:
        state["state_dict"] |= {
            f"projector.{key}": value.detach().cpu()
            for key, value in projector.state_dict().items()
        }
        state["projector_init"] = role == "rand"
    torch.save(state, run_dir / "model.pt")
    writer.add_artifact("checkpoint", "model.pt", f"fusion {role} head and frozen projector")
    writer.update_training(
        head_id=f"fusion_{role}",
        trainable_parameters=sum(p.numel() for p in head.parameters()),
        epochs_completed=epochs,
        wall_time_seconds=round(time.time() - started, 1),
    )
    test = next(metric for metric in writer.record.metrics if metric.is_primary)
    writer.complete(
        f"fusion {role}: test mIoU {test.value:.4f}; F and projector frozen, "
        f"source {lock['source_run_id']}"
    )
    return 0


def train_fusion(request, writer, emit) -> int:
    from rs_backbone_eval.registry import ProtocolSpec, TaskSpec, load_yaml_spec

    run_id = request.run_id
    try:
        task_spec: TaskSpec = load_yaml_spec(Path(request.task_config_path), TaskSpec)
        protocol_spec: ProtocolSpec = load_yaml_spec(
            Path(request.protocol_config_path), ProtocolSpec
        )
        protocol_id = protocol_spec.protocol_id
        if protocol_id not in FUSION_PROTOCOLS:
            raise RgbnFailure(f"fusion module has no protocol {protocol_id!r}")
        original_checkpoint = _checkpoint(request, writer)
        rows = read_rows(Path(request.manifest_path))
        profile, lock, encoder, mean, std, waves, device, cache, fingerprint = _validate_common(
            request, writer, task_spec, original_checkpoint
        )
        data_root = Path(request.data_root)
        if protocol_id == FIT_PROTOCOL:
            writer.update(tags={"fusion_role": "fit"})
            return _train_fit(
                request, writer, emit, task_spec, rows, cache, data_root, device, fingerprint, lock
            )
        role = {
            RAW_PROTOCOL: "raw",
            RAND_PROTOCOL: "rand",
            TEACHER_PROTOCOL: "teacher",
        }[protocol_id]
        writer.update(tags={"fusion_role": "head", "fusion_branch": role})
        return _train_head(
            request,
            writer,
            emit,
            task_spec,
            rows,
            cache,
            data_root,
            device,
            fingerprint,
            lock,
            role,
        )
    except Exception as exc:  # noqa: BLE001 - worker must record every failure
        emit(run_id, "fusion.crash", f"{type(exc).__name__}: {exc}", level="error")
        writer.fail(f"{type(exc).__name__}: {exc}")
        return 1
