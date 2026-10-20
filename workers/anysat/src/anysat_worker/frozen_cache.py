"""Disk-backed frozen-feature cache for the AnySat worker frozen tracks.

Audit rule (docs/reviews/评测框架复审_3b4460a.md): a feature cache's identity
must bind checkpoint, preprocessing, bands, dates, crop and the data version —
otherwise caches from different input protocols cross-contaminate. The key
below hashes everything the review lists plus a fingerprint of the worker
source files that define feature semantics (t5 + spectral adapter) so code
that changes the features rotates the cache directory atomically. This
module's own I/O (storage dtype, retry) is deliberately excluded so a pure
read-path fix never invalidates a 400 GB cache.

Layout: ``<cache_root>/anysat_t5_features/<sha16>/{split}/<row>.npz`` with an
uncompressed fp16 ``dense`` array (fast sequential reads: no codec on the hot
path) and an int16 ``label``. Row indexing follows the manifest order of the
dataset rows, so cache entries stay aligned with labels/group ids.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def state_identity_hash(
    model: nn.Module,
    *,
    prefixes: tuple[str, ...] = ("adapter.", "backbone."),
    chunk: int = 1 << 20,
) -> str:
    """Streamed sha256 over the fixed feature extractor's actual weights.

    Binds what put features on disk: the adapter and backbone state_dicts
    (params AND buffers). The task head keeps training across stage epochs, so
    binding it would recreate the cache after every resume; the review
    (#3) therefore pins the extractor only.
    """
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith(tuple(prefixes)):
            continue
        digest.update(name.encode("utf-8"))
        buffer = tensor.detach().cpu().contiguous().numpy().tobytes()
        for offset in range(0, len(buffer), chunk):
            digest.update(buffer[offset : offset + chunk])
    return digest.hexdigest()


def source_fingerprint(paths: list[Path]) -> str:
    """Worker source fingerprint so code semantics participate in the key."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.read_bytes())
    return digest.hexdigest()


class FrozenFeatureCache:
    """NumPy npz per manifest row, keyed by the full cache identity."""

    def __init__(
        self,
        *,
        cache_root: Path,
        identity_parts: dict[str, object],
        worker_sources: list[Path],
        split: str,
    ) -> None:
        payload_id = {
            **identity_parts,
            "sources": source_fingerprint(worker_sources),
            "split": split,
        }
        blob = json.dumps(payload_id, sort_keys=True, default=str)
        self.directory = (
            cache_root
            / "anysat_t5_features"
            / hashlib.sha256(blob.encode()).hexdigest()[:16]
            / split
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self.marker = self.directory / "identity.json"
        text = json.dumps(payload_id, sort_keys=True, default=str, indent=1)
        if self.marker.exists():
            if self.marker.read_text(encoding="utf-8") != text:
                raise ValueError("cache identity hash collision; purge the cache directory")
        else:
            self.marker.write_text(text, encoding="utf-8")

    def path(self, row_index: int) -> Path:
        return self.directory / f"{row_index:06d}.npz"

    def has(self, row_index: int) -> bool:
        return self.path(row_index).is_file()

    def load(self, row_index: int, *, attempts: int = 4) -> dict[str, torch.Tensor]:
        """Read one cached row, retrying transient corruption.

        An 18 h two-stage run died at epoch 69 on a single ``BadZipFile``
        (CRC-32) read while a full scan of the same cache found 0 bad files —
        a transient fault, not a poisoned cache. Re-open and retry a few times;
        a persistent failure names the row so an operator can delete it (on
        resume, ``populate`` re-extracts every missing row).
        """
        path = self.path(row_index)
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                with np.load(path) as block:
                    dense = block["dense"].astype(np.float32)
                    label = block["label"].astype(np.int64)
                return {
                    "dense": torch.from_numpy(dense),
                    "label": torch.from_numpy(label),
                }
            except (zipfile.BadZipFile, OSError, EOFError, ValueError) as exc:
                last = exc
                if attempt + 1 < attempts:
                    time.sleep(0.2 * (attempt + 1))
        raise RuntimeError(
            f"cache row {row_index} at {path} is unreadable after {attempts} "
            f"attempts ({type(last).__name__}: {last}); delete that row and "
            "resume so populate re-extracts it"
        ) from last

    def save(self, row_index: int, dense: torch.Tensor, label: torch.Tensor) -> None:
        # fp16 storage is lossless for this cache: the online dense is bf16
        # under autocast (7 mantissa bits), and fp16 carries 10 — every bf16
        # value in the observed range (~+/-5) is exactly representable. A full
        # forward-equivalence rerun confirmed fp16 and fp32 storage give
        # identical results, and fp16 halves the on-disk dense (per-row ~154 MB
        # vs ~308 MB: the full T5 train split is ~404 GB vs ~808 GB per seed).
        block = dense.detach().cpu()
        peak = block.abs().max().item()
        if peak > 65504.0:
            # fp16 max: a larger value would silently become inf on disk
            raise ValueError(f"dense peak {peak:.1f} exceeds the fp16 range")
        dense_block = block.to(torch.float16).numpy()
        label_block = label.detach().cpu().numpy().astype(np.int16)
        destination = self.path(row_index)
        with tempfile.NamedTemporaryFile(dir=self.directory, suffix=".tmp", delete=False) as handle:
            np.savez(handle, dense=dense_block, label=label_block)
        os.replace(handle.name, destination)

    def coverage(self, total: int) -> int:
        return min(total, sum(1 for index in range(total) if self.has(index)))


class CachedSampleDataset(Dataset):
    """Per-row tensors straight from the cache; the rasters are never touched."""

    def __init__(self, cache: FrozenFeatureCache, total: int) -> None:
        self.cache = cache
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        block = self.cache.load(index)
        return {"dense": block["dense"], "label": block["label"]}


def _rng_capture() -> dict[str, object]:
    import random

    rng_state = np.random.get_state()
    captured = {
        "python": random.getstate(),
        "numpy": (rng_state[0], rng_state[1].tolist(), *rng_state[2:]),
        "torch": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        # review: GPU extraction draws from the CUDA streams as well
        captured["cuda"] = torch.cuda.get_rng_state_all()
    return captured


def _rng_restore(captured: dict[str, object]) -> None:
    import random

    cuda = captured.get("cuda", [])
    random.setstate(captured["python"])
    kind, values, pos, has_gauss, cached = captured["numpy"]
    np.random.set_state((kind, np.asarray(values, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(captured["torch"])
    if isinstance(cuda, list) and cuda:
        torch.cuda.set_rng_state_all(cuda)


def isolated_training_rng() -> object:
    """Context manager: extraction must not consume the training random state.

    The review flagged the reproducibility split between the first build (the
    extraction loader eats RNG, shifting every later epoch) and a full cache
    hit (no consumption). Restoring the streams keeps both identical.
    """
    from contextlib import contextmanager

    @contextmanager
    def _impl():
        capture = _rng_capture()
        try:
            yield
        finally:
            _rng_restore(capture)

    return _impl()


def populate(
    cache: FrozenFeatureCache,
    dataset,
    model: torch.nn.Module,
    means: torch.Tensor,
    stds: torch.Tensor,
    run_id: str,
    emit: Callable[..., None],
    device: str,
    *,
    batch_size: int = 4,
    max_rows: int | None = None,
) -> bool:
    """Materialise the fused dense map + labels for rows of ``dataset``.

    Uses the exact autocast-bf16 context of the direct training path so the
    cached fp16 features sit one rounding step from the on-the-fly value.
    Complete cache hits return immediately (review #2d); a partial hit
    extracts only missing rows. The model is switched to eval() for the pass
    (fixed features, review #2c) and restored afterwards.
    """
    from torch.utils.data import DataLoader

    total = len(dataset)
    if max_rows is not None:
        total = min(total, max_rows)
    if cache.coverage(total) == total:
        emit(run_id, "train.cache", f"complete cache hit ({total} rows); skipping extraction")
        return total
    was_training = model.training
    model.eval()
    try:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        row_base = 0
        with isolated_training_rng(), torch.no_grad():
            for batch in loader:
                if row_base >= total:
                    break
                take = min(batch["images"].shape[0], total - row_base)
                row_indices = list(range(row_base, row_base + take))
                labels_cpu = batch["label"][:take]
                # review #2d: extract ONLY the missing rows of this batch —
                # rows already on disk are neither recomputed nor rewritten.
                missing_positions = [i for i, r in enumerate(row_indices) if not cache.has(r)]
                if missing_positions:
                    missing = torch.tensor(missing_positions)
                    images = (batch["images"][:take][missing].to(device) - means) / stds
                    dates = batch["dates"][:take][missing].to(device).float()
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"
                    ):
                        dense = model.fused_features(images, dates)
                    dense_cpu = dense.detach().cpu()
                    for offset, position in enumerate(missing_positions):
                        cache.save(
                            row_indices[position],
                            dense_cpu[offset],
                            labels_cpu[position],
                        )
                row_base += take
                emit(
                    run_id,
                    "train.cache",
                    f"frozen features cached {row_base}/{total}",
                    progress=0.3 * row_base / max(total, 1),
                )
    finally:
        if was_training:
            model.train()
    return cache.coverage(total)
