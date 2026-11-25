"""Provenance for the precomputed Galileo token cache used by the residual branch.

The token cache is an input to training and to the locked evaluation, so the ledger must
name the exact bytes each one read. Each split directory carries ``index_<split>.json``,
written by ``galileo_worker.token_index``: the SHA-256 of every token file plus the
encoder, band-map, normalisation and manifest hashes that produced them. This module
reads that index, checks its self-digest, and checks every token file a run reads.

The index format is shared with the galileo worker; this module only reads it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

INDEX_FORMAT = "galileo-token-index-v1"


def index_path(token_dir: Path, split: str) -> Path:
    return Path(token_dir) / f"index_{split}.json"


def index_digest(payload: dict) -> str:
    """SHA-256 over the canonical JSON of format, split, meta and per-file hashes."""
    core = {k: payload[k] for k in ("format", "split", "meta", "files")}
    text = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_index(token_dir: Path, split: str) -> dict:
    """Read a split's index and check its format, split and self-digest."""
    path = index_path(token_dir, split)
    if not path.is_file():
        raise FileNotFoundError(f"no Galileo token index at {path}; build it before use")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != INDEX_FORMAT:
        raise RuntimeError(f"{path} has format {payload.get('format')!r}, not {INDEX_FORMAT!r}")
    if payload.get("split") != split:
        raise RuntimeError(f"{path} indexes split {payload.get('split')!r}, not {split!r}")
    if payload.get("digest") != index_digest(payload):
        raise RuntimeError(f"{path} digest does not match its content")
    return payload


def verify_token_index(token_dir: Path, split: str, rows: list[dict]) -> str:
    """Check the split's index and every token file that ``rows`` will read.

    Raises when a row has no entry or a token file's bytes differ from the indexed
    SHA-256. Returns the index digest, which train and evaluate record in the ledger.
    """
    payload = load_index(token_dir, split)
    files = payload["files"]
    for row in rows:
        sample = row["sample_id"]
        if sample not in files:
            raise RuntimeError(f"token index for {split} has no entry for sample {sample}")
        token_file = Path(token_dir) / split / f"{sample}.pt"
        if not token_file.is_file():
            raise FileNotFoundError(f"missing Galileo token file {token_file}")
        if file_sha256(token_file) != files[sample]:
            raise RuntimeError(f"token file {token_file} differs from its indexed SHA-256")
    return payload["digest"]
