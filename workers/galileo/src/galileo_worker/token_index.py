"""Write the provenance index for a split of Galileo token files.

usage: python -m galileo_worker.token_index --manifest M --split train --out O \
        --encoder-dir E --map nir_to_b8.json --norm normalization.json [--pooled]

Reads every OUT/<split>/<sample_id>.pt that the manifest lists for the split and writes
OUT/index_<split>.json: the SHA-256 of each file, the hashes of the encoder weights,
config, band map, normalisation JSON and manifest, and a self-digest over all of it.
The LIM worker checks this index before reading tokens (lim_worker.galileo_index).
The index format is INDEX_FORMAT in lim_worker/galileo_index.py; keep them in step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from galileo_worker.extract_tokens import load_rows
from rs_backbone_eval.data.manifest import manifest_sha256

INDEX_FORMAT = "galileo-token-index-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def index_digest(payload: dict) -> str:
    """Same canonical form as lim_worker.galileo_index.index_digest."""
    core = {k: payload[k] for k in ("format", "split", "meta", "files")}
    text = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_index(
    *,
    manifest: Path,
    split: str,
    out: Path,
    encoder_dir: Path,
    map_json: Path,
    norm_json: Path,
    pooled: bool,
) -> Path:
    rows = load_rows(manifest, split)
    if not rows:
        raise ValueError(f"the manifest has no T13 {split} tiles")
    split_dir = out / split
    files: dict[str, str] = {}
    for row in rows:
        token_file = split_dir / f"{row['sample_id']}.pt"
        if not token_file.is_file():
            raise FileNotFoundError(f"missing token file {token_file}; extraction is incomplete")
        files[row["sample_id"]] = _sha256(token_file)
    meta = {
        "encoder_sha256": _sha256(encoder_dir / "encoder.pt"),
        "config_sha256": _sha256(encoder_dir / "config.json"),
        "map_sha256": _sha256(map_json),
        "norm_sha256": _sha256(norm_json),
        "manifest_sha256": manifest_sha256(manifest),
        "pooled": bool(pooled),
        "dtype": "float16",
        "group": "S2_NIR_10m",
    }
    payload = {"format": INDEX_FORMAT, "split": split, "meta": meta, "files": files}
    payload["digest"] = index_digest(payload)
    path = out / f"index_{split}.json"
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(f"indexed {len(files)} {split} tokens: digest {payload['digest']}")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--encoder-dir", type=Path, required=True)
    ap.add_argument("--map", dest="map_json", type=Path, required=True)
    ap.add_argument("--norm", dest="norm_json", type=Path, required=True)
    ap.add_argument("--pooled", action="store_true", help="the tokens are the 32x32 pooled grid")
    args = ap.parse_args()
    build_index(
        manifest=args.manifest,
        split=args.split,
        out=args.out,
        encoder_dir=args.encoder_dir,
        map_json=args.map_json,
        norm_json=args.norm_json,
        pooled=args.pooled,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
