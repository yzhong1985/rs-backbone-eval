"""Stage the Colab-Drive upload pack for the LIM T11 20-bin lane.

Everything the Colab A100 notebook needs that CANNOT be sourced directly on
Colab: the LIM checkpoint, the repo git bundle, the contract manifest, the
per-patch sidecars (dates_s2) and the derived ANNOTATIONS payloads referenced
by the manifest. The ~45 GB ``DATA_S2`` stacks are downloaded on Colab from
the official ``IGNF/PASTIS-HD`` HuggingFace snapshot, so they are NOT copied;
instead the script hashes every S2 file locally into ``s2_sha256.json``
(review 2026-11-04: 5 anchors are not enough for a formal run - the notebook
verifies each downloaded file byte-for-byte against the full list).

Usage::

    uv run python scripts/pack_t11_drive.py --out /home/yzhong/rs-eval/drive-t11-pack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{what} not found: {path}")
    return path


def _zip_tree(root: Path, archive: Path, arc_prefix: str, pattern: str = "*") -> tuple[int, int]:
    """Zip every file under ``root`` as ``arc_prefix/<name>`` (flat layout)."""
    members = 0
    total = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as bundle:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            bundle.write(path, f"{arc_prefix}/{path.name}")
            members += 1
            total += path.stat().st_size
    if members == 0:
        raise SystemExit(f"no files matched {root} pattern {pattern}")
    return members, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--data-aef",
        type=Path,
        default=Path("/home/yzhong/rs-eval/data-aef"),
        help="local PASTIS-HD derived root (features sidecars + PASTIS-HD trees)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="contract manifest; defaults to the newest cf092238*.jsonl",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/geodata/rs-eval/weights/lim/mit_huge_600.pth"),
    )
    parser.add_argument(
        "--annotations-hashes-from",
        type=Path,
        default=None,
        help="reuse a previously computed annotations_sha256.json",
    )
    parser.add_argument(
        "--route-c",
        action="store_true",
        help="small pack only (manifest/sidecars/hashes/bundle): data comes "
        "from the hf-mirror and the LIM checkpoint is supplied separately",
    )
    parser.add_argument(
        "--s2-hashes-from",
        type=Path,
        default=None,
        help="reuse a previously computed s2_sha256.json instead of rehashing 45 GB",
    )
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing staging dir {out}")
    manifest_dir = Path("/geodata/rs-eval/data/manifests/T11")
    manifest = args.manifest or next(iter(sorted(manifest_dir.glob("cf092238*.jsonl"))))
    _require(manifest, "contract manifest")

    side_root = _require(args.data_aef / "features/2019", "sidecar dir")
    annot_root = _require(args.data_aef / "PASTIS-HD/ANNOTATIONS", "annotations dir")
    s2_root = _require(args.data_aef / "PASTIS-HD/DATA_S2", "DATA_S2 dir")

    out.mkdir(parents=True)

    # 1. checkpoint (big); route C keeps the 6.2 GB file out of the pack but
    #    still records its sha256 so the notebook can refuse a mismatched
    #    upload (review 2026-11-04 round 7).
    _require(args.checkpoint, "LIM checkpoint")
    checkpoint_sha = _sha256(args.checkpoint)
    if args.route_c:
        print(f"checkpoint kept out of the pack; sha256 {checkpoint_sha[:16]}...")
    else:
        lim_dir = out / "lim"
        lim_dir.mkdir()
        shutil.copy2(args.checkpoint, lim_dir / args.checkpoint.name)

    # 2. repo bundle at the current commit (pin for dispatch on Colab);
    # the pinned HEAD goes into PROVENANCE so the notebook can assert it
    bundle = out / "repo" / "rs-backbone-eval.bundle"
    bundle.parent.mkdir()
    subprocess.run(
        ["git", "bundle", "create", bundle.as_posix(), "--all"],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    repo_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # 3. contract manifest
    manifest_dir_out = out / "manifest"
    manifest_dir_out.mkdir()
    shutil.copy2(manifest, manifest_dir_out / manifest.name)
    manifest_sha = _sha256(manifest)

    # 4. sidecars: only the dates_s2 jsons (the co-located AEF .npz stay local;
    # the LIM T11 variants never open them, and Drive upload would balloon 51GB)
    side_stats = _zip_tree(
        side_root, out / "sidecars.zip", "derived/aef-2019/features/2019", "*.json"
    )

    # 5. derived annotations: route C reads them from the hf-mirror, so the
    #    zip is skipped (it would add 280 MB to a ~12 MB pack)
    annot_stats = None
    if not args.route_c:
        annot_stats = _zip_tree(
            annot_root,
            out / "annotations.zip",
            "derived/aef-2019/PASTIS-HD/ANNOTATIONS",
            "*.npy",
        )

    # 6. per-file sha256 for everything the notebook downloads from HF
    annotations_hashes: dict[str, str] = {}
    if args.annotations_hashes_from is not None:
        _require(args.annotations_hashes_from, "reused annotations sha256 list")
        annotations_hashes = json.loads(args.annotations_hashes_from.read_text(encoding="utf-8"))
        if not annotations_hashes:
            raise SystemExit("the reused annotations sha256 list is empty")
    else:
        for path in sorted(annot_root.glob("*.npy")):
            annotations_hashes[path.name] = _sha256(path)
    (out / "annotations_sha256.json").write_text(
        json.dumps(annotations_hashes, indent=0, sort_keys=True), encoding="utf-8"
    )

    if args.s2_hashes_from is not None:
        _require(args.s2_hashes_from, "reused s2 sha256 list")
        hashes = json.loads(args.s2_hashes_from.read_text(encoding="utf-8"))
        if not hashes:
            # an empty list would let the notebook "verify" nothing; refuse it
            raise SystemExit("the reused s2 sha256 list is empty; refusing to pack it")
    else:
        hashes = {}
        for path in sorted(s2_root.glob("S2_*.npy")):
            hashes[path.name] = _sha256(path)
    (out / "s2_sha256.json").write_text(
        json.dumps(hashes, indent=0, sort_keys=True), encoding="utf-8"
    )

    provenance = {
        "repo_commit": repo_head,
        "manifest": {"path": manifest.name, "sha256": manifest_sha},
        "checkpoint": {
            "path": args.checkpoint.name,
            "sha256": checkpoint_sha,
            "route_c_supplied_separately": bool(args.route_c),
        },
        "sidecars_zip": {"members": side_stats[0], "bytes": side_stats[1]},
        "annotations_zip": (
            {"members": annot_stats[0], "bytes": annot_stats[1]}
            if annot_stats
            else {"skipped_route_c": True}
        ),
        "s2_sha256_list": {"files": len(hashes), "verified_pack": bool(len(hashes))},
        "annotations_sha256_list": {"files": len(annotations_hashes)},
        "route_c": bool(args.route_c),
        "hf_source": {
            "repo_id": "IGNF/PASTIS-HD",
            "endpoint_env": "HF_ENDPOINT",
            "endpoint_default": "https://hf-mirror.com",
            "patterns": ["DATA_S2/*", "ANNOTATIONS/*", "documentation/*"],
            "note": (
                "verify every DATA_S2 file sha256 against s2_sha256.json after "
                "snapshot_download (review 2026-11-04: not just 5 anchors); "
                "DATA_S2 must land at data_root/derived/aef-2019/PASTIS-HD/DATA_S2"
            ),
        },
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"staged {out} (s2 hashes: {len(hashes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
