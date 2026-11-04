"""Stage the Colab-Drive upload pack for the LIM T11 20-bin lane.

Everything the Colab A100 notebook needs that CANNOT be sourced directly on
Colab: the LIM checkpoint, the repo git bundle, the contract manifest, the
per-patch sidecars (dates_s2) and the derived ANNOTATIONS payloads referenced
by the manifest. The 45 GB ``DATA_S2`` stacks are downloaded on Colab from the
official ``IGNF/PASTIS-HD`` HuggingFace snapshot, so they are NOT copied here;
instead this script records sha256 trust anchors for a handful of local S2
files so the notebook can verify the HF download byte-for-byte without
network access to compare with.

Output layout (staging dir, user uploads to Drive and mounts at
``/content/drive/MyDrive/rs-t11-pack``)::

    lim/mit_huge_600.pth
    repo/rs-backbone-eval.bundle
    manifest/T11_<sha12>.jsonl
    sidecars/derived/aef-2019/features/2019/<id>.json   (zip)
    annotations/derived/aef-2019/PASTIS-HD/ANNOTATIONS/* (zip)
    PROVENANCE.md

Usage: uv run python scripts/pack_t11_drive.py --out /home/yzhong/rs-eval/drive-t11-pack
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
_MANIFEST_DIR = Path("/geodata/rs-eval/data/manifests/T11")
MANIFEST = next(iter(sorted(_MANIFEST_DIR.glob("cf092238*.jsonl"))))
CHECKPOINT = Path("/geodata/rs-eval/weights/lim/mit_huge_600.pth")
SIDE_ROOT = Path("/home/yzhong/rs-eval/data-aef/features/2019")
ANNOT_ROOT = Path("/home/yzhong/rs-eval/data-aef/PASTIS-HD/ANNOTATIONS")
# trust anchors are derived from the manifest itself (patch ids are not
# contiguous) — see main().


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    return members, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing staging dir {out}")
    out.mkdir(parents=True)

    # 1. checkpoint (big)
    lim_dir = out / "lim"
    lim_dir.mkdir()
    shutil.copy2(CHECKPOINT, lim_dir / CHECKPOINT.name)

    # 2. repo bundle at the current commit (pin for dispatch on Colab)
    bundle = out / "repo" / "rs-backbone-eval.bundle"
    bundle.parent.mkdir()
    subprocess.run(
        ["git", "bundle", "create", bundle.as_posix(), "--all"],
        cwd=REPO,
        check=True,
        capture_output=True,
    )

    # 3. contract manifest
    manifest_dir = out / "manifest"
    manifest_dir.mkdir()
    shutil.copy2(MANIFEST, manifest_dir / MANIFEST.name)
    manifest_sha = _sha256(MANIFEST)

    # 4. sidecars: only the dates_s2 jsons (the co-located AEF .npz stay local;
    # the LIM T11 variants never open them, and Drive upload would balloon 51GB)
    sidecars = out / "sidecars.zip"
    side_stats = _zip_tree(SIDE_ROOT, sidecars, "derived/aef-2019/features/2019", "*.json")

    # 5. derived annotations referenced as label_path / parcel_path
    annotations = out / "annotations.zip"
    annot_stats = _zip_tree(ANNOT_ROOT, annotations, "derived/aef-2019/PASTIS-HD/ANNOTATIONS")

    # 6. provenance: trust anchors for the DATA_S2 trees the notebook downloads
    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    quants = (0.0, 0.25, 0.5, 0.75, 1.0)
    anchor_ids = {rows[int(frac * (len(rows) - 1))]["sample_id"] for frac in quants}
    anchors = {}
    for sample_id in sorted(anchor_ids):
        stack = ANNOT_ROOT.parent / "DATA_S2" / f"S2_{sample_id}.npy"
        if stack.is_file():
            anchors[f"S2_{sample_id}.npy"] = {
                "sha256": _sha256(stack),
                "bytes": stack.stat().st_size,
            }
    if len(anchors) < len(anchor_ids):
        missing = [sid for sid in sorted(anchor_ids) if f"S2_{sid}.npy" not in anchors]
        raise SystemExit(f"manifest sample missing from DATA_S2: {missing}")

    provenance = {
        "manifest": {"path": MANIFEST.name, "sha256": manifest_sha},
        "checkpoint": {"path": CHECKPOINT.name, "sha256": _sha256(CHECKPOINT)},
        "sidecars_zip": {"members": side_stats[0], "bytes": side_stats[1]},
        "annotations_zip": {"members": annot_stats[0], "bytes": annot_stats[1]},
        "s2_trust_anchors": anchors,
        "hf_source": {
            "repo_id": "IGNF/PASTIS-HD",
            "patterns": ["DATA_S2/*", "ANNOTATIONS/*", "documentation/*"],
            "note": (
                "verify each anchor file sha256 after snapshot_download; "
                "DATA_S2 must land at data_root/derived/aef-2019/PASTIS-HD/DATA_S2"
            ),
        },
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"staged {out} (s2 anchors: {len(anchors)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
