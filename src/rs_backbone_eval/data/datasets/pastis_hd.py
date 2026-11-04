"""PASTIS-HD manifest builder for T11 (AEF annual-embedding track).

Rows carry a precomputed AEF 2019 embedding grid as the primary input
(``image_paths[0]`` resolves to ``.npz`` with features/valid members; raw int8
values stored as float16, NODATA -128, and the sidecar declares
``quantization``), the semantic mask as ``label_path`` (channel 0, classes
0-19 with void 19 ignored), and the parcel raster under ``aux_paths`` for the
single-pane ablations. The contract bands are the 64 AEF embedding channels,
not sensor bands; the task declares ``input_kind: aef_features``.

Manifest paths are relative to the data root; the extraction tree
(``/…/data-aef``) is expected to be linked under
``<data_root>/derived/aef-2019`` (features/ + PASTIS-HD/ANNOTATIONS).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ...contracts import ManifestSample
from ...registry import TaskSpec

BAND_NAMES = [f"AEF_{index:02d}" for index in range(64)]
GRID = 128
GSD_M = 10.0
TIMESTAMP = "2019"
DECLARED_ENCODING = "raw_aef_int8_as_float16"
SPLIT_BY_FOLD = {1: "train", 2: "train", 3: "train", 4: "validation", 5: "test"}


class ManifestBuildError(ValueError):
    """Raised when the local AEF payload cannot produce a complete manifest."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, data_root: Path) -> str:
    # Through the derived/aef-2019 symlink: os.path paths keep the logical
    # location so the manifest stays inside the data root, while the physical
    # payload may live on another filesystem.
    base = Path(os.path.abspath(data_root))
    target = Path(os.path.abspath(path))
    return target.relative_to(base).as_posix()


def build_pastis_hd_manifest(data_root: Path, task_spec: TaskSpec) -> list[ManifestSample]:
    if task_spec.input_kind != "aef_features":
        raise ManifestBuildError(f"task {task_spec.task_id} must declare input_kind: aef_features")
    base = data_root / "derived" / "aef-2019"
    features_dir = base / "features" / "2019"
    annotations = base / "PASTIS-HD" / "ANNOTATIONS"
    if not features_dir.is_dir() or not annotations.is_dir():
        raise ManifestBuildError(
            f"missing AEF payload under {base}; link the extraction tree first"
        )
    samples: list[ManifestSample] = []
    for sidecar in sorted(features_dir.glob("*.json"), key=lambda path: int(path.stem)):
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        patch = int(meta["patch"])
        voxel = features_dir / f"{patch}.npz"
        target = annotations / f"TARGET_{patch}.npy"
        parcel = annotations / f"ParcelIDs_{patch}.npy"
        if not voxel.is_file() or not target.is_file():
            raise ManifestBuildError(f"patch {patch}: extraction output incomplete")
        if meta.get("quantization") != DECLARED_ENCODING:
            raise ManifestBuildError(
                f"patch {patch}: sidecar encoding {meta.get('quantization')!r} "
                f"is not the declared {DECLARED_ENCODING!r}"
            )
        if meta.get("cog_sha256") is None:
            raise ManifestBuildError(f"patch {patch}: sidecar has no COG provenance")
        samples.append(
            ManifestSample(
                task_id=task_spec.task_id,
                sample_id=str(patch),
                group_id=str(patch),
                split=SPLIT_BY_FOLD[int(meta["fold"])],
                image_paths=[_relative(voxel, data_root)],
                label_path=_relative(target, data_root),
                aux_paths={
                    # Primary payload is the AEF grid; the aux slots carry the
                    # raw-modal inputs the T11 variants need: the LIM S2 stack
                    # (T,10,128,128; the worker picks year-mid / temporal-median
                    # frames) and the parcel raster for the per-scene ablations.
                    "s2_path": _relative(
                        base / "PASTIS-HD" / "DATA_S2" / f"S2_{patch}.npy", data_root
                    ),
                    "frame_date_anchor": "20190715",
                    "parcel_path": _relative(parcel, data_root),
                    "tile": str(meta["tile"]),
                    "label_year": TIMESTAMP,
                    "encoding": meta["quantization"],
                    "script_commit": meta.get("script_commit") or "",
                    "index_sha256": meta.get("index_sha256") or "",
                },
                sensor=task_spec.sensor,
                bands=list(BAND_NAMES),
                timestamps=[TIMESTAMP],
                temporal_mask=[True],
                gsd_m=GSD_M,
                checksum=_sha256(voxel),
                window=[0, 0, GRID, GRID],
                valid_fraction=float(meta.get("valid_share", 1.0)),
            )
        )
    if not samples:
        raise ManifestBuildError(f"no patches assembled under {features_dir}")
    return samples
