"""PASTIS-HD manifest builder for T11 (AEF) and T11O (OlmoEarth v1.3).

Rows carry a precomputed annual embedding grid as the primary input
(``image_paths[0]`` resolves to ``.npz`` with features/valid members, NODATA
-128, and the sidecar declares ``quantization``), the semantic mask as
``label_path`` (channel 0, classes 0-19 with void 19 ignored), and the parcel
raster under ``aux_paths`` for the single-pane ablations.

Two products share the task geometry and folds:

* ``aef_features`` (T11): 64 AEF channels, raw int8 stored as float16,
  encoding ``raw_aef_int8_as_float16``;
* ``embedding_features`` (T11O): 64 OlmoEarth v1.3 channels, raw int8,
  encoding ``olmoearth_int8``.

Manifest paths are relative to the data root; the extraction trees are
expected to be linked under ``<data_root>/derived/aef-2019`` and
``<data_root>/derived/olmoearth-2019`` (features/ + PASTIS-HD/ANNOTATIONS).
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
AEF_ENCODING = "raw_aef_int8_as_float16"
OLMOEARTH_ENCODING = "olmoearth_int8"
SPLIT_BY_FOLD = {1: "train", 2: "train", 3: "train", 4: "validation", 5: "test"}


class ManifestBuildError(ValueError):
    """Raised when the local embedding payload cannot produce a complete manifest."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, data_root: Path) -> str:
    # Through the derived/* symlink: os.path paths keep the logical location so
    # the manifest stays inside the data root, while the physical payload may
    # live on another filesystem.
    base = Path(os.path.abspath(data_root))
    target = Path(os.path.abspath(path))
    return target.relative_to(base).as_posix()


def build_pastis_hd_manifest(data_root: Path, task_spec: TaskSpec) -> list[ManifestSample]:
    if task_spec.input_kind == "aef_features":
        return _build(
            data_root,
            task_spec,
            tree="aef-2019",
            encoding=AEF_ENCODING,
            bands=BAND_NAMES,
            sensor=task_spec.sensor,
            provenance_key="cog_sha256",
        )
    if task_spec.input_kind == "embedding_features":
        dims = len(task_spec.bands)
        if dims == 64:
            tree = "olmoearth-2019"
        elif dims == 128:
            # The 128-dim cache lives in features/2019_128; the derived tree
            # links features/2019 to it so the same _build path works.
            tree = "olmoearth-2019-128"
        else:
            raise ManifestBuildError(
                f"task {task_spec.task_id}: OlmoEarth bands must be 64 or 128, got {dims}"
            )
        return _build(
            data_root,
            task_spec,
            tree=tree,
            encoding=OLMOEARTH_ENCODING,
            bands=[f"OLE_{index:02d}" for index in range(dims)],
            sensor=task_spec.sensor,
            provenance_key="sha256",
        )
    raise ManifestBuildError(
        f"task {task_spec.task_id}: PASTIS-HD needs input_kind aef_features "
        f"or embedding_features, got {task_spec.input_kind!r}"
    )


def _build(
    data_root: Path,
    task_spec: TaskSpec,
    *,
    tree: str,
    encoding: str,
    bands: list[str],
    sensor: str,
    provenance_key: str,
) -> list[ManifestSample]:
    base = data_root / "derived" / tree
    features_dir = base / "features" / "2019"
    annotations = base / "PASTIS-HD" / "ANNOTATIONS"
    if not features_dir.is_dir():
        raise ManifestBuildError(
            f"missing embedding payload under {features_dir}; link the extraction tree first"
        )
    if not annotations.is_dir():
        raise ManifestBuildError(f"missing PASTIS-HD annotations under {annotations}")
    samples: list[ManifestSample] = []
    for sidecar in sorted(features_dir.glob("*.json"), key=lambda path: int(path.stem)):
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        patch = int(meta["patch"])
        voxel = features_dir / f"{patch}.npz"
        target = annotations / f"TARGET_{patch}.npy"
        parcel = annotations / f"ParcelIDs_{patch}.npy"
        if not voxel.is_file() or not target.is_file():
            raise ManifestBuildError(f"patch {patch}: extraction output incomplete")
        if meta.get("quantization") != encoding:
            raise ManifestBuildError(
                f"patch {patch}: sidecar encoding {meta.get('quantization')!r} "
                f"is not the declared {encoding!r}"
            )
        if not meta.get(provenance_key):
            raise ManifestBuildError(
                f"patch {patch}: sidecar has no {provenance_key!r} payload provenance"
            )
        if int(meta.get("dims", 0)) != len(bands):
            raise ManifestBuildError(
                f"patch {patch}: sidecar dims {meta.get('dims')} != task bands {len(bands)}"
            )
        if not meta.get("tile"):
            raise ManifestBuildError(f"patch {patch}: sidecar has no tile/zone provenance")
        samples.append(
            ManifestSample(
                task_id=task_spec.task_id,
                sample_id=str(patch),
                group_id=str(patch),
                split=SPLIT_BY_FOLD[int(meta["fold"])],
                image_paths=[_relative(voxel, data_root)],
                label_path=_relative(target, data_root),
                aux_paths={
                    # Primary payload is the annual embedding grid; the aux
                    # slots carry the raw-modal inputs the T11 variants need:
                    # the LIM S2 stack (T,10,128,128) and the parcel raster.
                    "s2_path": _relative(
                        base / "PASTIS-HD" / "DATA_S2" / f"S2_{patch}.npy", data_root
                    ),
                    "frame_date_anchor": "20190715",
                    "parcel_path": _relative(parcel, data_root),
                    "tile": str(meta["tile"]),
                    "label_year": TIMESTAMP,
                    "encoding": encoding,
                    "script_commit": meta.get("script_commit") or "",
                    "index_sha256": meta.get("index_sha256") or "",
                },
                sensor=sensor,
                bands=list(bands),
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
