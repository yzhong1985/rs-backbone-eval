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

import numpy as np

from ...contracts import ManifestSample
from ...registry import TaskSpec

BAND_NAMES = [f"AEF_{index:02d}" for index in range(64)]
# Four-band RGB+NIR student (T11D): the visible bands plus NIR in the PASTIS-HD
# S2 order (B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12).
RGBN_BANDS = ["B02", "B03", "B04", "B08"]
RGBN_INDICES = [0, 1, 2, 6]
RGBN_MAX_FRAMES = 20
# Six-band matched-pair student (T11P): the Prithvi-native HLS subset, at four
# 2019 dates so Clay and Prithvi share one time budget. PASTIS-HD S2 order is
# B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 -> columns 0,1,2,7,8,9.
S2SIX_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
S2SIX_INDICES = [0, 1, 2, 7, 8, 9]
S2SIX_MAX_FRAMES = 4
S2_STACK_PROFILES = {
    tuple(RGBN_BANDS): (RGBN_INDICES, RGBN_MAX_FRAMES),
    tuple(S2SIX_BANDS): (S2SIX_INDICES, S2SIX_MAX_FRAMES),
}
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


def select_2019_frames(dates: list[int], max_frames: int = RGBN_MAX_FRAMES) -> list[int]:
    """2019-only frame indices, evenly spaced by date, at most ``max_frames``.

    Shared by the T11D manifest builder and the Clay worker so the declared
    timestamps and the tensors the student reads cannot drift. PASTIS-HD's crop
    year starts in Sep 2018; this track keeps only 2019 acquisitions and
    records the missing months as a property of the protocol.
    """
    indices = [index for index, date in enumerate(dates) if str(int(date)).startswith("2019")]
    if not indices:
        raise ManifestBuildError("patch has no 2019 S2 acquisition")
    if len(indices) <= max_frames:
        return indices
    picks = np.linspace(0, len(indices) - 1, max_frames)
    return sorted({indices[int(round(float(pick)))] for pick in picks})


def build_pastis_hd_manifest(data_root: Path, task_spec: TaskSpec) -> list[ManifestSample]:
    if task_spec.input_kind == "raster":
        return _build_rgbn(data_root, task_spec)
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
        f"task {task_spec.task_id}: PASTIS-HD needs input_kind raster, "
        f"aef_features or embedding_features, got {task_spec.input_kind!r}"
    )


def _build_rgbn(data_root: Path, task_spec: TaskSpec) -> list[ManifestSample]:
    """Raw S2 stack rows for T11D (4 bands, 20 frames) and T11P (6 bands, 4).

    Both read the same PASTIS-HD payload; only the declared band subset and the
    number of 2019 frames differ.

    The AEF extraction tree already links the full PASTIS-HD payload under
    ``derived/aef-2019/PASTIS-HD`` (DATA_S2 + ANNOTATIONS + metadata.geojson),
    so the four-band task reads the same files without copying 36 GB.
    """
    profile = S2_STACK_PROFILES.get(tuple(task_spec.bands))
    if profile is None:
        raise ManifestBuildError(
            f"task {task_spec.task_id}: PASTIS-HD raw-stack bands must be "
            f"{sorted(S2_STACK_PROFILES)}, got {list(task_spec.bands)}"
        )
    _, max_frames = profile
    if task_spec.time_steps != max_frames:
        raise ManifestBuildError(
            f"task {task_spec.task_id}: {len(task_spec.bands)}-band stack uses "
            f"{max_frames} frames, got time_steps={task_spec.time_steps}"
        )
    base = data_root / "derived" / "aef-2019" / "PASTIS-HD"
    sidecars = data_root / "derived" / "aef-2019" / "features" / "2019"
    annotations = base / "ANNOTATIONS"
    metadata_path = base / "metadata.geojson"
    if not metadata_path.is_file():
        raise ManifestBuildError(f"missing PASTIS-HD metadata under {metadata_path}")
    if not annotations.is_dir() or not (base / "DATA_S2").is_dir():
        raise ManifestBuildError(f"missing PASTIS-HD payload under {base}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    samples: list[ManifestSample] = []
    for feature in metadata["features"]:
        properties = feature["properties"]
        patch = int(properties["ID_PATCH"])
        fold = int(properties["Fold"])
        sidecar = sidecars / f"{patch}.json"
        if not sidecar.is_file():
            raise ManifestBuildError(f"patch {patch}: AEF sidecar {sidecar.name} missing")
        dates = [int(date) for date in json.loads(sidecar.read_text(encoding="utf-8"))["dates_s2"]]
        frames = select_2019_frames(dates, max_frames)
        stack = base / "DATA_S2" / f"S2_{patch}.npy"
        target = annotations / f"TARGET_{patch}.npy"
        parcel = annotations / f"ParcelIDs_{patch}.npy"
        if not stack.is_file() or not target.is_file():
            raise ManifestBuildError(f"patch {patch}: S2 stack or label missing")
        samples.append(
            ManifestSample(
                task_id=task_spec.task_id,
                sample_id=str(patch),
                group_id=str(patch),
                split=SPLIT_BY_FOLD[fold],
                image_paths=[_relative(stack, data_root)],
                label_path=_relative(target, data_root),
                aux_paths={
                    "s2_path": _relative(stack, data_root),
                    "parcel_path": _relative(parcel, data_root),
                    "tile": str(properties["TILE"]),
                    "label_year": TIMESTAMP,
                    "frame_rule": "2019-only-linspace",
                    "max_frames": str(max_frames),
                    "s2_columns": ",".join(str(index) for index in profile[0]),
                    "dates_source": _relative(sidecar, data_root),
                },
                sensor=task_spec.sensor,
                bands=list(task_spec.bands),
                timestamps=[str(dates[index]) for index in frames],
                temporal_mask=[True] * len(frames),
                gsd_m=GSD_M,
                checksum=_sha256(stack),
                window=[0, 0, GRID, GRID],
                valid_fraction=float(
                    json.loads(sidecar.read_text(encoding="utf-8")).get("valid_share", 1.0)
                ),
            )
        )
    if not samples:
        raise ManifestBuildError("no patches assembled for the raw-stack task")
    return samples


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
