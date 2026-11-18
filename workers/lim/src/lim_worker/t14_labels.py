"""T14 (SpaceNet 7) change masks: a pixel is 1 when a building covers it at t2 but not at t1.

The footprints are the ``*_Buildings.geojson`` files in each AOI's
``labels_match_pix`` directory. Their coordinates are already in pixel units of
the monthly raster, so they are rasterised with an identity transform. The
masks are written as uint8 GeoTIFFs at ``row["label_path"]`` under the data
root, with the same grid and CRS as the image.

Run in the LIM worker env (it needs rasterio), not in the control plane::

    uv run --project workers/lim python -m lim_worker.t14_labels \\
        --data-root DATA_ROOT --manifest MANIFEST.jsonl

Known gap: the UDM (cloud/unusable) polygons are not applied, so clouds can
appear as change. Existing masks are kept unless ``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine

RAW_TRAIN = Path("raw/spacenet7/train")
LABEL_PREFIX = "derived/spacenet7-change/"
TASK_ID = "T14"


def footprint_path(data_root: Path, row: dict, image_path: str) -> Path:
    """Buildings geojson that matches one monthly image of the row's AOI."""
    stem = Path(image_path).stem
    return (
        data_root / RAW_TRAIN / row["group_id"] / "labels_match_pix" / f"{stem}_Buildings.geojson"
    )


def footprint_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask of building pixels; an empty footprint file gives all False."""
    if not path.is_file():
        raise FileNotFoundError(f"missing building footprints {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    geometries = [f["geometry"] for f in data.get("features", []) if f.get("geometry")]
    if not geometries:
        return np.zeros(shape, dtype=bool)
    burned = rasterize(
        [(geom, 1) for geom in geometries],
        out_shape=shape,
        transform=Affine.identity(),
        fill=0,
        dtype="uint8",
    )
    return burned.astype(bool)


def change_mask(data_root: Path, row: dict) -> np.ndarray:
    """uint8 mask: 1 where a building is present at t2 and absent at t1."""
    _, _, width, height = (int(v) for v in row["window"])
    shape = (height, width)
    t1, t2 = row["image_paths"]
    before = footprint_mask(footprint_path(data_root, row, t1), shape)
    after = footprint_mask(footprint_path(data_root, row, t2), shape)
    return (after & ~before).astype(np.uint8)


def write_mask(data_root: Path, row: dict, mask: np.ndarray) -> None:
    """Write one-band uint8 GeoTIFF on the image grid (CRS and affine from the row)."""
    out = data_root / row["label_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    a, b, c, d, e, f = row["transform"]
    height, width = mask.shape
    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="uint8",
        crs=row["crs"],
        transform=Affine(a, b, c, d, e, f),
        compress="deflate",
    ) as dst:
        dst.write(mask, 1)


def build_masks(data_root: Path, manifest: Path, force: bool = False) -> dict[str, int]:
    """Write every missing change mask named by the T14 rows of ``manifest``."""
    counts = {"written": 0, "skipped": 0}
    seen: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task_id") != TASK_ID or row["label_path"] in seen:
                continue
            seen.add(row["label_path"])
            if (data_root / row["label_path"]).exists() and not force:
                counts["skipped"] += 1
                continue
            write_mask(data_root, row, change_mask(data_root, row))
            counts["written"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="rewrite existing masks")
    args = parser.parse_args(argv)
    counts = build_masks(args.data_root, args.manifest, force=args.force)
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
