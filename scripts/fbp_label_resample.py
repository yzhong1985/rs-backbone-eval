#!/usr/bin/env python3
"""Resample a 4 m FBP label mask onto the common 10 m grid without absorbing ignore.

``gdalwarp -r mode`` on the raw label is wrong for FBP because ``0`` is
*unlabeled*, not a class. Measured on scene ``GF2_PMS1__L1A0000564539-MSS1``,
the plain mode path turns 27.3% unlabeled at 4 m into **46.3%** at 10 m and
shrinks every class by the same ~0.70 factor, because a 10 m pixel is "unlabeled"
whenever no single class holds the plurality. The rule used here keeps the class
proportions (45.46% vs 45.45% at 4 m) and reports unlabeled separately:

* one 0/1 mask per class, each warped with ``-r average`` → fraction of the
  10 m pixel covered by that class;
* an all-ones mask warped the same way → coverage, so pixels outside the
  rotated footprint are not mistaken for unlabeled;
* ``unlabeled = (coverage - sum(fractions)) / coverage``;
* ``unlabeled > 0.5`` → ignore, otherwise the arg-max class wins.

``combine`` is the policy and is dependency-free; the CLI reads/writes rasters with
``rasterio``, so run it with a worker venv::

    workers/prithvi/.venv/bin/python scripts/fbp_label_resample.py ...

Usage::

    python scripts/fbp_label_resample.py --label SCENE_24label.png \\
        --rpb SCENE.rpb --srs EPSG:32650 --rpc-height 6.94 --out SCENE_label10.tif
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fbp_rpc_vrt import build_vrt, parse_rpb, raster_shape, rpc_metadata  # noqa: E402

IGNORE = 0
"""Output value for "not usable": unlabeled majority or outside the footprint."""


def combine(
    fractions: np.ndarray, coverage: np.ndarray, threshold: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """``(labels, unlabeled_fraction)`` from per-class fractions and coverage.

    ``fractions`` is ``(classes, rows, cols)`` holding, for classes ``1..classes``,
    the fraction of each 10 m pixel they cover. ``coverage`` is ``(rows, cols)``
    holding the fraction covered by valid 4 m data (1.0 inside the scene).
    Pixels with more than ``threshold`` unlabeled area, or no coverage at all,
    become :data:`IGNORE`.
    """
    if fractions.ndim != 3 or fractions.shape[1:] != coverage.shape:
        raise ValueError(f"fractions {fractions.shape} do not match coverage {coverage.shape}")
    covered = coverage > 1e-6
    total = fractions.sum(axis=0)
    unlabeled = np.divide(
        coverage - total, np.where(covered, coverage, 1.0), out=np.ones_like(total), where=covered
    )
    unlabeled = np.clip(unlabeled, 0.0, 1.0)
    labels = fractions.argmax(axis=0) + 1
    keep = covered & (unlabeled <= threshold)
    return np.where(keep, labels, IGNORE).astype(np.uint8), unlabeled.astype(np.float32)


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True)


def _one_hot(label: Path, dst: Path, expression: str) -> None:
    _run(
        [
            "gdal_calc.py",
            "--overwrite",
            "--quiet",
            f"--calc={expression}",
            "--NoDataValue=255",
            "--type=Byte",
            "-A",
            str(label),
            f"--outfile={dst}",
        ]
    )


def _warp(source: Path, dst: Path, srs: str, rpc_height: float) -> None:
    _run(
        [
            "gdalwarp",
            "-q",
            "-overwrite",
            "-rpc",
            "-t_srs",
            srs,
            "-tr",
            "10",
            "10",
            "-tap",
            "-r",
            "average",
            "-of",
            "GTiff",
            "-ot",
            "Float32",
            "-to",
            f"RPC_HEIGHT={rpc_height}",
            str(source),
            str(dst),
        ]
    )


def resample_scene(
    label: Path, rpb: Path, out: Path, *, srs: str, rpc_height: float, classes: int = 24
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp ``label`` to 10 m, apply :func:`combine`, and write ``out``."""
    import rasterio

    work = out.parent / f".{out.stem}_work"
    work.mkdir(parents=True, exist_ok=True)
    meta = rpc_metadata(parse_rpb(rpb.read_text(encoding="utf-8", errors="ignore")))
    width, height, _, _ = raster_shape(label)

    fractions = []
    for cls in range(1, classes + 1):
        mask = work / f"c{cls}.tif"
        _one_hot(label, mask, f"(A=={cls}).astype('uint8')")
        vrt = work / f"c{cls}.vrt"
        vrt.write_text(build_vrt(meta, mask.resolve(), width, height, 1, "Byte"), encoding="utf-8")
        warped = work / f"w{cls}.tif"
        _warp(vrt, warped, srs, rpc_height)
        with rasterio.open(warped) as dataset:
            fractions.append(dataset.read(1))

    coverage_mask = work / "coverage.tif"
    _one_hot(label, coverage_mask, "(A>=0).astype('uint8')")
    coverage_vrt = work / "coverage.vrt"
    coverage_vrt.write_text(
        build_vrt(meta, coverage_mask.resolve(), width, height, 1, "Byte"), encoding="utf-8"
    )
    _warp(coverage_vrt, work / "coverage_warped.tif", srs, rpc_height)
    with rasterio.open(work / "coverage_warped.tif") as dataset:
        coverage = dataset.read(1)
        profile = dataset.profile.copy()

    labels, unlabeled = combine(np.stack(fractions), coverage)
    profile.update(count=1, dtype="uint8", compress="deflate", nodata=IGNORE)
    with rasterio.open(out, "w", **profile) as dataset:
        dataset.write(labels, 1)
    np.save(out.with_suffix(".unlabeled.npy"), unlabeled)
    return labels, unlabeled, coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", type=Path, required=True)
    parser.add_argument("--rpb", type=Path, required=True)
    parser.add_argument("--srs", required=True, help="EPSG:326xx of the scene")
    parser.add_argument("--rpc-height", type=float, required=True, help="the RPB HEIGHT_OFF")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=24)
    args = parser.parse_args()

    labels, unlabeled, coverage = resample_scene(
        args.label,
        args.rpb,
        args.out,
        srs=args.srs,
        rpc_height=args.rpc_height,
        classes=args.classes,
    )
    inside = coverage > 0.99  # the rotated footprint, excluding the bbox corners
    print(
        f"wrote {args.out}: {labels.shape[1]}x{labels.shape[0]} px, "
        f"footprint {100 * float(inside.mean()):.1f}% of the bbox, "
        f"mean unlabeled {100 * float(unlabeled[inside].mean()):.1f}% inside it, "
        f"ignore {100 * float((labels[inside] == IGNORE).mean()):.2f}% inside"
    )


if __name__ == "__main__":
    main()
