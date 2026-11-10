#!/usr/bin/env python3
"""Extract AEF 2017 and OlmoEarth v1.3 2017 windows for FBP scenes (read-only).

The two archives live on different hosts (GCS and source.coop) and are read by
different mechanisms, so this script is split into phases that can run at the
same time without sharing state:

``labels``  ortho the 4 m label mask onto the common 10 m grid and pick the
            window positions (local disk, CPU only);
``aef``     read AEF 2017 for the chosen windows (GCS, ~0.3 MB/s here);
``olmo``    read OlmoEarth 2017 for the same windows (source.coop, ~2 MB/s,
            about 33 s per 512 window: the shard index dominates, so a 256
            and a 512 window cost the same).

Each phase writes one file per scene and skips existing outputs, so the three
can run as separate processes and resume after an interruption.

Measured 2026-11-11 per window:

* AEF: 256x256 costs 8-11 s, but one 512x512 read costs 12m44 because the COG
  blocks are 1024 px and GDAL falls back to whole-block reads; windows are
  therefore read as 256 tiles and stitched.
* OlmoEarth: 512x512 and 256x256 both cost ~33 s, so the bigger window is free.

Run with the worker-style dependency set (nothing is added to the root env)::

    uv run --python 3.12 --with 'zarr>=3' --with obstore --with numpy \\
        --with rasterio --with pyarrow python scripts/fbp_extract_windows.py \\
        --phase labels --out /home/yzhong/rs-eval/rs-eval-cache/fbp2017 --workers 2
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fbp_label_resample import combine  # noqa: E402
from fbp_rpc_vrt import build_vrt, parse_rpb, raster_shape, rpc_metadata  # noqa: E402

ROOT = Path("/geodata/Five-Billion-Pixels")
IMAGES = ROOT / "Image_16bit_BGRNir"
LABELS = ROOT / "Annotation__index"
RPC = ROOT / "Coordinate_files"
AEF_INDEX = Path("/geodata/rs-eval/data/derived/aef/aef_index.parquet")
OLMO_URL = "s3://olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr"
OLMO_ENDPOINT = "https://data.source.coop"
YEAR_INDEX = 0  # OlmoEarth `c/0` == 2017
WINDOW = 512
GRID = 10
TILE = 256  # AEF COG block-aligned read size
MIN_VALID = 0.30
SEED = 17
CLASSES = 24


os.environ.setdefault("GS_NO_SIGN_REQUEST", "YES")  # AEF 是公开 GCS bucket


def scenes() -> list[dict]:
    """Scene metadata straight from the RPB sidecars."""
    rows = []
    for path in sorted(RPC.glob("*.rpb")):
        values = parse_rpb(path.read_text(encoding="utf-8", errors="ignore"))
        rows.append(
            {
                "scene": path.stem,
                "lon": values["longOffset"][0],
                "lat": values["latOffset"][0],
                "height": values["heightOffset"][0],
            }
        )
    return rows


def zone_of(lon: float) -> int:
    return int((lon + 180.0) // 6.0) + 1


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, capture_output=True)


def _warp(source: Path, dst: Path, srs: str, height: float) -> None:
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
            f"RPC_HEIGHT={height}",
            str(source),
            str(dst),
        ]
    )


def _label_grid(
    scene: str, work: Path, srs: str, height: float
) -> tuple[np.ndarray, np.ndarray, list]:
    """(labels, coverage, transform) of the 10 m label raster for one scene."""
    import rasterio

    label_png = LABELS / f"{scene}_24label.png"
    image = IMAGES / f"{scene}.tiff"
    meta = rpc_metadata(
        parse_rpb((RPC / f"{scene}.rpb").read_text(encoding="utf-8", errors="ignore"))
    )
    width, height_px, _, _ = raster_shape(image)
    fractions = []
    for cls in range(1, CLASSES + 1):
        mask = work / f"c{cls}.tif"
        if not mask.exists():
            _run(
                [
                    "gdal_calc.py",
                    "--overwrite",
                    "--quiet",
                    f"--calc=(A=={cls}).astype('uint8')",
                    "--NoDataValue=255",
                    "--type=Byte",
                    "-A",
                    str(label_png),
                    f"--outfile={mask}",
                ]
            )
        vrt = work / f"c{cls}.vrt"
        vrt.write_text(
            build_vrt(meta, mask.resolve(), width, height_px, 1, "Byte"), encoding="utf-8"
        )
        warped = work / f"w{cls}.tif"
        if not warped.exists():
            _warp(vrt, warped, srs, height)
        with rasterio.open(warped) as dataset:
            fractions.append(dataset.read(1))

    coverage_mask = work / "coverage.tif"
    if not coverage_mask.exists():
        _run(
            [
                "gdal_calc.py",
                "--overwrite",
                "--quiet",
                "--calc=(A>=0).astype('uint8')",
                "--NoDataValue=255",
                "--type=Byte",
                "-A",
                str(label_png),
                f"--outfile={coverage_mask}",
            ]
        )
    coverage_vrt = work / "coverage.vrt"
    coverage_vrt.write_text(
        build_vrt(meta, coverage_mask.resolve(), width, height_px, 1, "Byte"), encoding="utf-8"
    )
    coverage_warped = work / "coverage_warped.tif"
    if not coverage_warped.exists():
        _warp(coverage_vrt, coverage_warped, srs, height)
    with rasterio.open(coverage_warped) as dataset:
        coverage = dataset.read(1)
        transform = dataset.transform
    labels, _ = combine(np.stack(fractions), coverage)
    return labels, coverage, transform


def pick_windows(labels: np.ndarray, transform) -> list[tuple[int, int, float, float, float]]:
    """``(row, col, valid, x_left, y_top)`` on a 2x2 quadrant grid, label-only rule."""
    rows, cols = labels.shape
    rng = np.random.default_rng(SEED)
    chosen: list[tuple[int, int, float, float, float]] = []
    for r0, r1 in ((0, rows // 2), (rows // 2, rows)):
        for c0, c1 in ((0, cols // 2), (cols // 2, cols)):
            jitter_r = int(rng.integers(0, max(1, (r1 - r0) - WINDOW)))
            jitter_c = int(rng.integers(0, max(1, (c1 - c0) - WINDOW)))
            for attempt in range(4):
                top = r0 + jitter_r + attempt * 64
                left = c0 + jitter_c + attempt * 64
                if top + WINDOW > rows or left + WINDOW > cols:
                    break
                valid = float((labels[top : top + WINDOW, left : left + WINDOW] != 0).mean())
                if valid >= MIN_VALID:
                    x_left, y_top = transform * (left, top)
                    chosen.append((top, left, valid, x_left, y_top))
                    break
    return chosen


def phase_labels(args, rows: list[dict]) -> None:
    (args.out / "labels").mkdir(parents=True, exist_ok=True)

    def one(row: dict) -> str:
        target = args.out / "labels" / f"{row['scene']}.npz"
        if target.exists():
            return "skip"
        work = args.out / "work" / row["scene"]
        work.mkdir(parents=True, exist_ok=True)
        labels, coverage, transform = _label_grid(
            row["scene"], work, f"EPSG:326{zone_of(row['lon']):02d}", row["height"]
        )
        windows = pick_windows(labels, transform)
        np.savez_compressed(
            target,
            labels=labels,
            coverage=coverage.astype(np.float16),
            windows=np.array(windows, dtype=np.float64).reshape(-1, 5),
            zone=zone_of(row["lon"]),
        )
        return f"{len(windows)} windows"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, status in zip(rows, pool.map(one, rows), strict=True):
            print(f"labels {row['scene']}: {status}", flush=True)


_AEF_LOCK = threading.Lock()
_AEF_TILES: dict[str, list[tuple[float, float, str]]] | None = None


def aef_tile_for(x: float, y: float, zone: int) -> tuple[str, float, float]:
    """``(path, x_left_edge, y_top_edge)`` of the AEF 2017 tile holding ``x, y``.

    Tiles are 8192 px at 10 m; the index carries ``utm_east``/``utm_south`` of
    the tile corner, so the origin is ``(utm_east - 81920, utm_south + 81920)``.
    """
    global _AEF_TILES
    with _AEF_LOCK:
        if _AEF_TILES is None:
            import pyarrow.parquet as pq

            table = pq.read_table(AEF_INDEX)
            columns = table.to_pydict()
            tiles: dict[str, list[tuple[float, float, str]]] = {}
            for i in range(table.num_rows):
                if columns["year"][i] != 2017:
                    continue
                tiles.setdefault(columns["utm_zone"][i], []).append(
                    (columns["utm_east"][i], columns["utm_south"][i], columns["path"][i])
                )
            _AEF_TILES = tiles
    for east, south, path in _AEF_TILES[f"{zone}N"]:
        if east - 81920 <= x < east and south <= y < south + 81920:
            return path, float(east - 81920), float(south + 81920)
    raise RuntimeError(f"no AEF 2017 tile at {x:.0f},{y:.0f} zone {zone}")


def read_aef_windows(meta: dict, scene: str) -> dict:
    import rasterio

    features = np.full((len(meta["windows"]), 64, WINDOW, WINDOW), -128, dtype=np.int8)
    started = time.time()
    for index, window in enumerate(meta["windows"]):
        _, _, _, x_left, y_top = window
        path, x0, y0 = aef_tile_for(x_left, y_top, int(meta["zone"]))
        col = int((x_left - x0) // GRID)
        row = int((y0 - y_top) // GRID)
        col -= col % TILE
        row -= row % TILE
        for r in range(0, WINDOW, TILE):
            for c in range(0, WINDOW, TILE):
                part = Path(f"/tmp/opencode/.aef_{scene}_{index}_{r}_{c}.tif")
                subprocess.run(
                    [
                        "gdal_translate",
                        "-q",
                        "-srcwin",
                        str(col + c),
                        str(row + r),
                        str(TILE),
                        str(TILE),
                        "-of",
                        "GTiff",
                        "-co",
                        "COMPRESS=DEFLATE",
                        path.replace("gs://", "/vsigs/"),
                        str(part),
                    ],
                    check=True,
                    capture_output=True,
                )
                with rasterio.open(part) as dataset:
                    features[index, :, r : r + TILE, c : c + TILE] = dataset.read()
                part.unlink(missing_ok=True)
    return {"features": features, "seconds": time.time() - started}


_OLMO = threading.local()


def _olmo_group(zone: int):
    root = getattr(_OLMO, "root", None)
    if root is None:
        import zarr
        from obstore.store import S3Store

        store = zarr.storage.ObjectStore(
            S3Store.from_url(OLMO_URL, endpoint=OLMO_ENDPOINT, skip_signature=True), read_only=True
        )
        zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 0})
        root = zarr.open_group(store, mode="r")
        _OLMO.root = root
    return root[f"utm{zone:02d}"]


def read_olmo_windows(meta: dict) -> dict:
    features = np.full((len(meta["windows"]), 128, WINDOW, WINDOW), -128, dtype=np.int8)
    started = time.time()
    zone = int(meta["zone"])
    group = _olmo_group(zone)
    a, _, c, _, e, f = group.attrs["spatial:transform"]
    embeddings = group["embeddings"]
    for index, window in enumerate(meta["windows"]):
        _, _, _, x_left, y_top = window
        row = int(np.floor((y_top - f) / e))
        col = int(np.floor((x_left - c) / a))
        features[index] = np.asarray(
            embeddings[YEAR_INDEX, 0:128, row : row + WINDOW, col : col + WINDOW]
        )
    return {"features": features, "seconds": time.time() - started}


def _phase_source(args, rows: list[dict], name: str, reader) -> None:
    (args.out / name).mkdir(parents=True, exist_ok=True)

    def one(row: dict) -> str:
        target = args.out / name / f"{row['scene']}.npz"
        if target.exists():
            return "skip"
        labels_file = args.out / "labels" / f"{row['scene']}.npz"
        if not labels_file.exists():
            return "no-labels"
        meta = dict(np.load(labels_file))
        result = reader(meta, row["scene"]) if name == "aef" else reader(meta)
        np.savez_compressed(
            target, features=result["features"], seconds=np.float64(result["seconds"])
        )
        return f"{result['features'].shape} in {result['seconds']:.0f}s"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, status in zip(rows, pool.map(one, rows), strict=True):
            print(f"{name} {row['scene']}: {status}", flush=True)


def phase_summary(args, rows: list[dict]) -> None:
    """Per-scene coverage and fill rates of whatever has been extracted so far."""
    import json

    report = []
    for row in rows:
        scene = row["scene"]
        entry: dict = {"scene": scene, "zone": zone_of(row["lon"]), "height_m": row["height"]}
        labels_file = args.out / "labels" / f"{scene}.npz"
        if labels_file.exists():
            meta = np.load(labels_file)
            labels = meta["labels"]
            coverage = meta["coverage"].astype(np.float32)
            inside = coverage > 0.99
            entry.update(
                windows=int(len(meta["windows"])),
                footprint=round(100 * float(inside.mean()), 1),
                unlabeled_inside=round(100 * float((1 - coverage[inside]).mean()), 1)
                if inside.any()
                else None,
                ignore_inside=round(100 * float((labels[inside] == 0).mean()), 1)
                if inside.any()
                else None,
            )
        for source in ("aef", "olmo"):
            path = args.out / source / f"{scene}.npz"
            if not path.exists():
                continue
            features = np.load(path)["features"]
            entry[f"{source}_fill"] = round(100 * float((features == -128).all(axis=1).mean()), 1)
            entry[f"{source}_windows"] = int(features.shape[0])
        report.append(entry)

    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    done = sum(1 for item in report if item.get("aef_windows") and item.get("olmo_windows"))
    with_aef = sum(1 for item in report if item.get("aef_windows"))
    labelled = sum("windows" in item for item in report)
    print(f"scenes {len(report)} | labels {labelled} | aef {with_aef} | both {done}")
    if with_aef:
        fills = [i["aef_fill"] for i in report if "aef_fill" in i]
        print(f"aef fill: median {np.median(fills):.2f}% max {max(fills):.2f}%")
    olmo = [i["olmo_fill"] for i in report if "olmo_fill" in i]
    if olmo:
        print(f"olmo fill: median {np.median(olmo):.2f}% max {max(olmo):.2f}%")
    ignore = [i["ignore_inside"] for i in report if i.get("ignore_inside") is not None]
    if ignore:
        print(f"ignore inside footprint: median {np.median(ignore):.1f}% max {max(ignore):.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["labels", "aef", "olmo", "summary"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = scenes()
    if args.limit:
        rows = rows[: args.limit]
    if args.phase == "labels":
        phase_labels(args, rows)
    elif args.phase == "aef":
        _phase_source(args, rows, "aef", read_aef_windows)
    elif args.phase == "olmo":
        _phase_source(args, rows, "olmo", read_olmo_windows)
    else:
        phase_summary(args, rows)


if __name__ == "__main__":
    main()
