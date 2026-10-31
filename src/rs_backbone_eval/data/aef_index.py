"""Bounding-box index for the public AlphaEarth Foundations (AEF) COGs.

AEF ships one COG per 8192 px tile, 64 bands at 10 m, under
``satellite_embedding/v1/annual/<year>/<zone>/``. There is no manifest of tile
footprints in the bucket, so resolving one point to its file means opening
candidates over HTTPS: measured at 246 s for 54N/2021 (261 COGs). This module
scans a (year, zone) once, stores the normalised bounds, and answers later
lookups in memory.

``rasterio`` is imported lazily: the control plane must not depend on it, and the
pure helpers (``find_cog``, ``load_index``) stay importable without GDAL.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://storage.googleapis.com/alphaearth_foundations"
PREFIX = "satellite_embedding/v1/annual"
NODATA = -128
CELL_M = 10.0


class AefIndexError(RuntimeError):
    """The index could not be built or does not describe the requested tile."""


def list_cogs(year: int, zone: str, *, timeout: int = 60, max_pages: int = 50) -> list[str]:
    """Every COG key under one ``year/zone`` prefix, following listing pages."""
    prefix = f"{PREFIX}/{year}/{zone}/"
    keys: list[str] = []
    marker: str | None = None
    for _ in range(max_pages):
        url = f"{BUCKET}?prefix={prefix}&max-keys=1000"
        if marker:
            url += "&marker=" + urllib.parse.quote(marker)
        with urllib.request.urlopen(url, timeout=timeout) as response:
            xml = response.read().decode("utf-8", "replace")
        page = re.findall(r"<Key>(.*?)</Key>", xml)
        keys.extend(key for key in page if key.endswith(".tiff"))
        if "<IsTruncated>true" not in xml:
            break
        marker = page[-1]
    return keys


def _entry(key: str) -> dict | None:
    """Open one COG far enough to record its footprint.

    AEF COGs are stored bottom-up (positive y scale), so ``dataset.bounds``
    reports bottom > top and cannot be used; the extent is rebuilt from the
    affine and normalised with min/max.
    """
    import rasterio  # worker-only dependency

    try:
        with rasterio.open("/vsicurl/" + f"{BUCKET}/{key}") as dataset:
            transform = dataset.transform
            xs = [transform.c, transform.c + transform.a * dataset.width]
            ys = [transform.f, transform.f + transform.e * dataset.height]
            return {
                "key": key,
                "bounds": [min(xs), min(ys), max(xs), max(ys)],
                "width": int(dataset.width),
                "height": int(dataset.height),
                "bands": int(dataset.count),
                "dtype": dataset.dtypes[0],
                "crs": str(dataset.crs) if dataset.crs else None,
            }
    except Exception:
        return None


def build_index(year: int, zone: str, out_dir: Path, *, workers: int = 8) -> dict:
    """Scan one year/zone and write ``<year>_<zone>.json`` under ``out_dir``."""
    started = time.time()
    keys = list_cogs(year, zone)
    if not keys:
        raise AefIndexError(f"no COGs under {year}/{zone}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        entries = [item for item in pool.map(_entry, keys) if item is not None]
    if not entries:
        raise AefIndexError(f"none of the {len(keys)} COGs could be opened")
    missing = sorted(set(keys) - {item["key"] for item in entries})
    crs = sorted({item["crs"] for item in entries if item["crs"]})
    index = {
        "year": year,
        "zone": zone,
        "bucket": BUCKET,
        "prefix": f"{PREFIX}/{year}/{zone}/",
        "crs": crs,
        "cells": sorted({item["width"] for item in entries}),
        "cog_count": len(entries),
        "unreadable": missing,
        "seconds": round(time.time() - started, 1),
        "entries": sorted(entries, key=lambda item: item["key"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{year}_{zone}.json"
    target.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return index


def index_path(out_dir: Path, year: int, zone: str) -> Path:
    return out_dir / f"{year}_{zone}.json"


def load_index(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AefIndexError(f"no AEF index at {path}") from error


def find_cog(index: dict, x: float, y: float) -> dict | None:
    """The entry whose bounds contain ``(x, y)`` in the index CRS."""
    for entry in index.get("entries", []):
        left, bottom, right, top = entry["bounds"]
        if left <= x <= right and bottom <= y <= top:
            return entry
    return None
