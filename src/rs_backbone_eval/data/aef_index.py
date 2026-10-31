"""AEF tile index: the official Parquet first, a scan only for cross-checks.

The public bucket ships a per-COG index at
``satellite_embedding/v1/annual/aef_index.parquet`` (302,466 rows, one per file,
with ``crs``, ``year``, ``utm_zone`` and the tile's UTM bounds). It covers every
year and zone, so it replaces the per-zone header scan: the scan took 665 s for
54N/2021 alone.

Reading the Parquet needs ``pyarrow``, which only the worker environments have;
``official_slice`` imports it lazily and the control plane never touches it. The
dependency-free path is a converted slice (``<year>_<zone>.json``) produced by
``scripts/aef_index.py``. ``scan_index`` is kept so a slice can be verified
against the COG headers; it is not the production path.
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
OFFICIAL_PARQUET = f"{PREFIX}/aef_index.parquet"
NODATA = -128
CELL_M = 10.0


class AefIndexError(RuntimeError):
    """The index could not be built, read or does not describe the tile."""


def list_cogs(year: int, zone: str, *, timeout: int = 60, max_pages: int = 50) -> list[str]:
    """Every COG key under one ``year/zone`` prefix (scan path only)."""
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
    """Open one COG far enough to record its footprint (scan path only).

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
                "path": "gs://alphaearth_foundations/" + key,
                "bounds": [min(xs), min(ys), max(xs), max(ys)],
                "width": int(dataset.width),
                "height": int(dataset.height),
                "bands": int(dataset.count),
                "dtype": dataset.dtypes[0],
                "crs": str(dataset.crs) if dataset.crs else None,
            }
    except Exception:
        return None


def scan_index(year: int, zone: str, out_dir: Path, *, workers: int = 8) -> dict:
    """Header scan of one year/zone. Cross-check only: 665 s for 54N/2021."""
    started = time.time()
    keys = list_cogs(year, zone)
    if not keys:
        raise AefIndexError(f"no COGs under {year}/{zone}")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        entries = [item for item in pool.map(_entry, keys) if item is not None]
    if not entries:
        raise AefIndexError(f"none of the {len(keys)} COGs could be opened")
    index = {
        "source": "scan",
        "year": year,
        "zone": zone,
        "bucket": BUCKET,
        "crs": sorted({item["crs"] for item in entries if item["crs"]}),
        "cog_count": len(entries),
        "unreadable": sorted(set(keys) - {item["key"] for item in entries}),
        "seconds": round(time.time() - started, 1),
        "entries": sorted(entries, key=lambda item: item["key"]),
    }
    write_slice(index, out_dir)
    return index


def official_slice(parquet: Path, year: int, zone: str) -> dict:
    """One year/zone slice of the official Parquet (needs ``pyarrow``)."""
    try:
        import pyarrow.parquet as parquet_file
    except ImportError as error:  # pragma: no cover - root env has no pyarrow
        raise AefIndexError(
            "reading the official index needs pyarrow; run this in a worker environment "
            "or convert a slice with scripts/aef_index.py --source official"
        ) from error
    table = parquet_file.read_table(
        parquet,
        columns=[
            "path",
            "year",
            "utm_zone",
            "crs",
            "utm_west",
            "utm_south",
            "utm_east",
            "utm_north",
        ],
    )
    frame = table.to_pandas()
    frame = frame[(frame["year"] == year) & (frame["utm_zone"] == zone)]
    if frame.empty:
        raise AefIndexError(f"official index has no rows for {year}/{zone}")
    entries = [
        {
            "key": str(row.path).split("/satellite_embedding/v1/annual/", 1)[-1],
            "path": str(row.path),
            "bounds": [
                float(row.utm_west),
                float(row.utm_south),
                float(row.utm_east),
                float(row.utm_north),
            ],
            "crs": str(row.crs),
        }
        for row in frame.itertuples()
    ]
    return {
        "source": "official",
        "year": year,
        "zone": zone,
        "parquet": str(parquet),
        "crs": sorted({entry["crs"] for entry in entries}),
        "cog_count": len(entries),
        "entries": sorted(entries, key=lambda item: item["key"]),
    }


def write_slice(index: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{index['year']}_{index['zone']}.json"
    target.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return target


def index_path(out_dir: Path, year: int, zone: str) -> Path:
    return out_dir / f"{year}_{zone}.json"


def load_index(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise AefIndexError(f"no AEF index at {path}") from error


def find_cog(index: dict, x: float, y: float) -> dict | None:
    """The entry whose UTM bounds contain ``(x, y)``."""
    for entry in index.get("entries", []):
        left, bottom, right, top = entry["bounds"]
        if left <= x <= right and bottom <= y <= top:
            return entry
    return None


def compare_indexes(left: dict, right: dict) -> list[str]:
    """Differences between two indexes of the same year/zone (cross-check)."""
    a = {entry["key"]: entry["bounds"] for entry in left.get("entries", [])}
    b = {entry["key"]: entry["bounds"] for entry in right.get("entries", [])}
    problems = []
    for key in sorted(set(a) - set(b)):
        problems.append(f"only in {left.get('source')}: {key}")
    for key in sorted(set(b) - set(a)):
        problems.append(f"only in {right.get('source')}: {key}")
    for key in sorted(set(a) & set(b)):
        if any(abs(x - y) > 0.51 for x, y in zip(a[key], b[key], strict=True)):
            problems.append(f"bounds differ for {key}: {a[key]} vs {b[key]}")
    return problems
