"""Hash-pinned index of the WHU-OPT-SAR SAR sheets (T13 SAR branch, CPU-side).

The index lives under ``data_root/derived/whu-sar-index/<index_sha256>.json``.
Its ``index_sha256`` is the SHA-256 of ``entries`` serialised as
``json.dumps(entries, sort_keys=True, separators=(",", ":"))``, which matches
the digest of the 100-sheet index already on disk.
The file name carries the same digest.

Each entry pins one SAR sheet by ``group_id`` (the sheet code shared with
the optical tile rows): relative ``path``, file ``sha256``, ``width``,
``height``, ``bands``, ``dtype`` and ``nodata``.

``load_sar_index`` refuses an index whose self-hash does not match, whose
entries do not cover the requested groups, or whose sheet on disk no longer
matches the pinned size or band count. Full file hashing is opt-in
(``verify_files=True``) because the 100 sheets are about 2 GB.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rasterio

INDEX_DIR = "whu-sar-index"
SAR_DIR = "raw/whu-opt-sar/sar"
EXPECTED_BANDS = 1
EXPECTED_DTYPE = "uint8"


class SarIndexError(ValueError):
    """The SAR index is missing, self-inconsistent, or does not match the data on disk."""


@dataclass(frozen=True)
class SarEntry:
    group_id: str
    path: str
    sha256: str
    width: int
    height: int
    bands: int
    dtype: str
    nodata: int | None


def canonical_entries_digest(entries: list[dict[str, Any]]) -> str:
    blob = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def index_path(data_root: Path, index_sha256: str) -> Path:
    return data_root / "derived" / INDEX_DIR / f"{index_sha256}.json"


def load_sar_index(
    data_root: Path,
    index_sha256: str,
    groups: set[str],
    *,
    verify_files: bool = False,
) -> dict[str, SarEntry]:
    """Return ``{group_id: SarEntry}`` for every group in ``groups``.

    Raises ``SarIndexError`` on any mismatch. ``groups`` is the set of sheets
    the run reads (every ``group_id`` in the manifest rows), so a missing
    sheet fails before training starts rather than in the middle of an epoch.
    """
    path = index_path(data_root, index_sha256)
    if not path.is_file():
        raise SarIndexError(f"SAR index not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SarIndexError(f"SAR index unreadable: {path}: {exc}") from exc
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise SarIndexError(f"{path.name}: no entries list")
    if payload.get("index_sha256") != index_sha256:
        raise SarIndexError(f"{path.name}: stored index_sha256 differs from the file name")
    digest = canonical_entries_digest(entries)
    if digest != index_sha256:
        raise SarIndexError(f"{path.name}: entries hash to {digest}, not {index_sha256}")

    by_group: dict[str, SarEntry] = {}
    for raw in entries:
        entry = _entry(raw)
        if entry.group_id in by_group:
            raise SarIndexError(f"duplicate SAR group {entry.group_id}")
        by_group[entry.group_id] = entry

    missing = sorted(groups - set(by_group))
    if missing:
        raise SarIndexError(
            f"{len(missing)} manifest group(s) have no SAR entry, e.g. {missing[:3]}"
        )

    selected = {group: by_group[group] for group in sorted(groups)}
    for entry in selected.values():
        _check_on_disk(data_root, entry, verify_files=verify_files)
    return selected


def _entry(raw: dict[str, Any]) -> SarEntry:
    try:
        entry = SarEntry(
            group_id=str(raw["group_id"]),
            path=str(raw["path"]),
            sha256=str(raw["sha256"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            bands=int(raw["bands"]),
            dtype=str(raw["dtype"]),
            nodata=None if raw.get("nodata") is None else int(raw["nodata"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SarIndexError(f"malformed SAR entry {raw!r}: {exc}") from exc
    if entry.bands != EXPECTED_BANDS or entry.dtype != EXPECTED_DTYPE:
        raise SarIndexError(
            f"{entry.group_id}: expected {EXPECTED_BANDS} {EXPECTED_DTYPE} band(s), "
            f"got {entry.bands} {entry.dtype}"
        )
    return entry


def _check_on_disk(data_root: Path, entry: SarEntry, *, verify_files: bool) -> None:
    # Identity: the file must be the sheet of the same name. A hash only says the
    # bytes match what the index recorded, not which sheet they belong to.
    expected = f"{SAR_DIR}/{entry.group_id}.tif"
    if entry.path != expected:
        raise SarIndexError(f"{entry.group_id}: index path {entry.path!r}, expected {expected!r}")
    target = data_root / entry.path
    if not target.is_file():
        raise SarIndexError(f"{entry.group_id}: SAR file missing: {target}")
    with rasterio.open(target) as src:
        if (src.width, src.height) != (entry.width, entry.height) or src.count != entry.bands:
            raise SarIndexError(
                f"{entry.group_id}: on disk {src.width}x{src.height}x{src.count}, "
                f"index says {entry.width}x{entry.height}x{entry.bands}"
            )
        # The index's dtype and nodata are claims; the file must agree with them.
        actual_dtype = src.dtypes[0]
        if actual_dtype != entry.dtype:
            raise SarIndexError(
                f"{entry.group_id}: on disk dtype {actual_dtype}, index says {entry.dtype}"
            )
        if src.nodata != entry.nodata:
            raise SarIndexError(
                f"{entry.group_id}: on disk nodata {src.nodata}, index says {entry.nodata}"
            )
    if verify_files and file_sha256(target) != entry.sha256:
        raise SarIndexError(f"{entry.group_id}: SHA-256 differs from the index")


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()
