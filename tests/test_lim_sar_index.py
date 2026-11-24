"""CPU tests for the pinned WHU SAR index loader (LIM worker env, tmp_path only).

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_sar_index.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")
from lim_worker.sar_index import (  # noqa: E402
    SarIndexError,
    canonical_entries_digest,
    file_sha256,
    index_path,
    load_sar_index,
)


def _write_sar(root: Path, group: str, width: int = 8, height: int = 6) -> dict:
    rel = f"raw/whu-opt-sar/sar/{group}.tif"
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    data = np.arange(width * height, dtype=np.uint8).reshape(height, width)
    with rasterio.open(
        target,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="uint8",
    ) as dst:
        dst.write(data, 1)
    return {
        "group_id": group,
        "path": rel,
        "sha256": file_sha256(target),
        "width": width,
        "height": height,
        "bands": 1,
        "dtype": "uint8",
        "nodata": None,
    }


def _write_index(root: Path, entries: list[dict]) -> str:
    digest = canonical_entries_digest(entries)
    path = index_path(root, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"index_sha256": digest, "entries": entries}, indent=2), encoding="utf-8"
    )
    return digest


def test_loads_selected_groups_and_matches_disk(tmp_path: Path) -> None:
    entries = [_write_sar(tmp_path, "NH49E001"), _write_sar(tmp_path, "NH49E002")]
    digest = _write_index(tmp_path, entries)
    got = load_sar_index(tmp_path, digest, {"NH49E002"}, verify_files=True)
    assert set(got) == {"NH49E002"}
    assert got["NH49E002"].path.endswith("NH49E002.tif")


def test_refuses_index_whose_entries_were_edited(tmp_path: Path) -> None:
    entries = [_write_sar(tmp_path, "NH49E001")]
    digest = _write_index(tmp_path, entries)
    path = index_path(tmp_path, digest)
    payload = json.loads(path.read_text())
    payload["entries"][0]["width"] = 9999
    path.write_text(json.dumps(payload))
    with pytest.raises(SarIndexError, match="hash to"):
        load_sar_index(tmp_path, digest, {"NH49E001"})


def test_refuses_missing_group(tmp_path: Path) -> None:
    digest = _write_index(tmp_path, [_write_sar(tmp_path, "NH49E001")])
    with pytest.raises(SarIndexError, match="no SAR entry"):
        load_sar_index(tmp_path, digest, {"NH49E001", "NH49E999"})


def test_refuses_sheet_whose_size_changed_on_disk(tmp_path: Path) -> None:
    entry = _write_sar(tmp_path, "NH49E001")
    entry["width"] = 7  # pinned size no longer matches the file
    digest = _write_index(tmp_path, [entry])
    with pytest.raises(SarIndexError, match="on disk"):
        load_sar_index(tmp_path, digest, {"NH49E001"})


def test_refuses_multi_band_entry(tmp_path: Path) -> None:
    entry = _write_sar(tmp_path, "NH49E001")
    entry["bands"] = 3
    digest = _write_index(tmp_path, [entry])
    with pytest.raises(SarIndexError, match="expected 1 uint8"):
        load_sar_index(tmp_path, digest, {"NH49E001"})


def test_verify_files_catches_content_change_with_same_size(tmp_path: Path) -> None:
    entry = _write_sar(tmp_path, "NH49E001")
    digest = _write_index(tmp_path, [entry])
    target = tmp_path / entry["path"]
    with rasterio.open(target, "r+") as dst:
        dst.write(np.full((6, 8), 255, dtype=np.uint8), 1)
    load_sar_index(tmp_path, digest, {"NH49E001"})  # size check alone passes
    with pytest.raises(SarIndexError, match="SHA-256"):
        load_sar_index(tmp_path, digest, {"NH49E001"}, verify_files=True)


def test_refuses_file_whose_dtype_differs_from_the_index(tmp_path: Path) -> None:
    entry = _write_sar(tmp_path, "NH49E001")
    target = tmp_path / entry["path"]
    with rasterio.open(
        target, "w", driver="GTiff", width=8, height=6, count=1, dtype="uint16"
    ) as dst:
        dst.write(np.zeros((6, 8), dtype=np.uint16), 1)
    entry["sha256"] = file_sha256(target)  # the hash now matches the uint16 bytes
    digest = _write_index(tmp_path, [entry])
    with pytest.raises(SarIndexError, match="dtype"):
        load_sar_index(tmp_path, digest, {"NH49E001"}, verify_files=True)


def test_refuses_entry_whose_path_belongs_to_another_sheet(tmp_path: Path) -> None:
    entry = _write_sar(tmp_path, "NH49E002")
    entry["group_id"] = "NH49E001"  # same bytes, filed under the wrong sheet
    digest = _write_index(tmp_path, [entry])
    with pytest.raises(SarIndexError, match="expected"):
        load_sar_index(tmp_path, digest, {"NH49E001"})
