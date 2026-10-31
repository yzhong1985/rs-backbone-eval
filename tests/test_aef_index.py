"""AEF tile index: official slice handling and lookup helpers (no pyarrow needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rs_backbone_eval.data.aef_index import (
    AefIndexError,
    compare_indexes,
    find_cog,
    index_path,
    load_index,
    official_slice,
    write_slice,
)


def sample_index(source: str = "official") -> dict:
    return {
        "source": source,
        "year": 2021,
        "zone": "54N",
        "crs": ["EPSG:32654"],
        "cog_count": 2,
        "entries": [
            {
                "key": "2021/54N/a.tiff",
                "path": "gs://b/2021/54N/a.tiff",
                "bounds": [0.0, 0.0, 81920.0, 81920.0],
                "crs": "EPSG:32654",
            },
            {
                "key": "2021/54N/b.tiff",
                "path": "gs://b/2021/54N/b.tiff",
                "bounds": [81920.0, 0.0, 163840.0, 81920.0],
                "crs": "EPSG:32654",
            },
        ],
    }


def test_find_cog_uses_utm_bounds() -> None:
    index = sample_index()
    assert find_cog(index, 10.0, 10.0)["key"] == "2021/54N/a.tiff"
    assert find_cog(index, 100000.0, 10.0)["key"] == "2021/54N/b.tiff"
    assert find_cog(index, -1.0, 10.0) is None


def test_slice_round_trips_through_disk(tmp_path: Path) -> None:
    index = sample_index()
    write_slice(index, tmp_path)
    loaded = load_index(index_path(tmp_path, 2021, "54N"))
    assert loaded["cog_count"] == 2
    assert [entry["key"] for entry in loaded["entries"]] == ["2021/54N/a.tiff", "2021/54N/b.tiff"]


def test_compare_reports_missing_and_shifted_tiles() -> None:
    official = sample_index("official")
    scanned = sample_index("scan")
    assert compare_indexes(official, scanned) == []

    scanned["entries"][1]["bounds"] = [81920.0, 0.0, 163840.0, 90000.0]
    problems = compare_indexes(official, scanned)
    assert len(problems) == 1 and "bounds differ" in problems[0]

    scanned["entries"].pop()
    problems = compare_indexes(official, scanned)
    assert any("only in official" in problem for problem in problems)


def test_official_slice_explains_the_missing_dependency(tmp_path: Path) -> None:
    parquet = tmp_path / "aef_index.parquet"
    parquet.write_bytes(b"not really parquet")
    with pytest.raises(AefIndexError):
        official_slice(parquet, 2021, "54N")


def test_slice_is_valid_json(tmp_path: Path) -> None:
    write_slice(sample_index(), tmp_path)
    payload = json.loads((tmp_path / "2021_54N.json").read_text(encoding="utf-8"))
    assert payload["source"] == "official"
