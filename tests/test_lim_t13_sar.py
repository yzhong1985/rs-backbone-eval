"""CPU tests for the T13 SAR data path (LIM worker env, tmp_path only).

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_t13_sar.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")
from lim_worker.sar_index import canonical_entries_digest, file_sha256, index_path  # noqa: E402
from lim_worker.t13_data import (  # noqa: E402
    T13Tiles,
    flip_plane,
    load_sar_entries,
    read_sar,
    sar_statistics,
)

SIZE = 512
GROUP = "NH49E001"


def _write_tif(path: Path, data: np.ndarray, count: int, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = data.shape[-2:]
    with rasterio.open(
        path, "w", driver="GTiff", width=width, height=height, count=count, dtype=dtype
    ) as dst:
        if count == 1:
            dst.write(data.reshape(height, width), 1)
        else:
            dst.write(data)


@pytest.fixture()
def dataset(tmp_path: Path):
    rng = np.random.default_rng(0)
    optical = rng.integers(0, 255, size=(4, 1024, 1024), dtype=np.uint8)
    sar = rng.integers(0, 255, size=(1024, 1024), dtype=np.uint8)
    labels = (rng.integers(0, 8, size=(1024, 1024)) * 10).astype(np.uint8)
    _write_tif(tmp_path / "raw/whu-opt-sar/optical" / f"{GROUP}.tif", optical, 4, "uint8")
    _write_tif(tmp_path / "raw/whu-opt-sar/lbl" / f"{GROUP}.tif", labels, 1, "uint8")
    sar_rel = f"raw/whu-opt-sar/sar/{GROUP}.tif"
    _write_tif(tmp_path / sar_rel, sar, 1, "uint8")
    entry = {
        "group_id": GROUP,
        "path": sar_rel,
        "sha256": file_sha256(tmp_path / sar_rel),
        "width": 1024,
        "height": 1024,
        "bands": 1,
        "dtype": "uint8",
        "nodata": None,
    }
    digest = canonical_entries_digest([entry])
    index = index_path(tmp_path, digest)
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"index_sha256": digest, "entries": [entry]}), encoding="utf-8")
    rows = [
        {
            "task_id": "T13",
            "sample_id": f"{GROUP}_x{x:04d}_y0000",
            "group_id": GROUP,
            "split": "train",
            "image_paths": [f"raw/whu-opt-sar/optical/{GROUP}.tif"],
            "label_path": f"raw/whu-opt-sar/lbl/{GROUP}.tif",
            "window": [x, 0, SIZE, SIZE],
        }
        for x in (0, 512)
    ]
    return tmp_path, rows, sar.astype(np.float64), digest


def test_read_sar_returns_the_optical_window(dataset) -> None:
    root, rows, sar, digest = dataset
    entries = load_sar_entries(root, digest, rows, formal=True)
    got = read_sar(root, rows[1], entries[GROUP])
    assert got.shape == (1, SIZE, SIZE)
    assert np.array_equal(got[0], sar[0:SIZE, 512 : 512 + SIZE].astype(np.uint8))


def test_read_sar_refuses_a_row_from_another_sheet(dataset) -> None:
    root, rows, _, digest = dataset
    entries = load_sar_entries(root, digest, rows, formal=False)
    wrong = dict(rows[0], group_id="NH49E999")
    with pytest.raises(ValueError, match="SAR entry is for"):
        read_sar(root, wrong, entries[GROUP])


def test_item_carries_normalised_sar_and_all_ones_validity(dataset) -> None:
    root, rows, sar, digest = dataset
    entries = load_sar_entries(root, digest, rows, formal=False)
    mean, std = sar_statistics(rows, root, entries, samples=2, seed=0)
    ds = T13Tiles(
        rows,
        root,
        with_aef=False,
        augment=False,
        sar_entries=entries,
        sar_stats=(float(mean), float(std)),
    )
    item = ds[1]
    expected = (sar[0:SIZE, 512:] - float(mean)) / float(std)
    assert item["sar"].shape == (1, SIZE, SIZE)
    assert np.allclose(item["sar"][0].numpy(), expected, atol=1e-5)
    assert bool((item["sar_valid"] == 1).all())


def test_sar_flips_follow_the_image_flips(dataset) -> None:
    root, rows, _, digest = dataset
    entries = load_sar_entries(root, digest, rows, formal=False)
    ds = T13Tiles(
        rows,
        root,
        with_aef=False,
        augment=True,
        seed=7,
        sar_entries=entries,
        sar_stats=(0.0, 1.0),
    )
    plain = T13Tiles(
        rows, root, with_aef=False, augment=False, sar_entries=entries, sar_stats=(0.0, 1.0)
    )
    for index in range(len(rows)):
        flips = ds._flips(index)
        expected = flip_plane(plain[index]["sar"].numpy(), flips)
        assert np.array_equal(ds[index]["sar"].numpy(), expected)


def test_sar_entries_and_stats_must_come_together(dataset) -> None:
    root, rows, _, digest = dataset
    entries = load_sar_entries(root, digest, rows, formal=False)
    with pytest.raises(ValueError, match="together"):
        T13Tiles(rows, root, with_aef=False, augment=False, sar_entries=entries)


def test_formal_loading_checks_file_hashes(dataset) -> None:
    root, rows, _, digest = dataset
    target = root / f"raw/whu-opt-sar/sar/{GROUP}.tif"
    with rasterio.open(target, "r+") as dst:
        dst.write(np.full((1024, 1024), 9, dtype=np.uint8), 1)
    load_sar_entries(root, digest, rows, formal=False)  # smoke path: size check only
    from lim_worker.sar_index import SarIndexError

    with pytest.raises(SarIndexError, match="SHA-256"):
        load_sar_entries(root, digest, rows, formal=True)
