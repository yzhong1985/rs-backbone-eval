"""CPU tests for the T13 data layer and sheet scoring (LIM worker env).

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_t13.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import rasterio  # noqa: E402
import torch  # noqa: E402
from lim_worker import t13  # noqa: E402
from lim_worker.t13_data import (  # noqa: E402
    AEF_DIR,
    T13Tiles,
    band_statistics,
    donor_rows,
    flip,
    read_aef,
    read_label,
)

W, H, CROP = 96, 64, 64


def _write(path: Path, array: np.ndarray, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[-1],
        height=array.shape[-2],
        count=array.shape[0],
        dtype=dtype,
    ) as dst:
        dst.write(array.astype(dtype))


def fake_sheet(root: Path, name: str = "NH49E004021", seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (4, H, W))
    label = rng.integers(0, 8, (1, H, W)) * 10
    label[0, :4, :4] = 0
    aef = rng.integers(-127, 127, (64, H // 2, W // 2))
    aef[:, 0, 0] = -128
    _write(root / "raw/whu/optical" / f"{name}.tif", image, "uint8")
    _write(root / "raw/whu/lbl" / f"{name}.tif", label, "uint8")
    _write(root / "derived" / AEF_DIR / f"{name}.tif", aef, "int8")
    rows = []
    for x in (0, W - CROP):  # clamped last tile, overlapping the first
        rows.append(
            {
                "sample_id": f"{name}_x{x}",
                "group_id": name,
                "split": "test",
                "image_paths": [f"raw/whu/optical/{name}.tif"],
                "label_path": f"raw/whu/lbl/{name}.tif",
                "window": [x, 0, CROP, CROP],
            }
        )
    return rows


def test_label_mapping_and_aef_window(tmp_path: Path) -> None:
    rows = fake_sheet(tmp_path)
    with rasterio.open(tmp_path / rows[1]["label_path"]) as src:
        raw = src.read(1)
    assert np.array_equal(read_label(tmp_path, rows[1]), raw[:, W - CROP :] // 10)
    aef = read_aef(tmp_path, rows[1])
    assert aef.shape == (64, CROP // 2, CROP // 2) and aef.dtype == np.int8
    with rasterio.open(tmp_path / "derived" / AEF_DIR / f"{rows[1]['group_id']}.tif") as src:
        assert np.array_equal(aef, src.read()[:, :, (W - CROP) // 2 :])
    odd = dict(rows[0], window=[1, 0, CROP, CROP])
    with pytest.raises(ValueError, match="AEF grid"):
        read_aef(tmp_path, odd)


def test_bad_label_values_are_rejected(tmp_path: Path) -> None:
    rows = fake_sheet(tmp_path)
    _write(tmp_path / rows[0]["label_path"], np.full((1, H, W), 15), "uint8")
    with pytest.raises(ValueError, match="label values"):
        read_label(tmp_path, rows[0])


def test_flip_moves_image_label_and_aef_together() -> None:
    image = np.arange(4 * 8 * 8).reshape(4, 8, 8)
    label = np.arange(64).reshape(8, 8)
    aef = np.arange(64 * 4 * 4).reshape(64, 4, 4)
    fi, fl, fa = flip(image, label, aef, (True, True))
    assert np.array_equal(fi, image[:, ::-1, ::-1]) and np.array_equal(fl, label[::-1, ::-1])
    assert np.array_equal(fa, aef[:, ::-1, ::-1])


def test_dataset_flips_are_reproducible_per_epoch(tmp_path: Path) -> None:
    rows = fake_sheet(tmp_path)
    ds = T13Tiles(rows, tmp_path, with_aef=True, augment=True, seed=17)
    flips_0 = [ds._flips(i) for i in range(2)]
    assert flips_0 == [ds._flips(i) for i in range(2)]
    item = ds[0]
    assert item["image"].shape == (4, CROP, CROP) and item["aef"].shape == (
        64,
        CROP // 2,
        CROP // 2,
    )


def test_band_statistics_use_labelled_train_pixels(tmp_path: Path) -> None:
    rows = fake_sheet(tmp_path)
    mean, std = band_statistics(rows, tmp_path, samples=8, seed=0)
    assert mean.shape == std.shape == (4,) and np.all(std > 0)


def test_score_sheets_averages_overlap_and_never_predicts_background(tmp_path: Path) -> None:
    rows = fake_sheet(tmp_path)
    with rasterio.open(tmp_path / rows[0]["label_path"]) as src:
        truth = src.read(1) // 10
    onehot = np.eye(8, dtype=np.float32)[truth].transpose(2, 0, 1)
    onehot[0] += 5.0  # a model that loves background must still not predict it

    seen = iter(rows)  # batch=1: one call per tile, in manifest order

    def predict(image: torch.Tensor, codes) -> torch.Tensor:
        x = next(seen)["window"][0]
        return torch.from_numpy(onehot[:, :, x : x + CROP])[None]

    metrics, groups = t13.score_sheets(rows, tmp_path, predict, with_aef=False, batch=1)
    assert metrics["mean_iou"] == pytest.approx(1.0)
    counts = np.array(groups["NH49E004021"])
    assert counts[0].sum() == 0 and counts[:, 0].sum() == 0
    assert counts.sum() == int((truth != 0).sum())


def test_aef_input_modes() -> None:
    codes = torch.randint(-127, 127, (3, 64, 4, 4), dtype=torch.int8)
    real = t13.aef_input(codes, "real")
    assert torch.equal(t13.aef_input(codes, "zero"), torch.zeros_like(real))
    # the shuffled control reads donor codes upstream; here it decodes like real
    assert torch.equal(t13.aef_input(codes, "shuffle"), real)
    with pytest.raises(ValueError):
        t13.aef_input(codes, "batch-roll")


def test_shuffle_donors_come_from_another_sheet_at_the_same_position() -> None:
    sheets = [f"S{i:02d}" for i in range(6)]
    rows = [
        {"sample_id": f"{s}_{x}_{y}", "group_id": s, "window": [x, y, 512, 512]}
        for s in sheets
        for y in (0, 384)
        for x in (0, 384, 768)
    ]
    donors = donor_rows(rows)
    assert set(donors) == {r["sample_id"] for r in rows}
    for row in rows:
        donor = donors[row["sample_id"]]
        assert donor["group_id"] != row["group_id"]
        assert donor["window"] == row["window"]
    # half the list away: S00 <- S03, never a neighbour in the sorted list
    assert donors["S00_0_0"]["group_id"] == "S03"
    assert donor_rows(list(reversed(rows))) == donors
    with pytest.raises(ValueError):
        donor_rows([r for r in rows if r["group_id"] == "S00"])


def test_flips_change_per_epoch_with_dataloader_workers(tmp_path: Path) -> None:
    rows = fake_sheet(tmp_path)
    ds = T13Tiles(rows, tmp_path, with_aef=False, augment=True, seed=3)
    loader = t13.train_loader(ds, batch=2, workers=2, seed=0)
    assert not loader.persistent_workers
    flips = []
    for epoch in range(6):
        ds.set_epoch(epoch)
        batch = next(iter(loader))
        index = int(batch["index"][0])
        flips.append(ds._flips(index))
        # the worker's copy must agree with the main process for this epoch
        expected = ds[index]["image"]
        assert torch.equal(batch["image"][0], expected)
    assert len(set(flips)) > 1


MANIFEST_SHA = "m" * 64
BACKBONE_SHA = "b" * 64


def test_dependency_lookup_needs_exactly_one_clean_run(tmp_path: Path) -> None:
    def run(
        name: str,
        *,
        seed: int = 17,
        notes: str = "",
        tags: dict | None = None,
        manifest: str = MANIFEST_SHA,
        backbone: str = BACKBONE_SHA,
        epochs: int = 20,
    ) -> None:
        record = {
            "task_id": "T13",
            "protocol_id": "whu_b0",
            "seed": seed,
            "model_id": "lim",
            "status": "completed",
            "model_checkpoint_sha256": backbone,
            "dataset": {"manifest_sha256": manifest},
            "training": {"epochs_completed": epochs},
            "tags": {"epochs_budget": str(epochs), "effective_batch": "8", **(tags or {})},
            "notes": notes,
        }
        (tmp_path / name).mkdir()
        (tmp_path / name / "run.json").write_text(json.dumps(record))
        (tmp_path / name / "best_checkpoint.pt").write_bytes(b"x")

    kwargs = {
        "task_id": "T13",
        "protocol_id": "whu_b0",
        "seed": 17,
        "model_id": "lim",
        "manifest_sha": MANIFEST_SHA,
        "backbone_sha": BACKBONE_SHA,
        "epochs": 20,
        "batch": 8,
    }
    with pytest.raises(RuntimeError, match="found 0"):
        t13.find_dependency(tmp_path, **kwargs)
    run("a")
    run("smoke", notes="... [SMOKE]")
    run("old", tags={"superseded": "by a"})
    run("other-seed", seed=42)
    run("other-manifest", manifest="x" * 64)
    run("other-backbone", backbone="y" * 64)
    run("one-epoch", epochs=1)
    assert t13.find_dependency(tmp_path, **kwargs).parent.name == "a"
    assert t13.find_dependency(tmp_path, **kwargs, smoke=True).parent.name == "smoke"
    run("b")
    with pytest.raises(RuntimeError, match="found 2"):
        t13.find_dependency(tmp_path, **kwargs)
