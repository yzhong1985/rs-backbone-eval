"""CPU tests for the T14 change baseline: manifest rows, pair reading, crop/flip
sync, change features, change F1, the null ignore index and a stub encoder.

No dataset, checkpoint or GPU is touched. Run in the LIM worker env::

    uv run --project workers/lim --with pytest python -m pytest tests/test_lim_t14_train.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("rasterio")
import rasterio  # noqa: E402
import torch  # noqa: E402
from lim_worker import t14  # noqa: E402
from rasterio.transform import Affine  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from rs_backbone_eval.evaluation.confusion import ConfusionMatrix  # noqa: E402

AOI = "L15-TEST_0_0_13"
SIZE = 32


def _row(split: str = "train", stamps: tuple[str, str] = ("2018_01", "2018_02")) -> dict:
    stem = "global_monthly_{}_mosaic_" + AOI
    return {
        "task_id": "T14",
        "sample_id": f"{AOI}_{stamps[0]}_{stamps[1]}",
        "group_id": AOI,
        "split": split,
        "image_paths": [
            f"raw/spacenet7/train/{AOI}/images/{stem.format(stamps[0])}.tif",
            f"raw/spacenet7/train/{AOI}/images/{stem.format(stamps[1])}.tif",
        ],
        "label_path": f"derived/spacenet7-change/{AOI}/{stamps[0]}_{stamps[1]}.tif",
        "window": [0, 0, SIZE, SIZE],
        "timestamps": ["2018-01", "2018-02"],
        "transform": [4.78, 0.0, 0.0, 0.0, -4.78, 0.0],
        "crs": "EPSG:3857",
    }


def _write(path: Path, array: np.ndarray, count: int, dtype: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=array.shape[-1],
        height=array.shape[-2],
        count=count,
        dtype=dtype,
        crs="EPSG:3857",
        transform=Affine(4.78, 0.0, 0.0, 0.0, -4.78, 0.0),
    ) as dst:
        dst.write(array.astype(dtype).reshape(count, *array.shape[-2:]))


def _write_pair(root: Path, row: dict, mask: np.ndarray) -> None:
    """RGBA Planet-style images (t1 blue=100 and t2 red=200 where the mask is 1)."""
    t1 = np.zeros((4, SIZE, SIZE), dtype=np.uint8)
    t2 = np.zeros((4, SIZE, SIZE), dtype=np.uint8)
    t1[3] = t2[3] = 255  # alpha: all valid
    t1[2] = mask * 100
    t2[0] = mask * 200
    _write(root / row["image_paths"][0], t1, 4, "uint8")
    _write(root / row["image_paths"][1], t2, 4, "uint8")
    _write(root / row["label_path"], mask.astype(np.uint8), 1, "uint8")


def _random_mask(seed: int, size: int = SIZE) -> np.ndarray:
    return (np.random.default_rng(seed).random((size, size)) < 0.3).astype(np.int64)


def test_manifest_keeps_only_t14_rows_of_the_split(tmp_path: Path) -> None:
    other = dict(_row("train"), task_id="T13")
    rows = [_row("train"), _row("validation"), other]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert [r["split"] for r in t14.load_rows(manifest, "train")] == ["train"]
    assert len(t14.load_rows(manifest, "validation")) == 1
    assert t14.load_rows(manifest, "test") == []


def test_manifest_row_must_be_a_pair(tmp_path: Path) -> None:
    single = dict(_row(), image_paths=[_row()["image_paths"][0]])
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(single) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="two image paths"):
        t14.load_rows(manifest, "train")
    with pytest.raises(ValueError, match="window"):
        t14.validate_row(dict(_row(), window=[0, 0, SIZE]))


def test_read_pair_returns_rgb_and_binary_mask(tmp_path: Path) -> None:
    row = _row()
    mask = _random_mask(1)
    _write_pair(tmp_path, row, mask)
    t1, t2, label = t14.read_pair(tmp_path, row)
    assert t1.shape == t2.shape == (3, SIZE, SIZE) and t1.dtype == np.uint8
    assert label.dtype == np.int64 and np.array_equal(label, mask)
    assert np.array_equal(t2[0] == 200, mask == 1)  # alpha band is not read as RGB


def test_read_pair_rejects_a_mask_on_another_grid(tmp_path: Path) -> None:
    row = _row()
    _write_pair(tmp_path, row, _random_mask(2))
    _write(tmp_path / row["label_path"], np.zeros((SIZE - 1, SIZE), np.uint8), 1, "uint8")
    with pytest.raises(ValueError, match="differ"):
        t14.read_pair(tmp_path, row)


def test_crop_and_flip_keep_images_and_mask_in_sync() -> None:
    crop = 16
    for seed in range(40):
        mask = _random_mask(seed)
        t1 = np.zeros((3, SIZE, SIZE), np.uint8)
        t2 = np.zeros((3, SIZE, SIZE), np.uint8)
        t1[2] = mask * 100
        t2[0] = mask * 200
        rng = np.random.default_rng(seed)
        a, b, m = t14.crop_and_flip(t1, t2, mask, rng, crop)
        assert a.shape == b.shape == (3, crop, crop) and m.shape == (crop, crop)
        assert np.array_equal(b[0] == 200, m == 1)
        assert np.array_equal(a[2] == 100, m == 1)


def test_crop_and_flip_rejects_tiles_smaller_than_the_crop() -> None:
    small = np.zeros((3, 8, 8), np.uint8)
    with pytest.raises(ValueError, match="smaller"):
        t14.crop_and_flip(small, small, np.zeros((8, 8), np.int64), np.random.default_rng(0), 16)


def test_dataset_items_are_crops_and_change_with_epoch(tmp_path: Path) -> None:
    row = _row()
    _write_pair(tmp_path, row, _random_mask(3))
    dataset = t14.ChangePairs([row], tmp_path, crop=16, seed=5)
    first = dataset[0]
    assert first["t1"].shape == (3, 16, 16) and first["label"].dtype == torch.int64
    assert torch.equal(first["t2"][0] == 200, first["label"] == 1)
    seen = set()
    for epoch in range(6):
        dataset.set_epoch(epoch)
        seen.add(tuple(dataset[0]["label"].flatten().tolist()))
    assert len(seen) > 1  # the crop/flip draw is per epoch, not fixed


def test_change_features_are_absolute_differences() -> None:
    gen = torch.Generator().manual_seed(0)
    f1 = [torch.randn(2, c, 4, 4, generator=gen) for c in (4, 8)]
    f2 = [torch.randn(2, c, 4, 4, generator=gen) for c in (4, 8)]
    diff = t14.change_features(f1, f2)
    for got, a, b in zip(diff, f1, f2, strict=True):
        assert torch.equal(got, (b - a).abs())
        assert torch.equal(got, t14.change_features([b], [a])[0])  # symmetric
        assert torch.equal(t14.change_features([a], [a])[0], torch.zeros_like(a))
    with pytest.raises(ValueError, match="pyramid levels differ"):
        t14.change_features([torch.zeros(1, 4, 4, 4)], [torch.zeros(1, 4, 5, 4)])
    with pytest.raises(ValueError, match="depth"):
        t14.change_features(f1, f1[:1])


def test_change_f1_matches_a_hand_count() -> None:
    # gt  = 1 1 0 0 1, pred = 1 0 1 0 1
    # class 1: TP 2, FP 1, FN 1 -> precision 2/3, recall 2/3, F1 2/3, IoU 2/4
    # class 0: TP 1 (index 3), FP 1, FN 1 -> IoU 1/3; mean_iou = (1/2 + 1/3) / 2
    matrix = ConfusionMatrix(2)
    matrix.update(np.array([1, 1, 0, 0, 1]), np.array([1, 0, 1, 0, 1]))
    scores = t14.change_scores(matrix)
    assert scores["change_f1"] == pytest.approx(2 / 3)
    assert scores["change_iou"] == pytest.approx(0.5)
    assert scores["mean_iou"] == pytest.approx(5 / 12)


def test_score_rows_reads_pairs_and_scores_per_aoi(tmp_path: Path) -> None:
    row = _row()
    mask = _random_mask(4)
    _write_pair(tmp_path, row, mask)

    def perfect(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        target = torch.from_numpy(mask)[None]
        return F.one_hot(target, 2).permute(0, 3, 1, 2).float()

    scores, groups = t14.score_rows([row], tmp_path, perfect)
    assert scores["change_f1"] == pytest.approx(1.0)
    assert list(groups) == [AOI] and np.asarray(groups[AOI]).shape == (2, 2)

    def background(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(1, 2, *t1.shape[-2:])
        logits[:, 0] = 1.0  # always predicts class 0: no change predicted, F1 is 0
        return logits

    scores, _ = t14.score_rows([row], tmp_path, background)
    assert scores["change_f1"] == 0.0


def test_loss_with_null_ignore_index_counts_every_pixel() -> None:
    # Pixel A: logits (0, 2), target 0 -> log(1 + e^2). Pixel B: logits (0, 0), target 1 -> log 2.
    logits = torch.tensor([[[[0.0, 0.0]], [[2.0, 0.0]]]])
    target = torch.tensor([[[0, 1]]])
    expected_all = (np.log(1 + np.exp(2.0)) + np.log(2.0)) / 2
    assert float(t14.change_loss(logits, target, None)) == pytest.approx(expected_all, rel=1e-6)
    # An ignore index of 0 drops pixel A, so only pixel B remains.
    assert float(t14.change_loss(logits, target, 0)) == pytest.approx(np.log(2.0), rel=1e-6)


def test_cloud_pixels_have_no_loss_and_no_gradient() -> None:
    # Pixel A: logits (0, 2), target 0 -> log(1 + e^2). Pixel B: logits (0, 0), target 1 -> log 2.
    # Pixel C: logits (5, -5), target 255 (cloud) -> must contribute nothing.
    logits = torch.tensor([[[[0.0, 0.0, 5.0]], [[2.0, 0.0, -5.0]]]], requires_grad=True)
    target = torch.tensor([[[0, 1, t14.IGNORE_INDEX]]])
    loss = t14.change_loss(logits, target, t14.IGNORE_INDEX)
    expected = (np.log(1 + np.exp(2.0)) + np.log(2.0)) / 2
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-6)
    loss.backward()
    assert torch.all(logits.grad[:, :, :, 2] == 0)
    assert torch.any(logits.grad[:, :, :, 0] != 0)


def test_read_pair_accepts_cloud_255_and_rejects_other_values(tmp_path: Path) -> None:
    row = _row()
    mask = _random_mask(6)
    mask[0, :4] = t14.IGNORE_INDEX
    _write_pair(tmp_path, row, mask)
    _, _, label = t14.read_pair(tmp_path, row)
    assert np.array_equal(label, mask) and (label[0, :4] == 255).all()
    bad = _random_mask(7)
    bad[1, 1] = 2
    _write_pair(tmp_path, row, bad)
    with pytest.raises(ValueError, match="outside"):
        t14.read_pair(tmp_path, row)


def test_cloud_is_excluded_from_f1_iou_and_mean_iou() -> None:
    # The first five pixels are the hand count from test_change_f1_matches_a_hand_count.
    # The three cloud pixels get predictions that would be wrong if they were scored.
    truth = np.array([1, 1, 0, 0, 1, 255, 255, 255])
    pred = np.array([1, 0, 1, 0, 1, 0, 1, 1])
    matrix = ConfusionMatrix(2)
    matrix.update(truth, pred, ignore_index=t14.IGNORE_INDEX)
    assert matrix.total == 5
    scores = t14.change_scores(matrix)
    assert scores["change_f1"] == pytest.approx(2 / 3)
    assert scores["change_iou"] == pytest.approx(0.5)
    assert scores["mean_iou"] == pytest.approx(5 / 12)


def test_score_rows_excludes_cloud_pixels(tmp_path: Path) -> None:
    row = _row()
    mask = _random_mask(8)
    mask[:4, :] = t14.IGNORE_INDEX  # a cloud band across the top of the tile
    _write_pair(tmp_path, row, mask)

    def perfect_on_valid(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        # Correct wherever the mask is 0/1 and predicts change on every cloud pixel.
        target = torch.from_numpy(np.where(mask == t14.IGNORE_INDEX, 1, mask))[None]
        return F.one_hot(target, 2).permute(0, 3, 1, 2).float()

    scores, groups = t14.score_rows([row], tmp_path, perfect_on_valid)
    assert scores["change_f1"] == pytest.approx(1.0)
    assert np.asarray(groups[AOI]).sum() == int((mask != t14.IGNORE_INDEX).sum())


def test_yaml_ignore_index_matches_the_code_constant() -> None:
    from rs_backbone_eval.registry import Registry

    root = Path(__file__).resolve().parents[1]
    task = next(t for t in Registry(root / "configs").tasks() if t.task_id == "T14")
    assert task.ignore_index == t14.IGNORE_INDEX == 255


def test_effective_batch_is_eight_and_plan_keeps_it(monkeypatch: pytest.MonkeyPatch) -> None:
    assert t14.EFFECTIVE_BATCH == 8
    monkeypatch.delenv("LIM_WORKER_MICRO_BATCH", raising=False)
    monkeypatch.delenv("LIM_WORKER_GRAD_ACCUM", raising=False)
    assert t14.batch_plan() == (8, 1)
    monkeypatch.setenv("LIM_WORKER_MICRO_BATCH", "2")
    assert t14.batch_plan() == (2, 4)  # memory fallback: accumulate to the same 8
    monkeypatch.setenv("LIM_WORKER_GRAD_ACCUM", "2")
    with pytest.raises(ValueError, match="effective batch 8"):
        t14.batch_plan()


class StubEncoder(nn.Module):
    """Four strided convs with the LIM pyramid widths and strides 4, 8, 16, 32."""

    def __init__(self) -> None:
        super().__init__()
        self.s1 = nn.Conv2d(3, 256, 4, 4)
        self.s2 = nn.Conv2d(256, 512, 2, 2)
        self.s3 = nn.Conv2d(512, 1024, 2, 2)
        self.s4 = nn.Conv2d(1024, 2048, 2, 2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.s1(x)
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        return [f1, f2, f3, self.s4(f3)]


def test_siamese_head_is_symmetric_and_only_the_head_gets_gradients() -> None:
    torch.manual_seed(0)
    encoder = StubEncoder().eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    head = t14.ChangeHead()
    norm = t14.Normalisation("cpu", torch.float32)
    t1 = torch.randint(0, 255, (2, 3, 128, 128), dtype=torch.uint8)
    t2 = torch.randint(0, 255, (2, 3, 128, 128), dtype=torch.uint8)
    f1 = t14.encode(encoder, t1, norm)
    f2 = t14.encode(encoder, t2, norm)
    assert [tuple(f.shape[-2:]) for f in f1] == [(32, 32), (16, 16), (8, 8), (4, 4)]
    assert [f.shape[1] for f in f1] == list(t14.CHANNELS)
    head.eval()
    logits = head(f1, f2, output_size=(128, 128))
    assert logits.shape == (2, 2, 128, 128)
    assert torch.allclose(logits, head(f2, f1, output_size=(128, 128)), atol=1e-6)
    head.train()
    labels = torch.randint(0, 2, (2, 128, 128))
    t14.change_loss(head(f1, f2, output_size=(128, 128)), labels, None).backward()
    assert all(p.grad is not None for p in head.parameters())
    assert all(p.grad is None for p in encoder.parameters())


def test_probe_pair_reports_both_dates_levels_and_strides(tmp_path: Path) -> None:
    row = _row("test")
    _write_pair(tmp_path, row, _random_mask(3))
    report = t14.probe_pair(StubEncoder().eval(), row, tmp_path, "cpu", torch.float32)
    assert report["sample"] == row["sample_id"]
    assert report["tile_px"] == [SIZE, SIZE]
    assert [lv["height"] for lv in report["levels"]] == [8, 4, 2, 1]
    assert [lv["stride"] for lv in report["levels"]] == [
        [4.0, 4.0],
        [8.0, 8.0],
        [16.0, 16.0],
        [32.0, 32.0],
    ]
    assert [lv["channels"] for lv in report["levels"]] == list(t14.CHANNELS)
    assert report["channels_match_protocol"] is True
    assert report["finite"] is True
    assert report["dtype"] == "float32"


def test_probe_pair_fails_on_a_mismatched_mask_grid(tmp_path: Path) -> None:
    row = _row("test")
    _write_pair(tmp_path, row, _random_mask(4))
    _write(tmp_path / row["label_path"], np.zeros((SIZE - 1, SIZE), dtype=np.uint8), 1, "uint8")
    with pytest.raises(ValueError, match="differ"):
        t14.probe_pair(StubEncoder().eval(), row, tmp_path, "cpu", torch.float32)


def test_runner_t14_probe_writes_report_and_completes(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from lim_worker import runner

    row = _row("test")
    _write_pair(tmp_path, row, _random_mask(5))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    class Writer:
        def __init__(self) -> None:
            self.artifacts: list[str] = []
            self.completed: str | None = None
            self.failed: str | None = None

        def add_artifact(self, kind: str, path: str, title: str | None = None) -> None:
            self.artifacts.append(path)

        def complete(self, notes: str | None = None) -> None:
            self.completed = notes

        def fail(self, message: str) -> None:
            self.failed = message

    request = SimpleNamespace(
        run_id="t14-probe-smoke", manifest_path=str(manifest), data_root=str(tmp_path)
    )
    writer = Writer()
    model = SimpleNamespace(encoder=StubEncoder().eval())
    rc = runner.probe_t14(request, writer, lambda *a, **k: None, model, "cpu", run_dir)
    assert rc == 0
    assert writer.failed is None and writer.completed.startswith("probe passed")
    assert writer.artifacts == ["probe_report.json"]
    report = json.loads((run_dir / "probe_report.json").read_text(encoding="utf-8"))
    assert [lv["height"] for lv in report["levels"]] == [8, 4, 2, 1]


def test_runner_t14_probe_fails_readably_without_test_pairs(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from lim_worker import runner

    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(_row("train")) + "\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    class Writer:
        failed: str | None = None

        def add_artifact(self, *args, **kwargs) -> None:
            raise AssertionError("no artifact on a failed probe")

        def complete(self, notes: str | None = None) -> None:
            raise AssertionError("a probe without test pairs must not complete")

        def fail(self, message: str) -> None:
            self.failed = message

    writer = Writer()
    request = SimpleNamespace(
        run_id="t14-probe-empty", manifest_path=str(manifest), data_root=str(tmp_path)
    )
    model = SimpleNamespace(encoder=StubEncoder().eval())
    rc = runner.probe_t14(request, writer, lambda *a, **k: None, model, "cpu", run_dir)
    assert rc == 1
    assert "no T14 test pairs" in writer.failed
