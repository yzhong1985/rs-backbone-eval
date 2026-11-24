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
        "label_path": f"derived/spacenet7-change-v2/{AOI}/{stamps[0]}_{stamps[1]}.tif",
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


class _Items(torch.utils.data.Dataset):
    """Sample ``i`` carries ``i`` in its first feature, so forwarded samples can be read back."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict:
        value = torch.tensor([float(index), 1.0])
        return {"t1": value, "t2": value, "label": value}


def _epoch_loader(n_samples: int, micro: int):
    """The loader ``train_t14`` builds: fixed micro batch, ``drop_last=True``, no shuffle."""
    from torch.utils.data import DataLoader

    return DataLoader(_Items(n_samples), batch_size=micro, shuffle=False, drop_last=True)


def _run_epoch_recording(loader, accumulation: int, limit: int, scheduler=None, optimizer=None):
    """Runs one epoch on a linear head; returns (steps, forwarded t1 indices, update count)."""
    torch.manual_seed(0)
    head = nn.Linear(2, 2)
    if optimizer is None:
        optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    calls = {"n": 0}

    def count_update(opt, args, kwargs) -> None:
        calls["n"] += 1

    optimizer.register_step_post_hook(count_update)
    forwarded: list[float] = []

    def predict(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        forwarded.extend(t1[:, 0].tolist())
        return head(t1)

    def loss_fn(out: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return ((out - labels) ** 2).mean()

    _, steps = t14.run_train_epoch(
        loader, "cpu", predict, loss_fn, optimizer, scheduler, accumulation, limit
    )
    return steps, forwarded, calls["n"]


@pytest.mark.parametrize("limit", [0, 5], ids=["full-epoch", "limited"])
@pytest.mark.parametrize("accumulation", [1, 2, 4])
@pytest.mark.parametrize("micro", [2, 4])
@pytest.mark.parametrize("n_samples", [8, 10, 11])
def test_steps_are_floor_of_micro_batches_over_accumulation(
    n_samples: int, micro: int, accumulation: int, limit: int
) -> None:
    from rs_backbone_eval.schedule import build_scheduler

    # The loader drops the partial last micro batch, so an epoch has floor(N / micro) of them.
    micro_batches = n_samples // micro
    if limit:
        micro_batches = min(micro_batches, limit)
    expected_steps = micro_batches // accumulation
    used_samples = expected_steps * accumulation * micro

    loader = _epoch_loader(n_samples, micro)
    head = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    scheduler = build_scheduler(optimizer, total_steps=max(3 * expected_steps, 3), warmup_steps=1)
    steps, forwarded, updates = _run_epoch_recording(
        loader, accumulation, limit, scheduler=scheduler, optimizer=optimizer
    )

    assert steps == expected_steps == updates
    assert scheduler.last_epoch == expected_steps  # the schedule advances once per real update
    # The trailing micro batches get no forward pass: only the first full groups are seen.
    assert forwarded == [float(i) for i in range(used_samples)]


def test_default_epoch_runs_1064_of_1070_samples() -> None:
    # Default plan: micro batch 8, accumulation 1. 1070 // 8 = 133 micro batches = 1064 samples.
    loader = _epoch_loader(1070, 8)
    assert len(loader) == 133
    steps, forwarded, updates = _run_epoch_recording(loader, accumulation=1, limit=0)
    assert steps == updates == 133
    assert len(forwarded) == 1064
    assert forwarded == [float(i) for i in range(1064)]


def test_change_weight_scales_only_the_change_class():
    logits = torch.zeros(1, 2, 1, 2)  # uniform prediction: each pixel costs log(2)
    target = torch.tensor([[[0, 1]]])
    plain = float(t14.change_loss(logits, target, None))
    weighted = float(t14.change_loss(logits, target, None, class_weight=10.0))
    # weighted mean: (1*log2 + 10*log2) / (1 + 10)
    assert plain == pytest.approx(np.log(2.0), rel=1e-6)
    assert weighted == pytest.approx(np.log(2.0) * 11.0 / 11.0, rel=1e-6)
    # Ignored pixels stay out of the weighted mean.
    ignored = torch.tensor([[[t14.IGNORE_INDEX, 1]]])
    assert float(t14.change_loss(logits, ignored, t14.IGNORE_INDEX, 10.0)) == pytest.approx(
        np.log(2.0), rel=1e-6
    )


def test_change_weight_all_ignored_is_graph_connected_zero():
    logits = torch.zeros(1, 2, 1, 2, requires_grad=True)
    target = torch.full((1, 1, 2), t14.IGNORE_INDEX)
    loss = t14.change_loss(logits, target, t14.IGNORE_INDEX, t14.CHANGE_WEIGHT)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert logits.grad is not None


def test_weighted_loss_pushes_change_pixels_up():
    logits = torch.zeros(1, 2, 1, 1, requires_grad=True)
    target = torch.tensor([[[1]]])
    t14.change_loss(logits, target, None, t14.CHANGE_WEIGHT).backward()
    # Gradient on the change logit is larger than the background logit's (weight 1).
    assert float(logits.grad[0, 1]) < float(logits.grad[0, 0])


def test_best_checkpoint_ties_keep_earlier_by_default_and_later_when_asked():
    from rs_backbone_eval.selection import BestCheckpoint

    model = torch.nn.Linear(1, 1)
    default = BestCheckpoint()
    default.offer(0.0, 5, model)
    default.offer(0.0, 10, model)
    assert default.epoch == 5
    later = BestCheckpoint()
    later.offer(0.0, 5, model, ties_go_later=True)
    later.offer(0.0, 10, model, ties_go_later=True)
    assert later.epoch == 10
    later.offer(0.2, 15, model, ties_go_later=True)
    later.offer(0.1, 20, model, ties_go_later=True)
    assert later.epoch == 15


# ---------------------------------------------------------------------------
# Protocol id and run tags: t14_b0_wce replaces t14_b0 (class weight and tie rule changed).

REPO = Path(__file__).resolve().parents[1]


def _manifest_row(split: str, group: str) -> dict:
    """A T14 row that also passes ManifestSample, which the leakage guard requires."""
    return {
        **_row(split),
        "group_id": group,
        "sample_id": f"{group}-{split}",
        "sensor": "planet",
        "bands": ["R", "G", "B", "A"],
        "temporal_mask": [True, True],
        "gsd_m": 4.78,
        "checksum": "a" * 64,
    }


def test_t14_protocol_is_registered_and_the_runner_accepts_only_the_new_id() -> None:
    from lim_worker.runner import TASK_PROTOCOLS

    from rs_backbone_eval.registry import Registry

    assert t14.PROTOCOL_ID == "t14_b0_wce"
    assert TASK_PROTOCOLS["T14"] == {"t14_b0_wce"}
    specs = {p.protocol_id: p for p in Registry(REPO / "configs").protocols()}
    spec = specs["t14_b0_wce"]
    assert spec.tasks == ["T14"]
    assert spec.regime == "frozen_probe" and spec.train_backbone is False
    assert spec.epochs == 20 and spec.lr_schedule == "warmup1-cosine"
    # The superseded run records still name a protocol that the registry resolves.
    assert specs["t14_b0"].tasks == ["T14"]


class _StopAfterTags(Exception):
    """Raised by the writer once the run tags are written, before training starts."""


class _TagWriter:
    def __init__(self) -> None:
        self.tags: dict[str, str] | None = None

    def update(self, tags: dict[str, str] | None = None, **_: object) -> None:
        self.tags = dict(tags or {})
        raise _StopAfterTags


def test_t14_run_tags_record_the_protocol_id_and_class_weight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from rs_backbone_eval.registry import Registry

    monkeypatch.delenv("LIM_WORKER_MICRO_BATCH", raising=False)
    monkeypatch.delenv("LIM_WORKER_GRAD_ACCUM", raising=False)
    # Distinct groups per split so the leakage guard passes. The rows share one
    # label file, which is all the pre-training check reads.
    rows = [
        _manifest_row("train", "aoi-train"),
        _manifest_row("validation", "aoi-val"),
        _manifest_row("test", "aoi-test"),
    ]
    _write(tmp_path / rows[0]["label_path"], np.zeros((SIZE, SIZE), np.uint8), 1, "uint8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    # The pinned v2 hash is a property of the real manifest; this synthetic one is
    # pinned to its own hash so the test reaches the run-tag code it is about.
    import hashlib

    monkeypatch.setattr(
        t14, "T14_MANIFEST_SHA256", hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(t14, "mit_huge", lambda **_: StubEncoder())
    monkeypatch.setattr(
        t14, "load_encoder", lambda path, encoder: {"loaded": 1, "missing": [], "unexpected": []}
    )
    protocol = next(
        p for p in Registry(REPO / "configs").protocols() if p.protocol_id == "t14_b0_wce"
    )
    request = SimpleNamespace(
        run_id="t14-tags", seed=17, data_root=str(tmp_path), manifest_path=str(manifest)
    )
    writer = _TagWriter()
    with pytest.raises(_StopAfterTags):
        t14.train_t14(
            request,
            writer,
            lambda *args, **kwargs: None,
            protocol=protocol,
            task_spec=SimpleNamespace(task_id="T14", ignore_index=255),
            checkpoint=tmp_path / "unused.pt",
        )
    assert writer.tags is not None
    assert writer.tags["protocol_id"] == "t14_b0_wce"
    assert writer.tags["class_weight"] == "change=34, background=1"
    assert t14.CHANGE_WEIGHT == 34.0


def test_t14_refuses_manifest_that_is_not_the_pinned_v2(tmp_path):
    manifest = tmp_path / "832e3540f4f192cf86ec64c5160bd6cbed551a0eaf839f99f9f5fe681bda6759.jsonl"
    manifest.write_text('{"task_id": "T14", "split": "train"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="pinned v2 manifest"):
        t14.require_pinned_manifest(manifest)
