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
from lim_worker.arms import ArmModel  # noqa: E402
from lim_worker.t13 import (  # noqa: E402
    FULL_CHECKPOINT,
    FULL_FORMAT,
    PROTOCOLS,
    FullModel,
    Inputs,
)
from lim_worker.t13_data import (  # noqa: E402
    AEF_DIR,
    IGNORE_INDEX,
    NUM_CLASSES,
    T13Tiles,
    band_statistics,
    donor_rows,
    flip,
    read_aef,
    read_label,
)
from torch import nn  # noqa: E402

from rs_backbone_eval.registry import Registry  # noqa: E402
from rs_backbone_eval.selection import BestCheckpoint  # noqa: E402

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


# ---------------------------------------------------------------------------
# Full fine-tune (whu_ft_b0 / whu_ft_r). A small conv stub stands in for MiT-Huge:
# same four channel widths and strides 4/8/16/32, no checkpoint, no GPU.

REPO = Path(__file__).resolve().parents[1]


class StubPyramid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        chans = t13.CHANNELS
        self.stem = nn.Conv2d(3, chans[0], 4, stride=4)
        self.stages = nn.ModuleList(
            nn.Conv2d(chans[i], chans[i + 1], 2, stride=2) for i in range(3)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        outs = [x]
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs


def _stub(arm: str = "B0") -> tuple[StubPyramid, ArmModel]:
    return StubPyramid(), ArmModel(arm, t13.CHANNELS, 8)


def _inputs(encoder: nn.Module, finetune: bool) -> Inputs:
    return Inputs(encoder, np.zeros(4), np.ones(4), "cpu", finetune=finetune)


def _images() -> torch.Tensor:
    generator = torch.Generator().manual_seed(0)
    return torch.randint(0, 255, (2, 4, 64, 64), generator=generator, dtype=torch.uint8)


def test_finetune_optimizer_has_backbone_group_at_its_own_lr() -> None:
    encoder, model = _stub()
    for p in encoder.parameters():
        p.requires_grad_(True)
    trainable = model.configure_trainable(train_head=True)
    backbone = list(encoder.parameters())
    opt = t13.make_optimizer(trainable, backbone, head_lr=1e-3, backbone_lr=5e-5)
    assert len(opt.param_groups) == 2
    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[1]["lr"] == 5e-5
    assert {id(p) for p in opt.param_groups[0]["params"]} == {id(p) for p in trainable}
    assert {id(p) for p in opt.param_groups[1]["params"]} == {id(p) for p in backbone}
    assert all(p.requires_grad for p in backbone)


def test_frozen_path_keeps_one_group_and_no_encoder_graph() -> None:
    encoder, model = _stub()
    for p in encoder.parameters():
        p.requires_grad_(False)
    trainable = model.configure_trainable(train_head=True)
    opt = t13.make_optimizer(trainable, [], head_lr=1e-3)
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == 1e-3
    assert all(not p.requires_grad for p in encoder.parameters())
    feats = _inputs(encoder, finetune=False).features(_images(), grad=True)
    assert all(not f.requires_grad for f in feats)


def test_finetune_features_keep_the_graph_only_when_asked() -> None:
    encoder, _ = _stub()
    for p in encoder.parameters():
        p.requires_grad_(True)
    inputs = _inputs(encoder, finetune=True)
    assert all(f.requires_grad for f in inputs.features(_images(), grad=True))
    assert all(not f.requires_grad for f in inputs.features(_images(), grad=False))


def _train_one_step(finetune: bool) -> list[str]:
    torch.manual_seed(0)
    encoder, model = _stub()
    for p in encoder.parameters():
        p.requires_grad_(finetune)
    trainable = model.configure_trainable(train_head=True)
    backbone = list(encoder.parameters()) if finetune else []
    opt = t13.make_optimizer(trainable, backbone, head_lr=1e-2, backbone_lr=1e-2)
    before = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    inputs = _inputs(encoder, finetune)
    image = _images()
    out = model(inputs.features(image, grad=finetune), inputs.plugin_image(image), None)
    generator = torch.Generator().manual_seed(1)
    labels = torch.randint(0, NUM_CLASSES, (2, 64, 64), generator=generator)
    loss = t13.safe_cross_entropy(out["logits"].float(), labels, IGNORE_INDEX)
    loss.backward()
    opt.step()
    after = encoder.state_dict()
    return [k for k in before if not torch.equal(before[k], after[k])]


def test_one_step_moves_backbone_only_when_finetuning() -> None:
    assert _train_one_step(finetune=False) == []
    changed = _train_one_step(finetune=True)
    assert "stem.weight" in changed
    assert "stages.2.weight" in changed


def test_full_checkpoint_round_trips_and_has_no_encoder_keys_in_state_dict(
    tmp_path: Path,
) -> None:
    encoder, model = _stub()
    best = BestCheckpoint()
    assert best.offer(0.4, 5, FullModel(encoder, model))
    path = tmp_path / FULL_CHECKPOINT
    t13.save_full_checkpoint(best, path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["format"] == FULL_FORMAT
    assert payload["epoch"] == 5
    assert set(payload["encoder_state_dict"]) == set(encoder.state_dict())
    assert set(payload["state_dict"]) == set(model.state_dict())
    # The frozen loaders read payload["state_dict"] as an ArmModel state, so no encoder keys.
    assert not any(k.startswith("encoder") for k in payload["state_dict"])

    encoder2, model2 = _stub()
    t13.load_full_checkpoint(path, encoder2, model2)
    for name, value in encoder.state_dict().items():
        assert torch.equal(value, encoder2.state_dict()[name]), name
    for name, value in model.state_dict().items():
        assert torch.equal(value, model2.state_dict()[name]), name


def test_full_checkpoint_refuses_a_head_only_state(tmp_path: Path) -> None:
    _, model = _stub()
    best = BestCheckpoint()
    best.offer(0.1, 1, model)
    with pytest.raises(RuntimeError, match="FullModel"):
        t13.save_full_checkpoint(best, tmp_path / FULL_CHECKPOINT)


def test_finetune_protocols_match_their_yaml_and_the_runner_allowlist() -> None:
    from lim_worker.runner import TASK_PROTOCOLS

    assert {"whu_ft_b0", "whu_ft_r"} <= TASK_PROTOCOLS["T13"]
    assert PROTOCOLS["whu_ft_b0"] == {"arm": "B0", "finetune": True}
    assert PROTOCOLS["whu_ft_r"] == {"arm": "R", "aef": "real", "finetune": True}
    specs = {p.protocol_id: p for p in Registry(REPO / "configs").protocols()}
    for pid in ("whu_ft_b0", "whu_ft_r"):
        spec = specs[pid]
        assert spec.regime == "full_finetune"
        assert spec.train_backbone and spec.train_head
        assert spec.learning_rate == 1e-3
        assert spec.backbone_learning_rate == 5e-5
        assert spec.epochs == 20 and spec.lr_schedule == "warmup1-cosine"
        assert spec.tasks == ["T13"]
    assert specs["whu_ft_b0"].train_input_adapter is False
    assert specs["whu_ft_r"].train_input_adapter is True
    for pid in ("whu_b0", "whu_r", "whu_d", "whu_bcap", "whu_p"):
        assert "finetune" not in PROTOCOLS[pid]
        assert specs[pid].regime == "frozen_probe"


def test_finetune_tracks_are_diagnostic_and_separate_from_the_frozen_ones() -> None:
    from rs_backbone_eval.matrix import DIAGNOSTIC_TRACKS, PROTOCOL_TRACKS

    assert PROTOCOL_TRACKS["whu_ft_b0"] == "WF0"
    assert PROTOCOL_TRACKS["whu_ft_r"] == "WFR"
    assert {"WF0", "WFR"} <= DIAGNOSTIC_TRACKS
    assert PROTOCOL_TRACKS["whu_b0"] == "W0"
    assert PROTOCOL_TRACKS["whu_r"] == "WR"


def test_finetune_run_is_never_picked_as_a_frozen_dependency(tmp_path: Path) -> None:
    record = {
        "task_id": "T13",
        "protocol_id": "whu_ft_r",
        "seed": 17,
        "model_id": "lim",
        "status": "completed",
        "model_checkpoint_sha256": BACKBONE_SHA,
        "dataset": {"manifest_sha256": MANIFEST_SHA},
        "training": {"epochs_completed": 20},
        "tags": {"epochs_budget": "20", "effective_batch": "8"},
        "notes": "",
    }
    run_dir = tmp_path / "ft-r"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps(record))
    (run_dir / "best_checkpoint.pt").write_bytes(b"x")
    (run_dir / FULL_CHECKPOINT).write_bytes(b"x")
    for protocol in ("whu_r", "whu_b0"):
        with pytest.raises(RuntimeError, match="found 0"):
            t13.find_dependency(
                tmp_path,
                task_id="T13",
                protocol_id=protocol,
                seed=17,
                model_id="lim",
                manifest_sha=MANIFEST_SHA,
                backbone_sha=BACKBONE_SHA,
                epochs=20,
                batch=8,
            )


# ---------------------------------------------------------------------------
# Full fine-tune smoke: a limit below one accumulation group must still take a step.


def _toy_epoch(limit: int, accumulation: int = 4, batches: int = 8):
    torch.manual_seed(0)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    before = model.weight.detach().clone()
    loader = [{"x": torch.full((1, 2), float(i + 1))} for i in range(batches)]

    def micro_loss(batch: dict):
        loss = model(batch["x"]).pow(2).mean()
        return loss, loss.detach(), loss * 0.0

    totals, ran, steps = t13.run_train_epoch(
        loader, micro_loss, optimizer, None, accumulation, limit
    )
    return model, before, ran, steps


def test_finetune_limit_rounds_up_to_whole_accumulation_groups() -> None:
    assert t13.finetune_limit(0, 4) == 0
    assert [t13.finetune_limit(n, 4) for n in (1, 2, 3, 4, 5)] == [4, 4, 4, 4, 8]


@pytest.mark.parametrize("limit", [1, 2, 3, 4])
def test_finetune_smoke_takes_an_optimizer_step_for_each_small_limit(limit: int) -> None:
    rounded = t13.finetune_limit(limit, 4)
    model, before, ran, steps = _toy_epoch(rounded)
    assert ran == rounded == 4
    assert steps >= 1
    assert not torch.equal(before, model.weight.detach())
    t13.require_finetune_steps(steps, "whu_ft_b0")


def test_unrounded_limit_below_one_group_runs_no_step_and_is_refused() -> None:
    model, before, ran, steps = _toy_epoch(1)
    assert ran == 1 and steps == 0
    assert torch.equal(before, model.weight.detach())
    with pytest.raises(ValueError, match="no optimizer step"):
        t13.require_finetune_steps(steps, "whu_ft_b0")


def test_full_fine_tune_refuses_a_loader_shorter_than_one_group() -> None:
    _, _, _, steps = _toy_epoch(0, accumulation=4, batches=3)
    assert steps == 0
    with pytest.raises(ValueError, match="whu_ft_b0"):
        t13.require_finetune_steps(steps, "whu_ft_b0")


def test_t13_load_rows_skips_other_tasks_in_shared_manifest(tmp_path):
    import json

    from lim_worker import t13_data

    manifest = tmp_path / "shared.jsonl"
    rows = [
        {"task_id": "T13", "split": "train", "sample_id": "keep"},
        {"task_id": "T11", "split": "train", "sample_id": "foreign"},
        {"task_id": "T13", "split": "test", "sample_id": "other_split"},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert [row["sample_id"] for row in t13_data.load_rows(manifest, "train")] == ["keep"]
