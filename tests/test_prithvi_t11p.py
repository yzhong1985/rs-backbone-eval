"""T11P six-band Prithvi contract: shapes, token grid and block unfreezing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers" / "prithvi" / "src"))

import prithvi_worker.t11p as t11p  # noqa: E402


class DummyEncoder(nn.Module):
    def __init__(self, depth: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(16, 16) for _ in range(depth)])
        self.norm = nn.LayerNorm(16)
        self.proj = nn.Linear(6, 16)
        self.cls = nn.Parameter(torch.zeros(1, 1, 16))

    def forward(self, pixels):
        batch, _, height, width = pixels.shape
        tokens = self.proj(pixels.mean(dim=(2, 3))).unsqueeze(1)
        grid = height // 16
        tokens = tokens.expand(batch, grid * grid, 16)
        return [torch.cat([self.cls.expand(batch, -1, -1), tokens], dim=1)]


def test_six_band_encode_and_logits_share_the_14x14_grid() -> None:
    encoder = DummyEncoder()
    model = t11p.PrithviT11P(encoder, train_encoder=False, projector=None, dims=16)
    images = torch.zeros(2, t11p.MAX_FRAMES, 6, 128, 128)

    features = model.encode(images)
    logits = model(images)

    assert features.shape == (2, 16, t11p.GRID, t11p.GRID)
    assert logits.shape == (2, t11p.NUM_CLASSES, t11p.LABEL_SIZE, t11p.LABEL_SIZE)
    assert not any(parameter.requires_grad for parameter in encoder.parameters())


def test_projector_widths_for_raw_and_distilled_readouts() -> None:
    raw = t11p.build_head(t11p.EMBED_DIM)
    projected = t11p.build_projector(t11p.B_STAGE_DIMS)
    assert raw[0].in_channels == t11p.EMBED_DIM
    assert projected[-1].out_channels == t11p.B_STAGE_DIMS
    assert projected[-1].in_channels == 256


def test_unfreeze_last_blocks_keeps_the_rest_frozen() -> None:
    encoder = DummyEncoder(depth=8)
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    trainable = t11p._unfreeze_last_blocks(encoder)

    trainable += 0  # keep the type stable for the assertion below
    assert trainable > 0
    for index, block in enumerate(encoder.blocks):
        expected = index >= len(encoder.blocks) - t11p.B_STAGE_BLOCKS
        assert all(parameter.requires_grad is expected for parameter in block.parameters())
    assert all(parameter.requires_grad for parameter in encoder.norm.parameters())
    assert not any(parameter.requires_grad for parameter in encoder.proj.parameters())


def test_teacher_decode_zeros_invalid_pixels() -> None:
    raw = np.array([[[127, -127], [63, 0]]], dtype=np.int8)
    valid = np.array([[True, True], [False, True]])

    decoded = t11p.decode_teacher(raw, valid)

    unit = (127 / 127.5) ** 2
    assert decoded[0, 0, 0] == pytest.approx(unit)
    assert decoded[0, 0, 1] == pytest.approx(-unit)
    assert decoded[0, 1, 0] == 0.0
    assert decoded[0, 1, 1] == 0.0


def test_teacher_tokens_pool_to_the_token_grid(tmp_path: Path) -> None:
    tree = "olmoearth-2019-128"
    path = tmp_path / "derived" / tree / "features" / "2019"
    path.mkdir(parents=True)
    features = np.ones((128, 128, 128), dtype=np.int8) * 127
    valid = np.ones((128, 128), dtype=bool)
    np.savez(path / "10000.npz", features=features, valid=valid)
    row = {"sample_id": "10000"}

    tokens, support = t11p.teacher_tokens("oe128", row, tmp_path, {"oe128": (tree, 128)})

    assert tokens.shape == (128, t11p.GRID, t11p.GRID)
    assert support.shape == (t11p.GRID, t11p.GRID)
    assert bool(support.all())
    assert torch.allclose(tokens, torch.full_like(tokens, (127 / 127.5) ** 2))


def test_feature_cache_round_trips_and_rejects_a_stale_meta(tmp_path: Path) -> None:
    cache = tmp_path / "features" / "abcd1234"
    features = torch.arange(2 * t11p.EMBED_DIM * t11p.GRID * t11p.GRID, dtype=torch.float32)
    features = features.reshape(2, t11p.EMBED_DIM, t11p.GRID, t11p.GRID).to(torch.float16)
    rows = [{"sample_id": "10000"}, {"sample_id": "10001"}]

    t11p._save_feature_cache(cache, [row["sample_id"] for row in rows], features)
    loaded = t11p._load_feature_cache(cache, rows)

    assert loaded is not None and torch.equal(loaded, features)

    meta_path = cache / "10001.json"
    meta = json.loads(meta_path.read_text())
    meta["version"] = "stale"
    meta_path.write_text(json.dumps(meta))

    assert t11p._load_feature_cache(cache, rows) is None


def test_cache_fingerprints_separate_init_normalisation_and_random_seed() -> None:
    base = dict(checkpoint_sha256="abc", manifest_sha256="def", seed=17)
    first = t11p._feature_fingerprint(**base, init="pretrained", normalization="norm-a")
    assert t11p._feature_fingerprint(**base, init="pretrained", normalization="norm-a") == first
    assert t11p._feature_fingerprint(**base, init="random", normalization="norm-a") != first
    assert t11p._feature_fingerprint(**base, init="pretrained", normalization="norm-b") != first
    assert (
        t11p._feature_fingerprint(**{**base, "seed": 42}, init="pretrained", normalization="norm-a")
        == first
    )
    assert t11p._feature_fingerprint(
        **{**base, "seed": 42}, init="random", normalization="norm-a"
    ) != t11p._feature_fingerprint(**base, init="random", normalization="norm-a")


def test_projector_fingerprint_accepts_the_condition_arguments() -> None:
    first = t11p._fingerprint("abc", "def", "oe128", 17, init="pretrained", normalization="norm-a")
    assert (
        t11p._fingerprint("abc", "def", "oe128", 17, init="pretrained", normalization="norm-b")
        != first
    )
    assert (
        t11p._fingerprint("abc", "def", "oe128", 17, init="random", normalization="norm-a") != first
    )


class _OrderDataset(torch.utils.data.Dataset):
    """Three samples whose feature is exactly their index."""

    def __init__(self, root: Path) -> None:
        self.data_root = root
        self.rows = [
            {"sample_id": str(index), "label_path": f"label_{index}.npy"} for index in range(3)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return {
            "image": torch.full((4, 6, 8, 8), float(index)),
            "label": torch.zeros(2, 2, dtype=torch.long),
            "group": str(index),
        }


class _OrderModel:
    def eval(self):
        return self

    def encode(self, images):
        values = images.mean(dim=(1, 2, 3, 4))
        return values[:, None, None, None].expand(-1, 4, 2, 2)


def test_feature_extraction_cold_and_warm_cache_keep_manifest_order(tmp_path: Path) -> None:
    dataset = _OrderDataset(tmp_path)
    for row in dataset.rows:
        np.save(tmp_path / row["label_path"], np.zeros((1, 2, 2), dtype=np.int64))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    cache = tmp_path / "cache"

    cold, _, cold_groups = t11p._extract_split(
        _OrderModel(), loader, torch.zeros(1), torch.ones(1), "cpu", cache_dir=cache
    )
    warm, _, warm_groups = t11p._extract_split(
        _OrderModel(), loader, torch.zeros(1), torch.ones(1), "cpu", cache_dir=cache
    )

    assert cold_groups == ["0", "1", "2"]
    assert warm_groups == cold_groups
    assert torch.equal(cold, warm)
    assert cold[:, 0, 0, 0].tolist() == [0.0, 1.0, 2.0]


def test_normalize_images_applies_the_train_split_statistics() -> None:
    images = torch.tensor([[[[[2.0]], [[4.0]]]]])
    means = torch.tensor([[[[[1.0]], [[2.0]]]]])
    stds = torch.tensor([[[[[1.0]], [[2.0]]]]])

    out = t11p.normalize_images(images, means, stds)

    assert torch.allclose(out, torch.tensor([[[[[1.0]], [[1.0]]]]]))


def test_wide_head_training_normalises_its_micro_batches() -> None:
    import inspect

    source = inspect.getsource(t11p._train_wide_b_stage)

    assert "normalize_images(" in source
