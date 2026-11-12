"""Worker-level tests for the RGB+NIR track (Clay environment, CPU only).

The Clay encoder is stubbed at stride 8, so the token cache, the projector and
the probe head are exercised in seconds without a GPU or the real dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
import clay_worker.rgbn as rgbn  # noqa: E402
import torch  # noqa: E402


class StubEncoder(torch.nn.Module):
    """(B,4,128,128) -> (B,256,1024) tokens; counts forward calls."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, datacube):
        self.calls += 1
        pixels = datacube["pixels"]
        grid = pixels.shape[-1] // rgbn.PATCH
        return torch.ones(pixels.shape[0], grid * grid, 1024, device=pixels.device)


def _patch(tmp_path: Path) -> dict:
    s2 = tmp_path / "S2_10000.npy"
    np.save(s2, np.zeros((25, 10, 128, 128), np.int16))
    dates = [20180924] + [20190100 + index for index in range(1, 25)]
    sidecar = tmp_path / "10000.json"
    sidecar.write_text(json.dumps({"dates_s2": dates}), encoding="utf-8")
    return {
        "sample_id": "10000",
        "split": "train",
        "image_paths": ["S2_10000.npy"],
        "label_path": "TARGET_10000.npy",
        "aux_paths": {"s2_path": "S2_10000.npy", "dates_source": "10000.json"},
    }


def test_frame_tokens_shape_and_frozen_graph(tmp_path: Path) -> None:
    encoder = StubEncoder()
    pixels = torch.zeros(2, 4, 128, 128)
    waves = torch.tensor([0.493, 0.56, 0.665, 0.842])

    tokens = rgbn.frame_tokens(encoder, pixels, waves)

    assert tokens.shape == (2, 1024, 16, 16)
    assert encoder.calls == 1


def test_token_forward_chunks_frames_without_changing_the_result() -> None:
    encoder = StubEncoder()
    pixels = torch.zeros(12, 4, 128, 128)

    tokens = rgbn._token_forward(encoder, pixels, torch.zeros(4), grad=False, chunk=5)

    assert tokens.shape == (12, 1024, 16, 16)
    assert encoder.calls == 3, "12 frames in chunks of 5 must be three forwards"


def test_feature_cache_writes_once_and_reuses_by_fingerprint(tmp_path: Path) -> None:
    encoder = StubEncoder()
    row = _patch(tmp_path)
    cache = rgbn.FeatureCache(
        tmp_path / "cache",
        "fingerprint-a",
        encoder,
        torch.tensor([0.49, 0.56, 0.66, 0.84]),
        torch.zeros(4),
        torch.ones(4),
        device="cpu",
    )

    first = np.asarray(cache.get(row, tmp_path))
    second = np.asarray(cache.get(row, tmp_path))

    assert encoder.calls == 1, "the second read must come from the cache"
    assert first.shape == (20, 1024, 16, 16)
    assert np.array_equal(first, second)
    assert cache.hits == 1 and cache.misses == 1

    # A different fingerprint is a different directory and must recompute.
    other = rgbn.FeatureCache(
        tmp_path / "cache",
        "fingerprint-b",
        encoder,
        torch.tensor([0.49, 0.56, 0.66, 0.84]),
        torch.zeros(4),
        torch.ones(4),
        device="cpu",
    )
    other.get(row, tmp_path)
    assert encoder.calls == 2


def test_frozen_encoder_receives_no_gradients() -> None:
    encoder = StubEncoder()
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    tokens = rgbn.frame_tokens(encoder, torch.zeros(1, 4, 128, 128), torch.zeros(4))
    tokens.detach().requires_grad_(True).sum().backward()

    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_teacher_pairing_reads_the_manifest_sample(tmp_path: Path) -> None:
    row = _patch(tmp_path)
    features = np.zeros((64, 128, 128), np.int8)
    features[:, 0:8, 0:8] = 127
    teacher_dir = tmp_path / "derived" / "aef-2019" / "features" / "2019"
    teacher_dir.mkdir(parents=True)
    np.savez(teacher_dir / "10000.npz", features=features, valid=np.ones((128, 128), bool))

    grid, support = rgbn.teacher_tokens("aef64", row, tmp_path)

    assert grid.shape == (64, 16, 16)
    assert support.all()
    assert grid[:, 0, 0].min() > 0.9  # the top-left 8x8 block is fully observed

    with pytest.raises(rgbn.RgbnFailure):
        rgbn.teacher_tokens("aef64", {**row, "sample_id": "99999"}, tmp_path)


class StubTransformer(torch.nn.Module):
    def __init__(self, depth: int = 6) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(depth)])
        self.norm = torch.nn.LayerNorm(4)


class StubBackbone(torch.nn.Module):
    def __init__(self, depth: int = 6) -> None:
        super().__init__()
        self.transformer = StubTransformer(depth)


def test_unfreeze_last_blocks_keeps_the_rest_frozen() -> None:
    encoder = StubBackbone(depth=6)
    for parameter in encoder.parameters():
        parameter.requires_grad = False

    names = rgbn._unfreeze_last_blocks(encoder, blocks=4)

    assert names[-1] == "transformer.norm"
    assert len(names) == 5
    for index, block in enumerate(encoder.transformer.layers):
        expected = index >= 2
        assert all(parameter.requires_grad is expected for parameter in block.parameters())
    assert all(parameter.requires_grad for parameter in encoder.transformer.norm.parameters())


def test_projector_and_head_output_widths() -> None:
    projector = rgbn.build_projector(1024, 64)
    head = rgbn.build_head(64, 20)

    tokens = torch.zeros(2, 1024, 16, 16)
    logits = head(projector(tokens))

    assert projector(tokens).shape == (2, 64, 16, 16)
    assert logits.shape == (2, 20, 16, 16)
    assert all(parameter.requires_grad for parameter in projector.parameters())


def test_cosine_alignment_is_one_for_parallel_vectors() -> None:
    prediction = torch.randn(2, 64, 16, 16)
    mask = torch.ones(2, 16, 16, dtype=torch.bool)

    assert float(rgbn._cosine(prediction, prediction, mask)) == pytest.approx(1.0, abs=1e-5)
    flipped = -prediction
    assert float(rgbn._cosine(prediction, flipped, mask)) == pytest.approx(-1.0, abs=1e-5)


def test_t8_transfer_head_is_self_contained(tmp_path: Path) -> None:
    """The head payload names the frozen readout it was trained on."""
    import clay_worker.transfer as transfer

    class Writer:
        def __init__(self) -> None:
            self.run_dir = tmp_path
            self.record = type("Record", (), {"model_checkpoint_sha256": "c0ffee"})()
            self.artifacts: list[tuple[str, str]] = []

        def add_artifact(self, kind: str, path: str, title: str | None = None) -> None:
            self.artifacts.append((kind, path))

    projector = tmp_path / "projector-source.pt"
    projector.write_bytes(b"frozen projector")
    writer = Writer()
    head = transfer.build_head(64)
    path = transfer._save_head(
        writer,
        head,
        dims=64,
        protocol="clay_rgbn_t8_a64",
        seed=17,
        epochs=8,
        means=np.arange(4, dtype=np.float32),
        stds=np.arange(4, dtype=np.float32) + 1,
        target="aef64",
        projector_path=projector,
        checkpoint=Path("/fake/clay.ckpt"),
    )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert set(payload["state_dict"]) == set(head.state_dict())
    assert payload["protocol"] == "clay_rgbn_t8_a64"
    assert payload["seed"] == 17
    assert payload["in_channels"] == 64
    assert payload["num_classes"] == transfer.NUM_CLASSES
    assert payload["bands"] == transfer.BANDS
    assert payload["normalization"]["mean"] == [0.0, 1.0, 2.0, 3.0]
    assert payload["projector"]["source"] == str(projector)
    assert payload["projector"]["sha256"] == transfer._sha256(projector)
    assert (tmp_path / "projector.pt").read_bytes() == b"frozen projector"
    assert payload["reduced_budget"] is True
    assert {kind for kind, _ in writer.artifacts} == {"checkpoint"}
    assert {path for _, path in writer.artifacts} == {"head.pt", "projector.pt"}


def test_t8_projector_selection_is_pinned_to_its_source_task(tmp_path: Path) -> None:
    """T11D and T11P projectors coexist; the T8 loader must pick T11D."""
    import clay_worker.transfer as transfer

    root = tmp_path / "cache"
    runs = tmp_path / "runs"
    seed_dir = root / "projector" / "oe128" / "seed17"
    seed_dir.mkdir(parents=True)
    for task, source in (("T11D", "clay-T11D-distill"), ("T11P", "clay-T11P-distill")):
        torch.save(
            {"target": "oe128", "source_run_id": source},
            seed_dir / f"{task.lower()}-fingerprint.pt",
        )
        (runs / source).mkdir(parents=True)
        (runs / source / "run.json").write_text(json.dumps({"task_id": task}), encoding="utf-8")

    path, payload = transfer.find_projector(
        "oe128", 17, source_task="T11D", runs_root=runs, root=root
    )
    assert path.name == "t11d-fingerprint.pt"
    assert payload["source_run_id"] == "clay-T11D-distill"
