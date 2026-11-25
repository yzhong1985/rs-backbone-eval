"""CPU tests for the Galileo token index check and the per-class Boundary IoU keys.

uv run --project workers/lim --with pytest python -m pytest tests/test_lim_galileo_index.py -q
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
from lim_worker.boundary import REPORTED_CLASSES, boundary_metrics  # noqa: E402
from lim_worker.galileo_index import (  # noqa: E402
    INDEX_FORMAT,
    file_sha256,
    index_digest,
    verify_token_index,
)


def _write_split(root, split: str, samples: list[str]) -> list[dict]:
    (root / split).mkdir(parents=True, exist_ok=True)
    files = {}
    for sample in samples:
        path = root / split / f"{sample}.pt"
        path.write_bytes(f"tokens-{sample}".encode())
        files[sample] = file_sha256(path)
    payload = {
        "format": INDEX_FORMAT,
        "split": split,
        "meta": {"encoder_sha256": "e", "manifest_sha256": "m"},
        "files": files,
    }
    payload["digest"] = index_digest(payload)
    (root / f"index_{split}.json").write_text(json.dumps(payload), encoding="utf-8")
    return [{"sample_id": s} for s in samples]


def test_matching_index_returns_its_digest(tmp_path) -> None:
    rows = _write_split(tmp_path, "test", ["a", "b"])
    digest = verify_token_index(tmp_path, "test", rows)
    assert len(digest) == 64


def test_replaced_token_file_is_refused(tmp_path) -> None:
    rows = _write_split(tmp_path, "test", ["a", "b"])
    (tmp_path / "test" / "b.pt").write_bytes(b"swapped")
    with pytest.raises(RuntimeError, match="differs from its indexed SHA-256"):
        verify_token_index(tmp_path, "test", rows)


def test_edited_index_without_new_digest_is_refused(tmp_path) -> None:
    rows = _write_split(tmp_path, "test", ["a"])
    path = tmp_path / "index_test.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["files"]["a"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest does not match"):
        verify_token_index(tmp_path, "test", rows)


def test_row_missing_from_index_is_refused(tmp_path) -> None:
    _write_split(tmp_path, "test", ["a"])
    with pytest.raises(RuntimeError, match="no entry for sample z"):
        verify_token_index(tmp_path, "test", [{"sample_id": "z"}])


def test_boundary_metrics_names_the_plan_classes_and_skips_empty_unions() -> None:
    total = {1: [3, 4], 4: [0, 0], 6: [1, 2], 2: [5, 5]}
    values = boundary_metrics(total)
    assert values["boundary_iou/farmland"] == pytest.approx(0.75)
    assert values["boundary_iou/road"] == pytest.approx(0.5)
    assert "boundary_iou/water" not in values  # empty union: not reported as 0.0
    assert "boundary_iou/forest" not in values
    assert "boundary_iou" in values
    assert set(REPORTED_CLASSES.values()) == {"farmland", "water", "forest", "road"}
