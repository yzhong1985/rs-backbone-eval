"""CPU-only tests for the T11P 10% label-fraction manifest builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_t11p_label_subset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("make_t11p_label_subset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_builder_keeps_the_declared_train_ids_and_full_eval_folds(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "t11p.jsonl"
    ids = tmp_path / "ids.jsonl"
    out = tmp_path / "subset.jsonl"
    write_rows(
        source,
        [
            {"sample_id": "a", "split": "train"},
            {"sample_id": "b", "split": "train"},
            {"sample_id": "c", "split": "train"},
            {"sample_id": "v1", "split": "validation"},
            {"sample_id": "v2", "split": "validation"},
            {"sample_id": "t1", "split": "test"},
        ],
    )
    write_rows(
        ids,
        [
            {"sample_id": "a", "split": "train"},
            {"sample_id": "c", "split": "train"},
            {"sample_id": "v1", "split": "validation"},
        ],
    )

    report = module.build_subset(source, ids, out)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [row["sample_id"] for row in rows] == ["a", "c", "v1", "v2", "t1"]
    assert report["train_ids"] == 2
    assert report["counts"] == {"train": 2, "validation": 2, "test": 1}
    assert report["output_sha256"] == module.sha256(out)


def test_builder_rejects_ids_missing_from_the_source(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "t11p.jsonl"
    ids = tmp_path / "ids.jsonl"
    write_rows(source, [{"sample_id": "a", "split": "train"}])
    write_rows(ids, [{"sample_id": "missing", "split": "train"}])

    with pytest.raises(SystemExit, match="missing"):
        module.build_subset(source, ids, tmp_path / "out.jsonl")
