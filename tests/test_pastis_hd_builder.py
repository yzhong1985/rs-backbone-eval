"""PASTIS-HD manifest builder: T11 (AEF) and T11O (OlmoEarth) share geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from rs_backbone_eval.data.datasets.pastis_hd import (
    ManifestBuildError,
    build_pastis_hd_manifest,
)
from rs_backbone_eval.registry import TaskSpec

ROOT = Path(__file__).resolve().parents[1]
GRID = 8


def _task_spec(name: str) -> TaskSpec:
    payload = yaml.safe_load((ROOT / "configs" / "tasks" / f"{name}.yaml").read_text())
    return TaskSpec.model_validate(payload)


def _patch(
    root: Path,
    tree: str,
    patch: int,
    fold: int,
    *,
    encoding: str,
    dtype: np.dtype,
) -> None:
    features = np.full((64, GRID, GRID), 40, dtype=dtype)
    valid = np.ones((GRID, GRID), dtype=bool)
    features_dir = root / "derived" / tree / "features" / "2019"
    features_dir.mkdir(parents=True, exist_ok=True)
    np.savez(features_dir / f"{patch}.npz", features=features, valid=valid)
    payload = (features_dir / f"{patch}.npz").read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    provenance = (
        {"cog_sha256": {"cog.tiff": digest}}
        if encoding == "raw_aef_int8_as_float16"
        else {"sha256": digest}
    )
    (features_dir / f"{patch}.json").write_text(
        json.dumps(
            {
                "patch": patch,
                "tile": "t30uxv",
                "fold": fold,
                "year": 2019,
                "dims": 64,
                "valid_share": 1.0,
                "quantization": encoding,
                **provenance,
            }
        ),
        encoding="utf-8",
    )
    annotations = root / "derived" / tree / "PASTIS-HD" / "ANNOTATIONS"
    annotations.mkdir(parents=True, exist_ok=True)
    np.save(annotations / f"TARGET_{patch}.npy", np.zeros((1, GRID, GRID), dtype=np.uint8))
    np.save(annotations / f"ParcelIDs_{patch}.npy", np.zeros((GRID, GRID), dtype=np.int32))
    s2 = root / "derived" / tree / "PASTIS-HD" / "DATA_S2"
    s2.mkdir(parents=True, exist_ok=True)
    np.save(s2 / f"S2_{patch}.npy", np.zeros((10, 10, GRID, GRID), dtype=np.uint16))


def _build_both(tmp_path: Path) -> tuple[list, list]:
    for tree, encoding, dtype in (
        ("aef-2019", "raw_aef_int8_as_float16", np.float16),
        ("olmoearth-2019", "olmoearth_int8", np.int8),
    ):
        for patch, fold in ((1, 1), (2, 4), (3, 5)):
            _patch(tmp_path, tree, patch, fold, encoding=encoding, dtype=dtype)
    return (
        build_pastis_hd_manifest(tmp_path, _task_spec("T11_pastis_hd")),
        build_pastis_hd_manifest(tmp_path, _task_spec("T11O_pastis_hd_olmoearth")),
    )


def test_olmoearth_manifest_mirrors_t11_samples(tmp_path: Path) -> None:
    aef, olmoearth = _build_both(tmp_path)

    assert [s.sample_id for s in aef] == [s.sample_id for s in olmoearth] == ["1", "2", "3"]
    assert [s.split for s in olmoearth] == ["train", "validation", "test"]
    assert {s.sensor for s in olmoearth} == {"olmoearth_v1_3_embedding"}
    assert olmoearth[0].bands == [f"OLE_{index:02d}" for index in range(64)]
    assert olmoearth[0].aux_paths["encoding"] == "olmoearth_int8"
    assert olmoearth[0].image_paths == ["derived/olmoearth-2019/features/2019/1.npz"]
    assert aef[0].aux_paths["encoding"] == "raw_aef_int8_as_float16"
    assert aef[0].image_paths == ["derived/aef-2019/features/2019/1.npz"]


def test_olmoearth_manifest_rejects_wrong_encoding(tmp_path: Path) -> None:
    _build_both(tmp_path)
    sidecar = tmp_path / "derived" / "olmoearth-2019" / "features" / "2019" / "1.json"
    payload = json.loads(sidecar.read_text())
    payload["quantization"] = "raw_aef_int8_as_float16"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ManifestBuildError, match="olmoearth_int8"):
        build_pastis_hd_manifest(tmp_path, _task_spec("T11O_pastis_hd_olmoearth"))
