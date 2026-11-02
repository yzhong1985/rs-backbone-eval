"""G1 cross-check input validation.

The supplementary review (2026-11-02) reproduced a dump with two patches where
only one had an annotation: the script accepted it and still printed
``map_50=1.0``. Missing labels must be a hard error, and predictions and
annotations must cover the same patch set.

The inference stage needs the MMRotate environment; the validation and scoring
stages are numpy-only, so these tests run in the control plane.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "g1_reference_crosscheck.py"


def load_script():
    spec = importlib.util.spec_from_file_location("g1_reference_crosscheck", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_patch(split: Path, patch: str, lines: list[str]) -> None:
    (split / "annfiles").mkdir(parents=True, exist_ok=True)
    (split / "annfiles" / f"{patch}.txt").write_text("\n".join(lines), encoding="utf-8")


def dump_for(patches: list[str], dump: Path) -> Path:
    payload = [
        {
            "patch": patch,
            "polygons": [[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]],
            "scores": [0.9],
            "labels": [0],
        }
        for patch in patches
    ]
    dump.write_text(json.dumps(payload), encoding="utf-8")
    return dump


def args(split: Path, dump: Path):
    module = load_script()
    return module, module.argparse.Namespace(split=split, dump=dump, limit=0)


def test_a_patch_without_an_annotation_file_fails(tmp_path: Path) -> None:
    split = tmp_path / "split"
    write_patch(split, "P0000__1024__0___0", ["1 1 11 1 11 11 1 11 plane 0"])
    dump = dump_for(["P0000__1024__0___0", "P0000__1024__824___0"], tmp_path / "dump.json")
    module, namespace = args(split, dump)

    with pytest.raises(SystemExit, match="do not cover the same patches"):
        module.run_patch_ap(namespace)

    # an existing but empty annfile is a legitimate "no objects here": once the
    # patch set matches, the stage scores instead of failing
    write_patch(split, "P0000__1024__824___0", [])
    assert module.run_patch_ap(namespace) is None


def test_duplicate_patches_in_the_dump_are_rejected(tmp_path: Path) -> None:
    split = tmp_path / "split"
    write_patch(split, "P0000__1024__0___0", ["1 1 11 1 11 11 1 11 plane 0"])
    dump = dump_for(["P0000__1024__0___0", "P0000__1024__0___0"], tmp_path / "dump.json")
    module, namespace = args(split, dump)

    with pytest.raises(SystemExit, match="duplicate patches"):
        module.run_patch_ap(namespace)


def test_matching_patch_sets_score_and_report_both_metrics(tmp_path: Path, capsys) -> None:
    split = tmp_path / "split"
    write_patch(split, "P0000__1024__0___0", ["1 1 11 1 11 11 1 11 plane 0"])
    dump = dump_for(["P0000__1024__0___0"], tmp_path / "dump.json")
    module, namespace = args(split, dump)

    assert module.run_patch_ap(namespace) is None
    summary = json.loads(capsys.readouterr().out.strip())
    assert summary["patches"] == 1
    # the prediction is the GT square shifted by 1 px: IoU 81/119, one TP
    assert summary["map50_reference_matching_11point"] == 1.0
    assert summary["map50_coco_matching_101point"] == 1.0
    assert summary["gt_per_class"] == {"plane": 1}
    assert len(summary["dump_sha256"]) == 64 and len(summary["annfiles_sha256"]) == 64
