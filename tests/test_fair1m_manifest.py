"""FAIR1M-2.0 manifest builder: tiling rule, visibility rule, annotation join.

The builder is the only place that decides which objects a tile may train on, so
the tests pin the three numbers a later metric depends on: the clamped tile
origins, the 0.7 visibility threshold, and the per-tile kept/ignored counts that
make the split auditable without opening annotations.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from rs_backbone_eval.data.datasets.fair1m import (
    BuildStats,
    build_fair1m_manifest,
    parse_fair1m_xml,
)
from rs_backbone_eval.data.datasets.sen1floods11 import ManifestBuildError
from rs_backbone_eval.data.manifest import audit_task_manifest, write_manifest
from rs_backbone_eval.data.tiling import tile_origins_clamped
from rs_backbone_eval.registry import TaskSpec

TASK = {
    "task_id": "T10",
    "application": "detection",
    "dataset_id": "FAIR1M",
    "family": "detection",
    "sensor": "aerial-rgb",
    "time_steps": 1,
    "bands": ["RED", "GREEN", "BLUE"],
    "num_outputs": 37,
    "primary_metric": {"name": "map", "direction": "maximize"},
    "official_config_url": "https://example.invalid/t10",
}


def fake_png(path: Path, width: int, height: int) -> None:
    """A 24-byte PNG header is all the control plane reads."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)
    )


def annotation_xml(objects: list[tuple[str, list[tuple[float, float]]]]) -> str:
    body = []
    for name, points in objects:
        closed = [*points, points[0]]
        text = "".join(f"<point>{x},{y}</point>" for x, y in closed)
        body.append(
            f"<object><possibleresult><name>{name}</name></possibleresult>"
            f"<points>{text}</points></object>"
        )
    return f"<annotation><objects>{''.join(body)}</objects></annotation>"


def make_tree(root: Path, width: int = 2000, height: int = 1200, objects=None) -> Path:
    objects = objects or [
        ("Small Car", [(100, 100), (200, 100), (200, 150), (100, 150)]),
        # hangs 100 px past the right edge of the image: 50% visible at best
        ("Bus", [(1900, 300), (2100, 300), (2100, 400), (1900, 400)]),
    ]
    fake_png(root / "raw" / "fair1m" / "images" / "train" / "1000.png", width, height)
    xml = root / "raw" / "fair1m" / "labelXml"
    xml.mkdir(parents=True, exist_ok=True)
    (xml / "1000.xml").write_text(annotation_xml(objects), encoding="utf-8")
    return root


def build(root: Path, monkeypatch: pytest.MonkeyPatch, **env: str):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    stats = BuildStats()
    samples = build_fair1m_manifest(root, TaskSpec.model_validate(TASK), stats)
    return samples, stats


def test_clamped_origins_reach_the_edge_without_padding() -> None:
    # 2000 px, 1024 crop, 824 stride: 0, 824, then flush at 976
    assert tile_origins_clamped(2000, 1024, 824) == [0, 824, 976]
    # an exact multiple needs no extra tile
    assert tile_origins_clamped(2048, 1024, 1024) == [0, 1024]
    # a small image is one whole tile (the worker pads it)
    assert tile_origins_clamped(600, 1024, 824) == [0]
    # the padded rule the iSAID track uses is unchanged
    from rs_backbone_eval.data.tiling import tile_origins

    assert tile_origins(2000, 1024, 824) == [0, 824, 1648]


def test_every_tile_is_fully_inside_the_image(tmp_path: Path, monkeypatch) -> None:
    samples, _ = build(make_tree(tmp_path, 2000, 1200), monkeypatch)
    for sample in samples:
        x, y, w, h = sample.window
        assert x + w <= 2000 and y + h <= 1200
        assert sample.valid_fraction == 1.0


def test_counts_follow_the_visibility_threshold(tmp_path: Path, monkeypatch) -> None:
    samples, stats = build(make_tree(tmp_path), monkeypatch)
    by_origin = {(s.window[0], s.window[1]): s for s in samples}
    assert len(samples) == 6

    # the fully visible car is trained once, in the top-left tile
    assert by_origin[(0, 0)].instance_count == 1
    assert by_origin[(0, 0)].ignored_instance_count == 0

    # the bus hangs past the image edge: 100/200 px visible = 0.5 < 0.7, so it is
    # ignored in both tiles that touch it and never trained as background
    assert by_origin[(976, 0)].instance_count == 0
    assert by_origin[(976, 0)].ignored_instance_count == 1
    assert by_origin[(976, 176)].ignored_instance_count == 1

    assert stats.instances == 2
    assert stats.kept_assignments == 1
    assert stats.ignored_assignments == 2
    assert stats.empty_tiles == 5


def test_annotation_file_carries_the_locked_class_ids(tmp_path: Path, monkeypatch) -> None:
    samples, _ = build(make_tree(tmp_path), monkeypatch)
    payload = json.loads((tmp_path / samples[0].annotation_path).read_text(encoding="utf-8"))
    assert len(payload["classes"]) == 37
    assert payload["width"] == 2000 and payload["height"] == 1200
    names = [instance["class_name"] for instance in payload["instances"]]
    assert names == ["Small Car", "Bus"]
    # aircraft 11 + ships 9 = 20, so Small Car is 20 and Bus is 21
    assert [instance["class_id"] for instance in payload["instances"]] == [20, 21]
    # every tile of one image shares the annotation checksum
    assert len({sample.checksum for sample in samples}) == 1


def test_manifest_rows_audit_against_the_task_spec(tmp_path: Path, monkeypatch) -> None:
    samples, _ = build(make_tree(tmp_path), monkeypatch)
    path = tmp_path / "manifests" / "T10.jsonl"
    write_manifest(path, samples)
    audit = audit_task_manifest(path, TaskSpec.model_validate(TASK))
    assert audit.sample_count == len(samples)
    assert audit.split_counts["train"] == len(samples)
    assert audit.all_errors() == []
    # no raster mask on a detection row, and no annotation on a mask row
    assert all(sample.label_path is None and sample.annotation_path for sample in samples)


def test_val_split_becomes_test(tmp_path: Path, monkeypatch) -> None:
    root = make_tree(tmp_path)
    val_images = root / "raw" / "fair1m" / "images" / "val"
    fake_png(val_images / "2000.png", 1200, 1200)
    (root / "raw" / "fair1m" / "labelXml" / "2000.xml").write_text(
        annotation_xml([("Bridge", [(10, 10), (60, 10), (60, 20), (10, 20)])]), encoding="utf-8"
    )
    samples, _ = build(root, monkeypatch)
    assert {sample.split for sample in samples} == {"train", "test"}
    assert {sample.group_id for sample in samples if sample.split == "test"} == {"2000"}


def test_missing_annotation_is_an_error_not_a_skipped_image(tmp_path: Path, monkeypatch) -> None:
    root = make_tree(tmp_path)
    (root / "raw" / "fair1m" / "labelXml" / "1000.xml").unlink()
    with pytest.raises(ManifestBuildError, match="missing annotation"):
        build(root, monkeypatch)


def test_parser_drops_the_closing_duplicate_and_counts_malformed(tmp_path: Path) -> None:
    path = tmp_path / "a.xml"
    path.write_text(
        annotation_xml([("Small Car", [(0, 0), (10, 0), (10, 5), (0, 5)])]).replace(
            "</points>", "<point>1,1</point><point>2,2</point></points>"
        ),
        encoding="utf-8",
    )
    instances, malformed = parse_fair1m_xml(path)
    assert instances == []
    assert malformed == 1

    good = tmp_path / "b.xml"
    good.write_text(annotation_xml([("Small Car", [(0, 0), (10, 0), (10, 5), (0, 5)])]), "utf-8")
    instances, malformed = parse_fair1m_xml(good)
    assert malformed == 0
    assert instances[0].points == ((0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0))


def test_inferred_stride_ignores_the_clamped_edge_gap() -> None:
    """The modal gap is the declared stride; the flush tile creates a small one."""
    from rs_backbone_eval.data.tiling import infer_stride

    origins = {"a": {0, 256, 488}}  # 1000 px image at 512/256, last tile clamped
    assert infer_stride(origins) == 256
    assert infer_stride({"a": {0, 256, 488}, "b": {0, 256, 488}, "c": {0}}) == 256
    assert infer_stride({"a": {0}}) is None


def test_valid_rect_tracks_padding_and_flips() -> None:
    from rs_backbone_eval.data.tiling import valid_rect

    # a 600x800 image padded into 1024 keeps the real block top-left
    assert valid_rect(1024, 800, 600) == (0, 0, 800, 600)
    # after a horizontal flip the block moves to the right edge, after both to
    # the far corner; the detector needs this to know which anchors are real
    assert valid_rect(1024, 800, 600, flip_x=True) == (224, 0, 800, 600)
    assert valid_rect(1024, 800, 600, flip_y=True) == (0, 424, 800, 600)
    assert valid_rect(1024, 800, 600, flip_x=True, flip_y=True) == (224, 424, 800, 600)
    # an image larger than the crop is fully real
    assert valid_rect(1024, 5000, 5000) == (0, 0, 1024, 1024)
    with pytest.raises(ValueError):
        valid_rect(1024, 0, 100)


def test_a_stale_annotation_size_is_rewritten_not_trusted(tmp_path: Path, monkeypatch) -> None:
    """Cached annotation sizes can outlive the image behind an id.

    Regression 2026-10-29: the val images were replaced after their annotations
    were derived, so 1,274 files still described the old 1500x1500 pixels. The
    worker cross-checks the size against valid_fraction and would have refused
    every such tile, so the builder now repairs the file instead of trusting it.
    """
    root = make_tree(tmp_path, 1200, 900)
    build(root, monkeypatch)
    annotation = next((tmp_path / "derived" / "fair1m" / "obb").glob("*.json"))
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    payload["width"], payload["height"] = 5000, 5000
    annotation.write_text(json.dumps(payload), encoding="utf-8")

    _, stats = build(root, monkeypatch)
    repaired = json.loads(annotation.read_text(encoding="utf-8"))
    assert (repaired["width"], repaired["height"]) == (1200, 900)
    assert stats.stale_annotations == 1
