import json
import re
from pathlib import Path

import pytest

from rs_backbone_eval.visualization import (
    MAX_SAMPLES,
    candidate_window_per_scene,
    compute_fingerprint,
    detection_bucket,
    order_candidates,
    replay_mode,
    segmentation_bucket,
    select_bucket_samples,
    take_pool,
    validate_manifest,
    window_digest,
)

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "docs/design/Portal可视化_2026-11-02.md"


def documented_manifest() -> dict:
    """The design's sample must stay valid: it is the schema's own test vector."""
    text = DESIGN.read_text(encoding="utf-8")
    section = text.split("## 8. manifest 契约（完整样例）", 1)[1]
    match = re.search(r"```json\n(.*?)\n```", section, re.S)
    assert match, "design §8 has no json block"
    return json.loads(match.group(1))


def row(scene: str, x: int, y: int, size: int = 512) -> dict:
    return {"group_id": scene, "window": [x, y, size, size], "split": "test"}


def test_documented_manifest_validates() -> None:
    payload = documented_manifest()
    validate_manifest(payload)
    assert payload["unit"] == "scene-window"
    assert payload["selection"]["bucket_metric"] == "foreground_binary_iou"


def test_digest_is_stable_and_seeded() -> None:
    window = (0, 512, 512, 512)
    assert window_digest(17, "P0443", window) == window_digest(17, "P0443", window)
    assert window_digest(17, "P0443", window) != window_digest(42, "P0443", window)
    assert window_digest(17, "P0443", window) != window_digest(17, "P0444", window)
    # Sixteen hex characters, i.e. a real hash rather than the memory address of
    # Python's randomised builtin hash().
    assert re.fullmatch(r"[0-9a-f]{64}", window_digest(17, "P0443", window))


def test_pool_order_is_hash_order_not_input_order() -> None:
    rows = [row("A", 0, 0), row("B", 0, 0), row("C", 0, 0)]
    ordered = [sample for _, _, sample in order_candidates(rows, seed=17)]
    assert sorted(ordered) == ["A@0_0_512_512", "B@0_0_512_512", "C@0_0_512_512"]
    assert ordered != ["A@0_0_512_512", "B@0_0_512_512", "C@0_0_512_512"]

    pool, digest = take_pool(rows, seed=17, pool=2)
    assert len(pool) == 2
    assert digest == take_pool(rows, seed=17, pool=50)[1]
    assert digest != take_pool(rows, seed=42, pool=2)[1]


def test_detection_buckets() -> None:
    assert detection_bucket(tp=3, fp=0, fn=0, gt=3, predictions=3) == "hit"
    assert detection_bucket(tp=3, fp=1, fn=0, gt=3, predictions=4) == "partial"
    assert detection_bucket(tp=0, fp=2, fn=3, gt=3, predictions=2) == "miss"
    assert detection_bucket(tp=0, fp=0, fn=0, gt=0, predictions=0) == "empty_correct"
    assert detection_bucket(tp=0, fp=1, fn=0, gt=0, predictions=1) == "empty_false_alarm"


def test_segmentation_buckets() -> None:
    assert (
        segmentation_bucket(iou=0.9, gt_foreground=10, predicted_foreground=10, has_background=True)
        == "hit"
    )
    assert (
        segmentation_bucket(iou=0.4, gt_foreground=10, predicted_foreground=10, has_background=True)
        == "partial"
    )
    assert (
        segmentation_bucket(iou=0.1, gt_foreground=10, predicted_foreground=10, has_background=True)
        == "miss"
    )
    assert (
        segmentation_bucket(iou=None, gt_foreground=0, predicted_foreground=0, has_background=False)
        == "ignored"
    )
    assert (
        segmentation_bucket(iou=1.0, gt_foreground=0, predicted_foreground=0, has_background=True)
        == "empty_correct"
    )
    assert (
        segmentation_bucket(iou=0.0, gt_foreground=0, predicted_foreground=5, has_background=True)
        == "empty_false_alarm"
    )
    # T5/T8 have no background class: an all-background window is not a bucket.
    assert (
        segmentation_bucket(iou=0.0, gt_foreground=0, predicted_foreground=0, has_background=False)
        == "miss"
    )


def test_selection_is_capped_and_never_truncates() -> None:
    scored = [
        {"digest": f"{index:064x}", "bucket": bucket}
        for index, bucket in enumerate(["hit", "hit", "partial", "partial", "miss", "miss"])
    ]
    chosen = select_bucket_samples(scored, per_bucket=2)
    assert len(chosen) == 6
    with pytest.raises(ValueError, match="hard cap"):
        select_bucket_samples(scored, per_bucket=5)
    with pytest.raises(ValueError):
        select_bucket_samples(scored, per_bucket=0)


def test_manifest_rejects_a_bucket_that_contradicts_its_counts() -> None:
    payload = documented_manifest()
    payload["samples"][0]["bucket"] = "partial"
    payload["samples"][0]["pixels"] = {"tp": 18234, "fp": 1, "fn": 0}
    payload["selection"]["buckets"] = ["hit", "partial", "miss"]
    with pytest.raises(ValueError, match="contradicts|partial"):
        validate_manifest(payload)


def test_manifest_rejects_unsafe_image_paths() -> None:
    payload = documented_manifest()
    payload["samples"][0]["image"] = "../run.json"
    with pytest.raises(ValueError, match="samples/"):
        validate_manifest(payload)


def test_manifest_rejects_more_samples_than_the_cap() -> None:
    payload = documented_manifest()
    payload["selection"]["per_bucket"] = 9
    with pytest.raises(ValueError, match="cap"):
        validate_manifest(payload)
    assert MAX_SAMPLES == 12


def test_fingerprint_and_replay_mode() -> None:
    first = compute_fingerprint({"a.py": "1"}, {"crop": 1024}, {"torch": "2.9"})
    assert first == compute_fingerprint({"a.py": "1"}, {"crop": 1024}, {"torch": "2.9"})
    assert first != compute_fingerprint({"a.py": "2"}, {"crop": 1024}, {"torch": "2.9"})
    assert first != compute_fingerprint({"a.py": "1"}, {"crop": 512}, {"torch": "2.9"})

    mode, reason = replay_mode(
        fingerprint_at_run=first, fingerprint_now=first, worktree_clean=True, stale_flags=[]
    )
    assert (mode, reason) == ("original_protocol", None)

    assert (
        replay_mode(
            fingerprint_at_run=first, fingerprint_now=first, worktree_clean=False, stale_flags=[]
        )[0]
        == "reevaluated"
    )
    assert (
        replay_mode(
            fingerprint_at_run=first,
            fingerprint_now=first,
            worktree_clean=True,
            stale_flags=["T9_PADDING"],
        )[0]
        == "reevaluated"
    )
    assert (
        replay_mode(
            fingerprint_at_run=None, fingerprint_now=first, worktree_clean=True, stale_flags=[]
        )[0]
        == "reevaluated"
    )
    assert (
        replay_mode(
            fingerprint_at_run=first, fingerprint_now="other", worktree_clean=True, stale_flags=[]
        )[0]
        == "reevaluated"
    )


def test_every_sample_is_validated_even_after_an_earlier_error() -> None:
    """A shared flag used to short-circuit the loop after the first bad sample."""
    payload = documented_manifest()
    sample = payload["samples"][0]
    payload["samples"] = [
        {k: v for k, v in sample.items() if k != "window"},
        {k: v for k, v in sample.items() if k != "metrics"},
    ]
    payload["selection"]["buckets"] = ["hit"]
    with pytest.raises(ValueError) as caught:
        validate_manifest(payload)
    message = str(caught.value)
    assert "samples[0] missing window" in message
    assert "samples[1] missing metrics" in message


def test_candidate_window_per_scene_is_hash_ordered() -> None:
    rows = [row("A", 0, 0), row("A", 512, 0), row("B", 0, 0)]
    chosen = candidate_window_per_scene(rows, seed=17)
    assert set(chosen) == {"A", "B"}
    expected = min(
        (row("A", 0, 0), row("A", 512, 0)),
        key=lambda item: window_digest(17, "A", (item["window"][0], 0, 512, 512)),
    )
    assert chosen["A"]["window"] == expected["window"]
    # Determinism: the same seed must not depend on input order.
    assert (
        candidate_window_per_scene(list(reversed(rows)), seed=17)["A"]["window"]
        == chosen["A"]["window"]
    )
