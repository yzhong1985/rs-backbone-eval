"""Pure helpers for the Colab A100 dispatch of the LIM T11 BB track.

Kept free of ``google.colab`` / network imports so the local test suite can
exercise the same logic the notebook runs (review 2026-11-04 round 3: the
notebook's verification/copy-back path must be reviewable and dry-runnable
without a GPU).
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

REQUIRED_ENV = (
    "RS_EVAL_DATA_ROOT",
    "RS_EVAL_WEIGHTS_ROOT",
    "RS_EVAL_RUNS_ROOT",
    "RS_EVAL_CACHE_ROOT",
)


def run_id_for(proto: str, seed: int, suffix: str = "a100") -> str:
    """Ledger-consistent run id (same shape as the T11BB lane ids).

    The notebook used to generate ``lim_s2_single-train-17-a100`` (missing the
    model/task prefix), which would not match the ledger naming (review
    round 3 #1d).
    """
    return f"lim_mit_huge-T11-{proto}-train-{seed}-{suffix}"


def dispatch_env(
    base: dict[str, str],
    *,
    repo: Path,
    pack: Path,
    data_root: Path,
    runs_root: Path,
    cache_root: Path,
    manifest: Path,
    checkpoint: Path,
    weights_root: Path | None = None,
) -> dict[str, str]:
    """Environment for ``rs-eval dispatch`` inside the notebook.

    The bundle checkout has no ``.env``, so every root must be pinned here or
    ``RuntimePaths`` falls back to ``REPO/data`` / ``REPO/runs`` and the
    manifest-relative paths stop resolving (review round 3 #1b).
    """
    env = dict(base)
    env.update(
        {
            "RS_EVAL_DATA_ROOT": data_root.as_posix(),
            "RS_EVAL_WEIGHTS_ROOT": (weights_root or checkpoint.parent).as_posix(),
            "RS_EVAL_RUNS_ROOT": runs_root.as_posix(),
            "RS_EVAL_CACHE_ROOT": cache_root.as_posix(),
            "RS_EVAL_CONFIG_ROOT": (repo / "configs").as_posix(),
            "RS_EVAL_MANIFEST": manifest.as_posix(),
            "LIM_CHECKPOINT": checkpoint.as_posix(),
            # a GPU-declared cell must never silently fall back to CPU
            # (review round 4): the worker turns this into a hard error
            "RS_EVAL_REQUIRE_DEVICE": "cuda",
            # the notebook sets HF_ENDPOINT; the workers copy it into run tags
            "RS_EVAL_HF_ENDPOINT": env.get("HF_ENDPOINT", ""),
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "8"),
        }
    )
    missing = [name for name in REQUIRED_ENV if not env.get(name)]
    if missing:
        raise ValueError(f"dispatch environment misses {missing}")
    return env


def assert_a100_40gb(gpu_name: str) -> None:
    normalised = gpu_name.lower()
    if "a100" not in normalised or "40gb" not in normalised:
        raise SystemExit(f"expected an A100 40GB runtime, got {gpu_name!r}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def needs_download(pastis_hd: Path, s2_hashes: dict[str, str], ann_hashes: dict[str, str]) -> bool:
    """Whether the Drive-side HF mirror is incomplete.

    The old gate counted only ``DATA_S2`` files. A Drive tree warmed by an
    earlier session (S2 complete, ANNOTATIONS never fetched) skipped the
    download and then failed verification with a misleading "missing" message
    (review 2026-11-04 round 7).
    """
    if not s2_hashes or not ann_hashes:
        raise SystemExit("empty sha256 list; refusing to judge completeness")

    def _present(directory: str) -> set[str]:
        root = pastis_hd / directory
        return {path.name for path in root.glob("*.npy")} if root.is_dir() else set()

    # Compare the *expected names*, not file counts: extra files used to hide a
    # missing one (review 2026-11-04 round 9).
    return not set(s2_hashes) <= _present("DATA_S2") or not set(ann_hashes) <= _present(
        "ANNOTATIONS"
    )


def assert_checkpoint_file(checkpoint: Path, expected_sha256: str) -> None:
    """Refuse to train from a checkpoint that is not the one the pack recorded."""
    if not expected_sha256:
        raise SystemExit("pack PROVENANCE has no checkpoint sha256; refusing to train")
    actual = file_sha256(checkpoint)
    if actual != expected_sha256:
        raise SystemExit(
            f"checkpoint sha256 {actual[:16]} != packed {expected_sha256[:16]} "
            f"({checkpoint}); upload the exact weights the pack was built for"
        )


def assert_record_checkpoint(run_dir: Path, expected_sha256: str) -> None:
    """Check the worker's own checkpoint hash before the run leaves the session."""
    import json

    record = run_dir / "run.json"
    try:
        payload = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {record}: {exc}") from exc
    actual = payload.get("model_checkpoint_sha256")
    if actual != expected_sha256:
        raise SystemExit(
            f"{run_dir.name}: run.json checkpoint sha256 {actual!r} != packed "
            f"{expected_sha256[:16]} (do not mix this run with the formal cell)"
        )


def verify_tree(tree_dir: Path, hashes: dict[str, str], label: str) -> None:
    """Verify every packed sha against ``tree_dir/<name>``."""
    if not hashes:
        raise SystemExit(f"empty {label} sha256 list; refusing to verify nothing")
    missing = [name for name in sorted(hashes) if not (tree_dir / name).is_file()]
    if missing:
        raise SystemExit(f"{label} incomplete: missing {missing[:5]} ({len(missing)})")
    bad = [name for name in sorted(hashes) if file_sha256(tree_dir / name) != hashes[name]]
    if bad:
        raise SystemExit(f"{label} corrupt: {bad[:5]} ({len(bad)})")


def verification_fresh(marker: Path, hash_list: Path) -> bool:
    """Whether a previous session already verified this exact hash list.

    Route C downloads 45 GB into Drive once; re-hashing it every session is
    slow. The marker stores the hash-list digest, so editing/replacing the
    list invalidates the cache.
    """
    return marker.is_file() and marker.read_text(encoding="utf-8").strip() == file_sha256(hash_list)


def mark_verified(marker: Path, hash_list: Path) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(file_sha256(hash_list) + "\n", encoding="utf-8")


def verify_s2(pastis_hd_dir: Path, hashes: dict[str, str]) -> None:
    """Verify every packed sha against ``pastis_hd_dir/DATA_S2/<name>``.

    ``snapshot_download(local_dir=...)`` keeps the repo's own directory
    structure, so the files live under ``PASTIS-HD/DATA_S2`` -- pointing the
    check at the download dir itself reported every file missing (review
    round 3 #1a).
    """
    verify_tree(pastis_hd_dir / "DATA_S2", hashes, "S2")


def collect_run(source_run_dir: Path, notebook_runs: Path, run_id: str) -> Path:
    """Copy one finished run into the Drive-side staging dir.

    Copying *into* the shared parent flattened the first run and made the
    second call fail with FileExistsError (review round 3 #1c); the target is
    per-run.
    """
    if not source_run_dir.is_dir():
        raise FileNotFoundError(f"run directory missing: {source_run_dir}")
    destination = notebook_runs / run_id
    if destination.exists():
        raise FileExistsError(f"run already collected: {destination}")
    shutil.copytree(source_run_dir, destination)
    return destination


def run_state(run_dir: Path) -> str:
    """Classify an existing run directory for the notebook's resume logic.

    ``absent``  -> dispatch it
    ``completed`` -> copy it to the Drive staging dir (if not yet collected)
    ``unfinished`` -> stop: per AGENTS a failed run is retried as a new -vN id
                     or moved to quarantine, never silently skipped
                     (review 2026-11-04 round 4 #2)
    """
    import json

    record = run_dir / "run.json"
    if not record.is_file():
        return "absent"
    try:
        status = json.loads(record.read_text(encoding="utf-8")).get("status", "")
    except (OSError, ValueError):
        return "unfinished"
    return "completed" if status == "completed" else "unfinished"


def sync_tree(source_dir: Path, target_dir: Path, hashes: dict[str, str], label: str) -> None:
    """Copy missing/corrupt files from the Drive cache, then verify all.

    The notebook used to call ``copytree`` only when the target directory was
    absent or empty: a partially copied tree was left incomplete, and an empty
    directory made ``copytree`` fail (review 2026-11-04 round 9).
    """
    if not hashes:
        raise SystemExit(f"empty {label} sha256 list; refusing to stage nothing")
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name, digest in sorted(hashes.items()):
        target = target_dir / name
        if target.is_file() and file_sha256(target) == digest:
            continue
        origin = source_dir / name
        if not origin.is_file():
            raise SystemExit(f"{origin} missing from the Drive download")
        shutil.copy2(origin, target)
        copied += 1
    verify_tree(target_dir, hashes, label)
    if copied:
        print(f"{label}: copied {copied} missing/corrupt file(s)")


def stage_local_data(drive_pastis_hd: Path, local_pastis_hd: Path, hashes: dict[str, str]) -> None:
    """Copy the verified DATA_S2 tree from Drive onto the session's local disk.

    Reading 1,455 stacks per epoch over Drive FUSE is slow and quote-prone
    (review round 4); Drive keeps the download cache, training reads local.
    """
    import shutil

    if not hashes:
        raise SystemExit("empty S2 sha256 list; refusing to stage nothing")
    source = drive_pastis_hd / "DATA_S2"
    target = local_pastis_hd / "DATA_S2"
    target.mkdir(parents=True, exist_ok=True)
    present = {path.name for path in target.glob("S2_*.npy")}
    missing = sorted(set(hashes) - present)
    if missing:
        for name in missing:
            origin = source / name
            if not origin.is_file():
                raise SystemExit(f"{origin} missing from the Drive download")
            shutil.copy2(origin, target / name)
    verify_s2(local_pastis_hd, hashes)


def unpack_payload(pack: Path, data_root: Path, require_annotations: bool = True) -> None:
    """Extract sidecars.zip / annotations.zip into the data root.

    The zip members already carry the manifest-relative prefix
    (``derived/aef-2019/...``); extracting into ``data_root/derived`` produced
    ``derived/derived/...`` and the worker could not find the labels
    (review 2026-11-04 round 5). After unpacking, the expected paths are
    asserted so a repacked zip with a different layout fails loudly.
    """
    import zipfile

    expected = [Path("derived/aef-2019/features/2019")]
    if require_annotations:
        expected.append(Path("derived/aef-2019/PASTIS-HD/ANNOTATIONS"))
    archives = ["sidecars.zip"] + (["annotations.zip"] if require_annotations else [])
    for archive in archives:
        source = pack / archive
        if not source.is_file():
            raise SystemExit(f"pack is missing {source}")
        with zipfile.ZipFile(source) as bundle:
            bundle.extractall(data_root)
    for relative in expected:
        target = data_root / relative
        if not target.is_dir() or not any(target.iterdir()):
            raise SystemExit(f"unpacked layout wrong: {target} is empty/missing")
