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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_s2(pastis_hd_dir: Path, hashes: dict[str, str]) -> None:
    """Verify every packed sha against ``pastis_hd_dir/DATA_S2/<name>``.

    ``snapshot_download(local_dir=...)`` keeps the repo's own directory
    structure, so the files live under ``PASTIS-HD/DATA_S2`` -- pointing the
    check at the download dir itself reported every file missing (review
    round 3 #1a).
    """
    if not hashes:
        raise SystemExit("empty S2 sha256 list; refusing to verify nothing")
    data_dir = pastis_hd_dir / "DATA_S2"
    missing = [name for name in sorted(hashes) if not (data_dir / name).is_file()]
    if missing:
        raise SystemExit(f"S2 download incomplete: missing {missing[:5]} ({len(missing)})")
    bad = [name for name in sorted(hashes) if _sha256(data_dir / name) != hashes[name]]
    if bad:
        raise SystemExit(f"S2 download corrupt: {bad[:5]} ({len(bad)})")


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


def unpack_payload(pack: Path, data_root: Path) -> None:
    """Extract sidecars.zip / annotations.zip into the data root.

    The zip members already carry the manifest-relative prefix
    (``derived/aef-2019/...``); extracting into ``data_root/derived`` produced
    ``derived/derived/...`` and the worker could not find the labels
    (review 2026-11-04 round 5). After unpacking, the expected paths are
    asserted so a repacked zip with a different layout fails loudly.
    """
    import zipfile

    expected = (
        Path("derived/aef-2019/features/2019"),
        Path("derived/aef-2019/PASTIS-HD/ANNOTATIONS"),
    )
    for archive in ("sidecars.zip", "annotations.zip"):
        source = pack / archive
        if not source.is_file():
            raise SystemExit(f"pack is missing {source}")
        with zipfile.ZipFile(source) as bundle:
            bundle.extractall(data_root)
    for relative in expected:
        target = data_root / relative
        if not target.is_dir() or not any(target.iterdir()):
            raise SystemExit(f"unpacked layout wrong: {target} is empty/missing")
