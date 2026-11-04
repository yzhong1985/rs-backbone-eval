# %% [markdown]
# Colab A100 dispatch for the LIM T11 BB track (single / median / 20-bin).
# Upload `~/rs-eval/drive-t11-pack-v2` to Drive as `MyDrive/rs-t11-pack`,
# pick an A100 40GB runtime and execute the cells top-down. Every step that
# could silently pass is turned into a hard stop (review 2026-11-04):
# provenance checks, per-file sha verification of the HuggingFace download,
# GPU-class assertion and forced run-id suffixes.

# %%
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import torch
from google.colab import drive
from huggingface_hub import snapshot_download

drive.mount("/content/drive")
PACK = Path("/content/drive/MyDrive/rs-t11-pack")
assert PACK.is_dir(), f"upload pack not mounted at {PACK}"
PROVENANCE = json.loads((PACK / "PROVENANCE.json").read_text())
S2_HASHES = json.loads((PACK / "s2_sha256.json").read_text())
assert S2_HASHES, "empty s2 sha256 list in the pack; refusing to proceed"
print(f"pack ok: {len(S2_HASHES)} S2 files listed")

# %%
# Work root: keep the big trees OUT of /content so a runtime crash does not
# lose the already-verified DATA_S2.
ROOT = Path("/content/drive/MyDrive/rs-t11-work")
ROOT.mkdir(exist_ok=True)
DATA_ROOT = ROOT / "data"
(DATA_ROOT / "derived/aef-2019/PASTIS-HD").mkdir(parents=True, exist_ok=True)

# 1. unpack the small payload zips into the manifest-relative layout
with zipfile.ZipFile(PACK / "sidecars.zip") as bundle:
    bundle.extractall(DATA_ROOT / "derived")
with zipfile.ZipFile(PACK / "annotations.zip") as bundle:
    bundle.extractall(DATA_ROOT / "derived")
print("sidecars + annotations unpacked")

# %%
# 2. bundle checkout at the pinned commit (this is what run.json records)
REPO = ROOT / "rs-backbone-eval"
if not REPO.is_dir():
    subprocess.run(
        ["git", "clone", "-q", (PACK / "repo/rs-backbone-eval.bundle").as_posix(), REPO], check=True
    )
HEAD = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
).stdout.strip()
assert HEAD in PROVENANCE["repo_commit"], f"bundle HEAD {HEAD} is not the packed commit"
print(f"bundle HEAD {HEAD}")

# %%
# 3. uv environment: the ROOT project carries the dispatch control layer
# (torch stays out of it by design); its worker_command spawns the lim env.
subprocess.run(["pip", "install", "-q", "uv"], check=True)
subprocess.run(["uv", "sync"], cwd=REPO, check=True)

# %%
# 4. HuggingFace download of DATA_S2 straight into Drive, then verify
#    every file against the packed sha256 list before dispatching anything.
S2_DEST = DATA_ROOT / "derived/aef-2019/PASTIS-HD/DATA_S2"
have = sum(1 for _ in S2_DEST.glob("S2_*.npy")) if S2_DEST.is_dir() else 0
if have < len(S2_HASHES):
    snapshot_download(repo_id="IGNF/PASTIS-HD", allow_patterns=["DATA_S2/*"], local_dir=S2_DEST)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_all() -> None:
    bad, missing = [], []
    for name, expected in sorted(S2_HASHES.items()):
        path = S2_DEST / name
        if not path.exists():
            missing.append(name)
        elif _sha(path) != expected:
            bad.append(name)
    if missing or bad:
        raise SystemExit(f"HF download mismatch: missing={missing[:5]} bad={bad[:5]}")


_verify_all()
print(f"all {len(S2_HASHES)} S2 files verified")

# %%
# 5. environment for dispatch: A100 class assertion + weights + env pin
gpu = torch.cuda.get_device_name(0).lower()
assert "a100" in gpu and "40gb" in gpu, f"expected A100 40GB, got {gpu}"
ENV = {
    **os.environ,
    "LIM_CHECKPOINT": (PACK / "lim/mit_huge_600.pth").as_posix(),
    "RS_EVAL_MANIFEST": (PACK / "manifest").glob("*.jsonl").__next__().as_posix(),
    "RS_EVAL_SKIP_ENV": "1",
    "OMP_NUM_THREADS": "8",
}


def _run(proto: str, seed: int, run_id: str) -> None:
    local = ROOT / "runs" / run_id
    if local.is_dir():
        print(f"skip {run_id} (exists)")
        return
    subprocess.run(
        [
            "uv",
            "run",
            "rs-eval",
            "dispatch",
            "--action",
            "train",
            "--task",
            "T11",
            "--model",
            "lim_mit_huge",
            "--protocol",
            proto,
            "--seed",
            str(seed),
            "--run-id",
            run_id,
        ],
        cwd=REPO,
        check=True,
        env=ENV,
    )


# %%
# 6. the formal cell: all three protocols, all three seeds, one GPU class.
#    run ids carry the mandatory -a100 suffix (no unsuffixed handwritten ids).
for proto in ("lim_s2_single", "lim_s2_median", "lim_s2_20bin"):
    for seed in (17, 42, 73):
        _run(proto, seed, f"{proto}-train-{seed}-a100")
        shutil.copytree(
            ROOT / "runs" / f"{proto}-train-{seed}-a100", PACK / "runs", dirs_exist_ok=False
        )

print("cell complete; copy runs/ from the pack back into the local ledger")
