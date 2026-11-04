# %% [markdown]
# Colab A100 dispatch for the LIM T11 BB track (single / median / 20-bin).
#
# Upload `~/rs-eval/drive-t11-pack-v4` to Drive as `MyDrive/rs-t11-pack`,
# pick an A100 40GB runtime and execute the cells top-down. The verification
# and copy-back logic lives in `colab/dispatch_lib.py` (committed, unit
# tested locally); this file is thin glue only.

# %%
import json
import subprocess
import sys
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

# %%
# 1. working trees on Drive (survive runtime crashes)
ROOT = Path("/content/drive/MyDrive/rs-t11-work")
DATA_ROOT = ROOT / "data"
RUNS_ROOT = ROOT / "runs"
CACHE_ROOT = ROOT / "cache"
PASTIS_HD = DATA_ROOT / "derived/aef-2019/PASTIS-HD"
for directory in (DATA_ROOT, RUNS_ROOT, CACHE_ROOT, PASTIS_HD):
    directory.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(PACK / "sidecars.zip") as bundle:
    bundle.extractall(DATA_ROOT / "derived")
with zipfile.ZipFile(PACK / "annotations.zip") as bundle:
    bundle.extractall(DATA_ROOT / "derived")

# %%
# 2. bundle checkout pinned to the packed commit
REPO = ROOT / "rs-backbone-eval"
if not REPO.is_dir():
    subprocess.run(
        ["git", "clone", "-q", (PACK / "repo/rs-backbone-eval.bundle").as_posix(), REPO],
        check=True,
    )
HEAD = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
).stdout.strip()
assert PROVENANCE["repo_commit"] == HEAD, (
    f"bundle HEAD {HEAD} != packed {PROVENANCE['repo_commit']}"
)
sys.path.insert(0, (REPO / "colab").as_posix())
import dispatch_lib as lib  # noqa: E402

# %%
# 3. environments: root project carries the dispatch control layer (torch-free)
subprocess.run(["pip", "install", "-q", "uv"], check=True)
subprocess.run(["uv", "sync"], cwd=REPO, check=True)

# %%
# 4. DATA_S2 straight from the official snapshot, then sha-verify every file.
#    local_dir keeps the repo structure, so files land in PASTIS-HD/DATA_S2.
files_present = sum(1 for _ in (PASTIS_HD / "DATA_S2").glob("S2_*.npy"))
if files_present < len(S2_HASHES):
    snapshot_download(
        repo_id="IGNF/PASTIS-HD",
        allow_patterns=["DATA_S2/*"],
        local_dir=PASTIS_HD.as_posix(),
    )
lib.verify_s2(PASTIS_HD, S2_HASHES)
print(f"all {len(S2_HASHES)} S2 files verified")

# %%
# 5. dispatch environment (root pinning) + GPU-class assertion
lib.assert_a100_40gb(torch.cuda.get_device_name(0))
MANIFEST = next((PACK / "manifest").glob("*.jsonl"))
ENV = lib.dispatch_env(
    {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/root"},
    repo=REPO,
    pack=PACK,
    data_root=DATA_ROOT,
    runs_root=RUNS_ROOT,
    cache_root=CACHE_ROOT,
    manifest=MANIFEST,
    checkpoint=PACK / "lim/mit_huge_600.pth",
)

# %%
# 6. the formal cell: three protocols x three seeds on one GPU class
NOTEBOOK_RUNS = PACK / "runs"


def dispatch(proto: str, seed: int) -> None:
    run_id = lib.run_id_for(proto, seed)
    if (RUNS_ROOT / run_id).is_dir():
        print(f"skip {run_id} (already run)")
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
    lib.collect_run(RUNS_ROOT / run_id, NOTEBOOK_RUNS, run_id)


for proto in ("lim_s2_single", "lim_s2_median", "lim_s2_20bin"):
    for seed in (17, 42, 73):
        dispatch(proto, seed)

print("cell complete: copy rs-t11-pack/runs/ back into the local ledger")
