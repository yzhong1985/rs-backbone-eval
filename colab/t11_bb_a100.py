# %% [markdown]
# Colab A100 dispatch for the LIM T11 BB track (single / median / 20-bin).
#
# Upload `~/rs-eval/drive-t11-pack-v5` to Drive as `MyDrive/rs-t11-pack`,
# pick an A100 40GB runtime and execute the cells top-down. Verification,
# staging and resume logic live in `colab/dispatch_lib.py` (committed, unit
# tested locally); this file is thin glue only.

# %%
import json
import os
import subprocess
import sys
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
# 1. Drive-side trees (download cache + runs ledger mirror); training data is
#    staged to the session-local disk afterwards.
DRIVE_ROOT = Path("/content/drive/MyDrive/rs-t11-work")
DRIVE_DATA = DRIVE_ROOT / "data"
DRIVE_PASTIS_HD = DRIVE_DATA / "derived/aef-2019/PASTIS-HD"
DRIVE_RUNS = DRIVE_ROOT / "runs"
LOCAL_DATA = Path("/content/rs-t11-data")
LOCAL_PASTIS_HD = LOCAL_DATA / "derived/aef-2019/PASTIS-HD"
LOCAL_CACHE = Path("/content/rs-t11-cache")
for directory in (
    DRIVE_DATA,
    DRIVE_PASTIS_HD,
    DRIVE_RUNS,
    LOCAL_DATA,
    LOCAL_PASTIS_HD,
    LOCAL_CACHE,
):
    directory.mkdir(parents=True, exist_ok=True)

# sidecars/annotations are unpacked after the repo checkout (helper below)

# %%
# 2. bundle checkout pinned to the packed commit
REPO = Path("/content/rs-backbone-eval")
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

lib.unpack_payload(PACK, LOCAL_DATA)

# %%
# 3. environments: --frozen keeps uv.lock intact (a dirty tree would put
#    -dirty into run.json's git_commit)
subprocess.run(["pip", "install", "-q", "uv"], check=True)
subprocess.run(["uv", "sync", "--frozen"], cwd=REPO, check=True)
dirty = subprocess.run(
    ["git", "status", "--porcelain"], cwd=REPO, check=True, capture_output=True, text=True
).stdout.strip()
assert not dirty, f"bundle tree is dirty: {dirty.splitlines()[:3]}"

# %%
# 4. DATA_S2: download to Drive (cache), verify, then stage to local disk
files_present = sum(1 for _ in (DRIVE_PASTIS_HD / "DATA_S2").glob("S2_*.npy"))
if files_present < len(S2_HASHES):
    snapshot_download(
        repo_id="IGNF/PASTIS-HD",
        repo_type="dataset",
        allow_patterns=["DATA_S2/*"],
        local_dir=DRIVE_PASTIS_HD.as_posix(),
    )
lib.verify_s2(DRIVE_PASTIS_HD, S2_HASHES)
lib.stage_local_data(DRIVE_PASTIS_HD, LOCAL_PASTIS_HD, S2_HASHES)
print(f"all {len(S2_HASHES)} S2 files verified and staged locally")

# %%
# 5. dispatch environment: keeps the runtime env (LD_LIBRARY_PATH!) and pins
#    every root; RS_EVAL_REQUIRE_DEVICE=cuda fails loudly instead of CPU
lib.assert_a100_40gb(torch.cuda.get_device_name(0))
MANIFEST = next((PACK / "manifest").glob("*.jsonl"))
ENV = lib.dispatch_env(
    dict(os.environ),
    repo=REPO,
    pack=PACK,
    data_root=LOCAL_DATA,
    runs_root=DRIVE_RUNS,
    cache_root=LOCAL_CACHE,
    manifest=MANIFEST,
    checkpoint=PACK / "lim/mit_huge_600.pth",
)

# %%
# 6. the formal cell: three protocols x three seeds on one GPU class.
#    Resume policy (review round 4 #2): completed runs are collected once,
#    unfinished ones stop the notebook for manual handling (-vN or quarantine).


def dispatch(proto: str, seed: int) -> None:
    run_id = lib.run_id_for(proto, seed)
    run_dir = DRIVE_RUNS / run_id
    state = lib.run_state(run_dir)
    if state == "completed":
        if not (DRIVE_ROOT / "collected" / run_id).is_dir():
            lib.collect_run(run_dir, DRIVE_ROOT / "collected", run_id)
        print(f"skip {run_id} (already completed and collected)")
        return
    if state == "unfinished":
        raise SystemExit(
            f"{run_id} exists but is not completed; retry as a new -vN id or "
            "move it to quarantine per AGENTS.md - do not overwrite"
        )
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
    lib.collect_run(DRIVE_RUNS / run_id, DRIVE_ROOT / "collected", run_id)


for proto in ("lim_s2_single", "lim_s2_median", "lim_s2_20bin"):
    for seed in (17, 42, 73):
        dispatch(proto, seed)

print("cell complete: copy rs-t11-work/collected/ back into the local ledger")
