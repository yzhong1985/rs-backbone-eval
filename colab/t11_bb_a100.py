# %% [markdown]
# Colab A100 dispatch for the LIM T11 BB track (route C: data from the
# hf-mirror straight into Drive, uploaded pack is small).
#
# Upload `~/rs-eval/drive-t11-pack-c/` (~12 MB: manifest, sidecars, sha
# lists, repo bundle) to Drive as `MyDrive/rs-t11-pack`. DATA_S2 and
# ANNOTATIONS are downloaded ONCE into Drive from hf-mirror (verified
# byte-identical to the local ledger 2026-11-04) and reused by later
# sessions; the LIM checkpoint is supplied separately at
# `rs-t11-pack/lim/mit_huge_600.pth`. Until it exists the notebook only
# warms the data cache and exits.

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
ANN_HASHES = json.loads((PACK / "annotations_sha256.json").read_text())
assert S2_HASHES and ANN_HASHES, "empty sha lists in the pack"

# %%
# 1. Drive-side trees (download cache + persistent data + runs mirror)
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

# %%
# 2. bundle checkout pinned to the packed commit (repo lives on local disk:
#    Drive FUSE is unreliable for venvs, review round 5)
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

lib.unpack_payload(PACK, LOCAL_DATA, require_annotations=False)

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
# 4. DATA_S2 + ANNOTATIONS from the hf-mirror into Drive (once), verified
#    against the packed sha lists; the marker makes later sessions skip the
#    45 GB re-hash.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
files_present = (
    sum(1 for _ in (DRIVE_PASTIS_HD / "DATA_S2").glob("S2_*.npy"))
    if (DRIVE_PASTIS_HD / "DATA_S2").is_dir()
    else 0
)
if files_present < len(S2_HASHES):
    snapshot_download(
        repo_id="IGNF/PASTIS-HD",
        repo_type="dataset",
        allow_patterns=["DATA_S2/*", "ANNOTATIONS/*"],
        local_dir=DRIVE_PASTIS_HD.as_posix(),
    )

s2_marker = DRIVE_ROOT / "verified_s2.marker"
ann_marker = DRIVE_ROOT / "verified_annotations.marker"
if lib.verification_fresh(s2_marker, PACK / "s2_sha256.json"):
    print("S2 already verified in a previous session (marker)")
else:
    lib.verify_s2(DRIVE_PASTIS_HD, S2_HASHES)
    lib.mark_verified(s2_marker, PACK / "s2_sha256.json")
if lib.verification_fresh(ann_marker, PACK / "annotations_sha256.json"):
    print("ANNOTATIONS already verified (marker)")
else:
    lib.verify_tree(DRIVE_PASTIS_HD / "ANNOTATIONS", ANN_HASHES, "ANNOTATIONS")
    lib.mark_verified(ann_marker, PACK / "annotations_sha256.json")

# stage the training data to the session-local disk
lib.stage_local_data(DRIVE_PASTIS_HD, LOCAL_PASTIS_HD, S2_HASHES)
local_annotations = LOCAL_PASTIS_HD / "ANNOTATIONS"
if not local_annotations.is_dir() or not any(local_annotations.iterdir()):
    import shutil

    shutil.copytree(DRIVE_PASTIS_HD / "ANNOTATIONS", local_annotations)
print("data staged locally")

# %%
# 5. checkpoint gate: the LIM weights are supplied separately; without them
#    only the data-cache warm-up runs (useful to prepare the Drive tree once).
CHECKPOINT = PACK / "lim/mit_huge_600.pth"
if not CHECKPOINT.is_file():
    print(
        "LIM checkpoint not present at "
        f"{CHECKPOINT}; data cache is warm, dispatch skipped. "
        "Upload the checkpoint and re-run this cell."
    )
    raise SystemExit(0)

# %%
# 6. dispatch environment + GPU-class assertion
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
    checkpoint=CHECKPOINT,
)

# %%
# 7. the formal cell: three protocols x three seeds on one GPU class.
#    Resume policy: completed runs are collected once, unfinished ones stop
#    the notebook for manual handling (-vN or quarantine).


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
