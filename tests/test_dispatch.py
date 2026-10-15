import json
import sys
from pathlib import Path

import pytest

from rs_backbone_eval.contracts import WorkerRequest
from rs_backbone_eval.data.manifest import write_manifest
from rs_backbone_eval.dispatch import DispatchError, dispatch, plan_dispatch, write_request
from rs_backbone_eval.paths import RuntimePaths

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def paths(tmp_path: Path, make_row) -> RuntimePaths:
    """External roots plus a real T3 manifest and the repo config root."""
    (tmp_path / "runs").mkdir()
    write_manifest(
        tmp_path / "manifests" / "T3" / "abc.jsonl", [make_row("one", "fire-a", "train")]
    )
    write_manifest(
        tmp_path / "manifests" / "T1" / "def.jsonl",
        [make_row("one", "fire-a", "train", task_id="T1")],
    )
    return RuntimePaths(
        data_root=tmp_path,
        weights_root=tmp_path / "weights",
        runs_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        config_root=ROOT / "configs",
    )


def bare_paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(
        data_root=tmp_path / "empty-data",
        weights_root=tmp_path / "weights",
        runs_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        config_root=ROOT / "configs",
    )


def test_plan_dispatch_rejects_unknown_ids(paths: RuntimePaths) -> None:
    with pytest.raises(DispatchError, match="unknown task"):
        plan_dispatch(
            action="probe",
            task_id="T9",
            model_id="prithvi_eo_v2_300_tl",
            protocol_id="prithvi_repo_config",
            seed=42,
            paths=paths,
        )
    with pytest.raises(DispatchError, match="unknown action"):
        plan_dispatch(
            action="explode",
            task_id="T3",
            model_id="prithvi_eo_v2_300_tl",
            protocol_id="prithvi_repo_config",
            seed=42,
            paths=paths,
        )


def test_plan_dispatch_requires_manifest(tmp_path: Path) -> None:
    with pytest.raises(DispatchError, match="build-manifest"):
        plan_dispatch(
            action="probe",
            task_id="T3",
            model_id="prithvi_eo_v2_300_tl",
            protocol_id="prithvi_repo_config",
            seed=42,
            paths=bare_paths(tmp_path),
        )


def test_plan_dispatch_train_seed_guard(paths: RuntimePaths) -> None:
    with pytest.raises(DispatchError, match="seed 99"):
        plan_dispatch(
            action="train",
            task_id="T3",
            model_id="prithvi_eo_v2_300_tl",
            protocol_id="prithvi_repo_config",
            seed=99,
            paths=paths,
        )


def test_write_request_uses_real_config_filenames(paths: RuntimePaths) -> None:
    plan = plan_dispatch(
        action="probe",
        task_id="T1",
        model_id="prithvi_eo_v2_300_tl",
        protocol_id="prithvi_repo_config",
        seed=42,
        paths=paths,
    )

    request = write_request(plan, paths)

    assert plan.request_path.is_file()
    loaded = WorkerRequest.model_validate(json.loads(plan.request_path.read_text(encoding="utf-8")))
    assert loaded == request
    assert Path(loaded.task_config_path).name == "T1_flood.yaml"
    assert Path(loaded.model_config_path).name == "prithvi.yaml"
    assert Path(loaded.protocol_config_path).name == "prithvi_repo_config.yaml"
    assert loaded.manifest_path.endswith(".jsonl")
    assert loaded.action == "probe"


def test_dispatch_runs_stub_worker_and_forwards_events(
    paths: RuntimePaths, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    stub = tmp_path / "stub_worker.py"
    stub.write_text(
        "import json, sys\n"
        "request = json.load(open(sys.argv[1]))\n"
        'print(json.dumps({"run_id": request["run_id"], "level": "info",\n'
        '                  "event": "stub.done", "message": "stub executed"}))\n'
        "run_dir = request['runs_root'] + '/' + request['run_id']\n"
        "open(run_dir + '/marker.txt', 'w').write('ok')\n",
        encoding="utf-8",
    )

    code = dispatch(
        action="probe",
        task_id="T3",
        model_id="prithvi_eo_v2_300_tl",
        protocol_id="prithvi_repo_config",
        seed=42,
        paths=paths,
        worker_command=[sys.executable, str(stub)],
    )

    assert code == 0
    captured = capsys.readouterr()
    assert "[INFO] stub.done: stub executed" in captured.out
    marker = paths.runs_root / "prithvi_eo_v2_300_tl-T3-prithvi_repo_config-probe-42" / "marker.txt"
    assert marker.is_file()


def test_dispatch_rejects_duplicate_run_dir(paths: RuntimePaths) -> None:
    run_dir = paths.runs_root / "prithvi_eo_v2_300_tl-T3-prithvi_repo_config-probe-42"
    run_dir.mkdir(parents=True)
    with pytest.raises(DispatchError, match="already exists"):
        plan_dispatch(
            action="probe",
            task_id="T3",
            model_id="prithvi_eo_v2_300_tl",
            protocol_id="prithvi_repo_config",
            seed=42,
            paths=paths,
        )


def test_repo_model_configs_declare_worker_commands() -> None:
    from rs_backbone_eval.registry import Registry

    models = {spec.model_id: spec for spec in Registry(ROOT / "configs").models()}
    assert models["prithvi_eo_v2_300_tl"].worker_command is not None
    assert models["prithvi_eo_v2_300_tl"].worker_command[0] == "uv"


def test_resume_request_keeps_source_and_requires_a_new_run_id(paths, tmp_path):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"checkpoint validated by worker")
    plan = plan_dispatch(
        action="train",
        task_id="T3",
        model_id="anysat_base",
        protocol_id="full_finetune",
        seed=42,
        paths=paths,
        run_id="resume-attempt",
        resume_from=checkpoint,
    )
    request = write_request(plan, paths)
    assert request.resume_from == str(checkpoint.resolve())
    with pytest.raises(DispatchError, match="already exists"):
        plan_dispatch(
            action="train",
            task_id="T3",
            model_id="anysat_base",
            protocol_id="full_finetune",
            seed=42,
            paths=paths,
            run_id="resume-attempt",
            resume_from=checkpoint,
        )


@pytest.mark.parametrize("model,action", [("clay_v1_5", "train"), ("anysat_base", "probe")])
def test_resume_rejects_unsupported_workers_and_actions(paths, tmp_path, model, action):
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"fixture")
    with pytest.raises(DispatchError, match="only for AnySat"):
        plan_dispatch(
            action=action,
            task_id="T3",
            model_id=model,
            protocol_id="full_finetune",
            seed=42,
            paths=paths,
            resume_from=checkpoint,
        )


def test_resume_rejects_missing_checkpoint(paths, tmp_path):
    with pytest.raises(DispatchError, match="does not exist"):
        plan_dispatch(
            action="train",
            task_id="T3",
            model_id="anysat_base",
            protocol_id="full_finetune",
            seed=42,
            paths=paths,
            resume_from=tmp_path / "missing.pt",
        )
