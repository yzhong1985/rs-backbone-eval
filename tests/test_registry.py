from pathlib import Path

from rs_backbone_eval.registry import Registry

ROOT = Path(__file__).resolve().parents[1]


def test_registry_contains_expected_matrix() -> None:
    registry = Registry(ROOT / "configs")
    tasks = registry.tasks()
    models = registry.models()
    protocols = registry.protocols()

    assert [task.task_id for task in tasks] == [
        "T1",
        "T1S",
        "T2",
        "T3",
        "T4",
        "T5",
        "T6a",
        "T6b",
        "T7",
        "T8",
        "T9",
    ]
    assert {model.model_id for model in models} == {
        "anysat_base",
        "clay_v1_5",
        "olmoearth_v1_2_tiny",
        "olmoearth_v1_2_base",
        "prithvi_eo_v2_300_tl",
        "lim_mit_huge",
    }
    assert {protocol.protocol_id for protocol in protocols} == {
        "frozen_adapter",
        "frozen_adapter_two_stage",
        "frozen_probe",
        "full_finetune",
        "native_input",
        "prithvi_paper_protocol",
        "prithvi_repo_config",
        "upernet_finetune",
        "upernet_frozen",
    }


# T9 (iSAID) is deliberately RGB-only: it is LIM's home-turf benchmark, where
# the other backbones are the away side. Every other task must carry a non-RGB
# band, so a task that quietly drops one still fails this guard.
RGB_ONLY_TASKS = {"T9"}


def test_all_other_tasks_use_non_rgb_bands() -> None:
    tasks = Registry(ROOT / "configs").tasks()
    rgb = {"RED", "GREEN", "BLUE"}
    offenders = [
        task.task_id
        for task in tasks
        if task.task_id not in RGB_ONLY_TASKS and not (set(task.bands) - rgb)
    ]
    assert offenders == []
