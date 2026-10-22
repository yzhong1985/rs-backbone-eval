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
    ]
    assert {model.model_id for model in models} == {
        "anysat_base",
        "clay_v1_5",
        "olmoearth_v1_2_tiny",
        "olmoearth_v1_2_base",
        "prithvi_eo_v2_300_tl",
    }
    assert {protocol.protocol_id for protocol in protocols} == {
        "frozen_adapter",
        "frozen_adapter_two_stage",
        "frozen_probe",
        "full_finetune",
        "native_input",
        "prithvi_paper_protocol",
        "prithvi_repo_config",
    }


def test_all_tasks_use_non_rgb_bands() -> None:
    tasks = Registry(ROOT / "configs").tasks()
    rgb = {"RED", "GREEN", "BLUE"}
    assert all(set(task.bands) - rgb for task in tasks)
