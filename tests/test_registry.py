from pathlib import Path

from rs_backbone_eval.registry import Registry

ROOT = Path(__file__).resolve().parents[1]


def test_registry_contains_expected_matrix() -> None:
    registry = Registry(ROOT / "configs")
    tasks = registry.tasks()
    models = registry.models()
    protocols = registry.protocols()

    # sorted by the registry file names (T10/T11 before the "T1_"-prefixed
    # files because '_' sorts after '0'/'1')
    assert [task.task_id for task in tasks] == [
        "T10",
        "T11D",
        "T11O128",
        "T11O",
        "T11P",
        "T11",
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
        "aef_annual_2019",
        "anysat_base",
        "clay_v1_5",
        "olmoearth_v1_2_tiny",
        "olmoearth_v1_2_base",
        "olmoearth_v13_embedding",
        "prithvi_eo_v2_300_tl",
        "lim_mit_huge",
    }
    assert {protocol.protocol_id for protocol in protocols} == {
        "aef_probe_mlp",
        "aef_probe_mlp_20ep",
        "aef_probe_upernet",
        "aef_probe_upernet_20ep",
        "anysat_s2_20bin",
        "anysat_s2_single",
        "anysat_t11p_distill_o128",
        "anysat_t11p_ft_last3",
        "anysat_t11p_ft_last3_distill",
        "anysat_t11p_ft_last3_wide",
        "anysat_t11p_ft_last3_wide_distill",
        "anysat_t11p_probe_o128",
        "anysat_t11p_probe_raw",
        "anysat_t11p_probe_s128",
        "clay_rgbn_distill_a64",
        "clay_rgbn_distill_o128",
        "clay_rgbn_distill_o64",
        "clay_rgbn_ft_last4",
        "clay_rgbn_ft_last4_distill",
        "clay_rgbn_ft_last4_distill_v1",
        "clay_rgbn_ft_last4_raw1024",
        "clay_rgbn_ft_last4_v1",
        "clay_rgbn_ft_last8",
        "clay_rgbn_ft_last8_distill",
        "clay_rgbn_probe_a64",
        "clay_rgbn_probe_o128",
        "clay_rgbn_probe_o64",
        "clay_rgbn_probe_pca64",
        "clay_rgbn_probe_raw",
        "clay_rgbn_probe_s128",
        "clay_rgbn_probe_s64",
        "clay_rgbn_t8_a64",
        "clay_rgbn_t8_oe128",
        "clay_rgbn_t8_raw",
        "clay_s2_20bin",
        "clay_s2_single",
        "clay_s2six_distill_o128",
        "clay_s2six_ft_last4",
        "clay_s2six_ft_last4_distill",
        "clay_s2six_probe_o128",
        "clay_s2six_probe_raw",
        "clay_s2six_probe_s128",
        "frozen_adapter",
        "frozen_adapter_two_stage",
        "frozen_probe",
        "full_finetune",
        "lim_s2_20bin",
        "lim_s2_20bin_finetune",
        "lim_s2_20bin_finetune_20ep_lr5e5",
        "lim_s2_20bin_finetune_lr1e5",
        "lim_s2_20bin_finetune_lr5e5",
        "lim_s2_median",
        "lim_s2_single",
        "native_input",
        "olmoearth_s2_20bin",
        "olmoearth_s2_single",
        "olmoearth_v13_probe_mlp",
        "olmoearth_v13_probe_mlp_128",
        "olmoearth_v13_probe_mlp_128_20ep",
        "olmoearth_v13_probe_mlp_20ep",
        "olmoearth_v13_probe_upernet",
        "olmoearth_v13_probe_upernet_128",
        "olmoearth_v13_probe_upernet_128_20ep",
        "olmoearth_v13_probe_upernet_20ep",
        "prithvi_paper_protocol",
        "prithvi_repo_config",
        "prithvi_t11p_distill_o128",
        "prithvi_t11p_ft_last4",
        "prithvi_t11p_ft_last4_distill",
        "prithvi_t11p_probe_o128",
        "prithvi_t11p_probe_raw",
        "prithvi_t11p_probe_s128",
        "rotated_finetune",
        "rotated_frozen",
        "upernet_finetune",
        "upernet_frozen",
    }


# T9 (iSAID) is deliberately RGB-only: it is LIM's home-turf benchmark, where
# the other backbones are the away side. Every other task must carry a non-RGB
# band, so a task that quietly drops one still fails this guard.
# T10 (FAIR1M) is high-resolution aerial RGB for the same reason, so it joins
# the allow-list explicitly rather than by accident.
RGB_ONLY_TASKS = {"T9", "T10"}


def test_all_other_tasks_use_non_rgb_bands() -> None:
    tasks = Registry(ROOT / "configs").tasks()
    rgb = {"RED", "GREEN", "BLUE"}
    offenders = [
        task.task_id
        for task in tasks
        if task.task_id not in RGB_ONLY_TASKS and not (set(task.bands) - rgb)
    ]
    assert offenders == []


def test_lr_exploration_protocols_declare_their_learning_rates() -> None:
    registry = Registry(ROOT / "configs")
    protocols = {protocol.protocol_id: protocol for protocol in registry.protocols()}
    baseline = protocols["lim_s2_20bin_finetune"]
    five = protocols["lim_s2_20bin_finetune_lr5e5"]
    one = protocols["lim_s2_20bin_finetune_lr1e5"]
    assert baseline.backbone_learning_rate is None
    assert five.learning_rate == 0.001
    assert five.backbone_learning_rate == 0.00005
    assert one.learning_rate == 0.001
    assert one.backbone_learning_rate == 0.00001
    protocol = protocols["lim_s2_20bin_finetune_20ep_lr5e5"]
    assert protocol.epochs == 20
    assert protocol.learning_rate == 0.001
    assert protocol.backbone_learning_rate == 0.00005


def test_budget_diagnostic_protocols_declare_their_panel() -> None:
    """20-epoch probes must be marked diagnostic, not primary."""
    protocols = {p.protocol_id: p for p in Registry(ROOT / "configs").protocols()}
    for name in (
        "olmoearth_v13_probe_mlp_20ep",
        "olmoearth_v13_probe_upernet_20ep",
        "olmoearth_v13_probe_mlp_128_20ep",
        "olmoearth_v13_probe_upernet_128_20ep",
        "aef_probe_mlp_20ep",
        "aef_probe_upernet_20ep",
    ):
        assert protocols[name].panel == "diagnostic"
        assert protocols[name].epochs == 20
        assert protocols[name].seeds == [17, 42, 73]
    assert protocols["olmoearth_v13_probe_upernet"].panel == "primary"


def test_every_aef_protocol_is_mapped_to_a_head() -> None:
    """Every protocol with workers: [aef] needs a PROTOCOL_HEADS entry.

    The 20ep and 128 protocols were registered without the mapping twice; the
    worker then fails at ``aef worker has no head`` and a STOP_ON_FAIL lane
    stops at its first run (review 2026-11-08).
    """
    import ast

    registry = Registry(ROOT / "configs")
    aef = {
        protocol.protocol_id
        for protocol in registry.protocols()
        if protocol.workers and "aef" in protocol.workers
    }
    source = (ROOT / "workers" / "aef" / "src" / "aef_worker" / "runner.py").read_text(
        encoding="utf-8"
    )
    mapping = None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) == "PROTOCOL_HEADS" for target in node.targets
        ):
            mapping = ast.literal_eval(node.value)
    assert mapping is not None, "PROTOCOL_HEADS assignment not found"
    assert aef == set(mapping), f"unmapped protocols: {sorted(aef - set(mapping))}"
