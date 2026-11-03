"""Regression guards for the visualisation generator (review 2026-11-03)."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    target = ROOT / "scripts/visualize_run.py"
    spec = importlib.util.spec_from_file_location("visualize_run", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda_indexed_devices_need_the_idle_check() -> None:
    module = _module()
    assert module._is_cuda_device("cuda")
    assert module._is_cuda_device("cuda:0")
    assert module._is_cuda_device("cuda:1")
    assert not module._is_cuda_device("cpu")


def test_anchor_decode_weights_are_in_the_replay_fingerprint() -> None:
    module = _module()
    assert "src/rs_backbone_eval/detection/anchors.py" in module.INFERENCE_FILES


def test_every_inference_import_is_fingerprinted() -> None:
    """A module imported by the inference path must be in INFERENCE_FILES.

    New dependencies were repeatedly missed (anchors.py, then fpn.py/heads.py),
    so this walks the imports of the entry points instead of trusting a hand
    list. Training-only modules are explicitly excluded with a reason.
    """
    import ast

    module = _module()
    entry_points = [
        "src/rs_backbone_eval/detection/detector.py",
        "workers/lim/src/lim_worker/t10.py",
        "workers/lim/src/lim_worker/replay.py",
    ]
    training_only = {
        "rs_backbone_eval.checkpoint",
        "rs_backbone_eval.contracts",
        "rs_backbone_eval.evaluation",
        "rs_backbone_eval.matrix",
        "rs_backbone_eval.normalization",
        "rs_backbone_eval.repro",
        "rs_backbone_eval.selection",
        "rs_backbone_eval.worker_runtime",
    }

    def imported_modules(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if "src/rs_backbone_eval" in str(path):
            base_parts = list(path.relative_to(ROOT / "src").with_suffix("").parts)
        else:
            base_parts = list(path.relative_to(ROOT / "workers/lim/src").with_suffix("").parts)
        package = base_parts[:-1]
        found: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                up = node.level - 1
                prefix_parts = package[: len(package) - up] if up else package
                target = ".".join(prefix_parts + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if target.startswith("rs_backbone_eval."):
                found.add(target)
        return found

    missing = []
    for entry in entry_points:
        for name in imported_modules(ROOT / entry):
            if name in training_only:
                continue
            relative = "src/" + name.replace(".", "/") + ".py"
            if relative not in module.INFERENCE_FILES:
                missing.append((entry, name, relative))
    assert not missing, f"inference imports missing from INFERENCE_FILES: {missing}"
