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
    """Walk the import graph, not just the entry points' direct imports.

    New dependencies were repeatedly missed (anchors.py, then fpn.py/heads.py),
    so this resolves imports recursively through both ``rs_backbone_eval.*`` and
    ``lim_worker.*``. Training-only modules are cut off explicitly; anything
    reachable from the inference path must be in INFERENCE_FILES.
    """
    import ast

    module = _module()
    entry_points = [
        ROOT / "src/rs_backbone_eval/detection/detector.py",
        ROOT / "workers/lim/src/lim_worker/t10.py",
        ROOT / "workers/lim/src/lim_worker/replay.py",
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

    def resolve(name: str):
        if name.startswith("rs_backbone_eval."):
            return ROOT / "src" / (name.replace(".", "/") + ".py")
        if name.startswith("lim_worker."):
            return ROOT / "workers/lim/src" / (name.replace(".", "/") + ".py")
        return None

    def imports(path: Path) -> set[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if "src/rs_backbone_eval" in str(path):
            package = list(path.relative_to(ROOT / "src").with_suffix("").parts[:-1])
        else:
            package = list(path.relative_to(ROOT / "workers/lim/src").with_suffix("").parts[:-1])
        found: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level:
                up = node.level - 1
                prefix = package[: len(package) - up] if up else package
                target = ".".join(prefix + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if target.startswith(("rs_backbone_eval.", "lim_worker.")):
                found.add(target)
        return found

    missing = []
    seen: set[Path] = set()
    queue = list(entry_points)
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        relative = str(path.relative_to(ROOT))
        if relative not in module.INFERENCE_FILES:
            missing.append((relative, "entry or reachable file"))
        for name in imports(path):
            if name in training_only:
                continue
            target = resolve(name)
            if target is None or not target.is_file():
                continue
            if str(target.relative_to(ROOT)) not in module.INFERENCE_FILES:
                missing.append((str(target.relative_to(ROOT)), f"imported by {relative}"))
            queue.append(target)
    assert not missing, f"inference dependencies missing from INFERENCE_FILES: {missing}"
