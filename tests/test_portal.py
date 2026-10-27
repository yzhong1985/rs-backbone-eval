import asyncio
from pathlib import Path

import httpx
import pytest
from test_results import run_payload, write_run

from rs_backbone_eval.portal.app import create_app

ROOT = Path(__file__).resolve().parents[1]


async def request(
    app, path: str, params: dict | None = None, method: str = "GET"
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, params=params)


def test_portal_health_and_run_api(tmp_path: Path) -> None:
    write_run(tmp_path, run_payload())
    app = create_app(tmp_path, ROOT / "web")

    assert asyncio.run(request(app, "/api/health")).json() == {"status": "ok"}
    assert asyncio.run(request(app, "/api/summary")).json()["run_count"] == 1
    runs = asyncio.run(request(app, "/api/runs", {"task_id": "T1"})).json()["items"]
    assert runs[0]["primary_metric"]["name"] == "water_iou"
    assert asyncio.run(request(app, "/api/runs/T1-anysat-seed42")).status_code == 200
    downloads = asyncio.run(request(app, "/api/downloads")).json()
    assert downloads["resource_count"] == 12
    assert asyncio.run(request(app, "/")).status_code == 200


def test_portal_returns_404_for_unknown_run(tmp_path: Path) -> None:
    app = create_app(tmp_path, ROOT / "web")
    assert asyncio.run(request(app, "/api/runs/missing-run")).status_code == 404
    response = asyncio.run(request(app, "/api/downloads/missing/start", method="POST"))
    assert response.status_code == 404


def test_matrix_endpoint_drops_pre_fix_runs(tmp_path: Path, monkeypatch) -> None:
    """Review 2026-10-22: the report filters stale runs, so the portal must too."""
    from rs_backbone_eval.portal import app as app_module

    write_run(tmp_path, run_payload())
    monkeypatch.setattr(app_module, "excluded_run_ids", lambda records, repo: {"T1-anysat-seed42"})
    app = create_app(tmp_path, ROOT / "web")
    cells = asyncio.run(request(app, "/api/matrix")).json()["cells"]
    assert cells == []


def test_report_endpoint_survives_a_smoke_only_label_fraction(tmp_path: Path, monkeypatch) -> None:
    """Review 2026-10-23: the empty few-shot group used to make /api/report.md 500."""
    from rs_backbone_eval import report as report_module

    monkeypatch.setattr(report_module, "excluded_run_ids", lambda records, repo: set())
    monkeypatch.setattr(report_module, "stale_run_ids", lambda records, repo: set())
    official = run_payload()
    official["tags"]["label_fraction"] = "1.0"
    official["training"]["epochs_completed"] = 50
    write_run(tmp_path, official)
    smoke = run_payload("T1-smoke-17")
    smoke["seed"] = 17
    smoke["tags"]["label_fraction"] = "0.1"
    smoke["training"]["epochs_completed"] = 1
    write_run(tmp_path, smoke)

    app = create_app(tmp_path, ROOT / "web")
    response = asyncio.run(request(app, "/api/report.md"))
    assert response.status_code == 200
    assert "标注=0.1" not in response.text


def _fake_paths(tmp_path: Path, monkeypatch, *, excluded: set[str] | None = None):
    """Point the portal's RuntimePaths at the repo configs and a tmp data root."""
    from rs_backbone_eval.paths import RuntimePaths
    from rs_backbone_eval.portal import app as app_module

    data_root = tmp_path / "data"
    (data_root / "manifests" / "T1").mkdir(parents=True)
    (data_root / "manifests" / "T1" / "m.jsonl").write_text(
        "\n".join(
            [
                '{"split": "train", "group_id": "g1", "image_paths": ["a.tif"]}',
                '{"split": "test", "group_id": "g1", "image_paths": ["a.tif"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    paths = RuntimePaths(
        data_root=data_root,
        weights_root=tmp_path / "weights",
        runs_root=tmp_path,
        cache_root=tmp_path / "cache",
        config_root=ROOT / "configs",
    )
    monkeypatch.setattr(
        app_module,
        "RuntimePaths",
        type("Stub", (), {"from_environment": staticmethod(lambda: paths)}),
    )
    # The audit walks real git ancestry, which the synthetic payloads cannot
    # satisfy; every existing portal test stubs this out for the same reason.
    monkeypatch.setattr(app_module, "excluded_run_ids", lambda records, repo: set(excluded or ()))
    return paths


def test_tasks_endpoint_lists_every_registered_task(tmp_path: Path, monkeypatch) -> None:
    write_run(tmp_path, run_payload())
    _fake_paths(tmp_path, monkeypatch)
    app = create_app(tmp_path, ROOT / "web")

    payload = asyncio.run(request(app, "/api/tasks")).json()
    ids = [item["task_id"] for item in payload["items"]]
    assert {"T1", "T2", "T4", "T5", "T8"} <= set(ids)
    t1 = next(item for item in payload["items"] if item["task_id"] == "T1")
    assert t1["metric"] == "water_iou"
    assert t1["cells"] == 1 and t1["seeds_max"] == 1
    assert t1["best"]["mean"] == pytest.approx(0.81)
    assert t1["has_notes"] is True
    assert any(track["id"] == "B0" for track in payload["tracks"])


def test_task_detail_matches_the_matrix_and_flags_caveats(tmp_path: Path, monkeypatch) -> None:
    write_run(tmp_path, run_payload())
    _fake_paths(tmp_path, monkeypatch, excluded={"T1-smoke-run"})
    smoke = run_payload("T1-smoke-run")
    smoke["training"]["epochs_completed"] = 1
    write_run(tmp_path, smoke)
    app = create_app(tmp_path, ROOT / "web")

    detail = asyncio.run(request(app, "/api/tasks/T1")).json()
    matrix = asyncio.run(request(app, "/api/matrix")).json()
    assert detail["cells"] == [cell for cell in matrix["cells"] if cell["task"] == "T1"]
    assert detail["notes"]["summary"]
    kinds = {warning["kind"] for warning in detail["warnings"]}
    assert {"single_seed", "group_overlap", "excluded_runs"} <= kinds
    assert any(run["excluded"] for run in detail["runs"])


def test_unknown_task_is_404(tmp_path: Path, monkeypatch) -> None:
    _fake_paths(tmp_path, monkeypatch)
    app = create_app(tmp_path, ROOT / "web")
    assert asyncio.run(request(app, "/api/tasks/T99")).status_code == 404
