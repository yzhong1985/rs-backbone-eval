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


def test_stale_notes_raise_a_warning_and_are_marked(tmp_path: Path, monkeypatch) -> None:
    """A conclusion older than 30 days must advertise its age (design §九)."""
    from datetime import date, timedelta

    from rs_backbone_eval.portal import app as app_module
    from rs_backbone_eval.registry import TaskNotes, TaskSpec

    write_run(tmp_path, run_payload())
    _fake_paths(tmp_path, monkeypatch)
    spec = TaskSpec.model_validate(
        {
            "task_id": "T1",
            "application": "Flood",
            "dataset_id": "Sen1Floods11",
            "family": "segmentation",
            "sensor": "sentinel-2",
            "time_steps": 1,
            "bands": ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
            "num_outputs": 2,
            "primary_metric": {"name": "water_iou", "direction": "maximize"},
            "official_config_url": "https://example.invalid/t1",
            "notes": TaskNotes(
                updated=date.today() - timedelta(days=45),
                summary="old",
                findings=[],
                caveats=[],
            ),
        }
    )
    monkeypatch.setattr(app_module, "_task_specs", lambda root: {"T1": spec})
    app = create_app(tmp_path, ROOT / "web")

    tasks = asyncio.run(request(app, "/api/tasks")).json()["items"]
    assert next(item for item in tasks if item["task_id"] == "T1")["notes_age_days"] >= 45

    detail = asyncio.run(request(app, "/api/tasks/T1")).json()
    assert detail["notes_age_days"] >= 45
    assert any(warning["kind"] == "stale_notes" for warning in detail["warnings"])


def _cell(
    model="m", track="B1", mean=0.5, *, direction="maximize", init="pretrained", fraction="1.0", n=3
):
    return {
        "track": track,
        "model": model,
        "task": "T1",
        "mean": mean,
        "sd": 0.01,
        "n": n,
        "direction": direction,
        "init": init,
        "label_fraction": fraction,
        "learning_rate": "1e-4",
        "values": {"17": mean},
    }


def test_best_cell_ignores_random_init_and_partial_labels() -> None:
    """Review 2026-10-27 P1: 'strongest' must not be a random-init/holdout cell."""
    from rs_backbone_eval.portal.app import _pick_best

    cells = [
        _cell("default", mean=0.70),
        _cell("random-init", mean=0.95, init="random"),
        _cell("ten-percent", mean=0.99, fraction="0.1"),
    ]
    best = _pick_best(cells)

    assert best["model"] == "default"
    assert best["init"] == "pretrained"
    assert best["label_fraction"] == "1.0"


def test_best_cell_honours_the_metric_direction() -> None:
    """T7 reports RMSE: the best cell is the smallest value, not the largest."""
    from rs_backbone_eval.portal.app import _pick_best

    cells = [
        _cell("low-rmse", mean=27.5, direction="minimize"),
        _cell("high-rmse", mean=40.0, direction="minimize"),
    ]
    assert _pick_best(cells)["model"] == "low-rmse"


def test_runs_carry_flags_for_superseded_and_reduced(tmp_path: Path, monkeypatch) -> None:
    write_run(tmp_path, run_payload())
    smoke = run_payload("T1-smoke")
    smoke["training"]["epochs_completed"] = 1
    write_run(tmp_path, smoke)
    superseded = run_payload("T1-superseded")
    superseded["tags"] = {"superseded": "replaced by v2"}
    write_run(tmp_path, superseded)
    _fake_paths(tmp_path, monkeypatch)
    app = create_app(tmp_path, ROOT / "web")

    runs = {run["run_id"]: run for run in asyncio.run(request(app, "/api/tasks/T1")).json()["runs"]}
    assert "reduced" in runs["T1-smoke"]["flags"]
    assert "superseded" in runs["T1-superseded"]["flags"]
    assert runs["T1-anysat-seed42"]["flags"] == []


def test_task_without_config_still_returns_cells(tmp_path: Path, monkeypatch) -> None:
    """A historical task missing from configs must not 404 or crash the page."""
    payload = run_payload("T99-run")
    payload["task_id"] = "T99"
    (tmp_path / "T99-run").mkdir(parents=True)
    (tmp_path / "T99-run" / "run.json").write_text(
        __import__("json").dumps(payload), encoding="utf-8"
    )
    _fake_paths(tmp_path, monkeypatch)
    app = create_app(tmp_path, ROOT / "web")

    response = asyncio.run(request(app, "/api/tasks/T99"))
    assert response.status_code == 200
    detail = response.json()
    assert detail["spec"] is None
    assert detail["cells"]
    assert detail["notes"] is None


async def request_with_headers(app, path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request("GET", path, **kwargs)


def test_shared_token_guards_every_route(tmp_path: Path, monkeypatch) -> None:
    """Portal is exposed through frp, so the token is the only gate."""
    write_run(tmp_path, run_payload())
    monkeypatch.setenv("RS_EVAL_PORTAL_TOKEN", "s3cret-token")
    app = create_app(tmp_path, ROOT / "web")

    # Health stays open for the launcher's readiness probe.
    assert asyncio.run(request(app, "/api/health")).status_code == 200
    # Everything else is refused without credentials, and the browser is told
    # which scheme to use.
    refused = asyncio.run(request(app, "/api/summary"))
    assert refused.status_code == 401
    assert refused.headers["www-authenticate"].startswith("Basic")
    assert asyncio.run(request(app, "/")).status_code == 401
    assert asyncio.run(request(app, "/api/matrix")).status_code == 401

    # Basic auth: the user name is ignored, the password is the token.
    ok = asyncio.run(
        request_with_headers(
            app, "/api/summary", headers={"Authorization": "Basic " + _b64("anyone:s3cret-token")}
        )
    )
    assert ok.status_code == 200

    wrong = asyncio.run(
        request_with_headers(
            app, "/api/summary", headers={"Authorization": "Basic " + _b64("anyone:nope")}
        )
    )
    assert wrong.status_code == 401


def _b64(value: str) -> str:
    import base64

    return base64.b64encode(value.encode()).decode()


def test_key_query_sets_the_cookie_and_strips_the_token(tmp_path: Path, monkeypatch) -> None:
    write_run(tmp_path, run_payload())
    monkeypatch.setenv("RS_EVAL_PORTAL_TOKEN", "s3cret-token")
    app = create_app(tmp_path, ROOT / "web")

    response = asyncio.run(request(app, "/", {"key": "s3cret-token"}))
    assert response.status_code == 303
    assert response.headers["location"] in {"http://test/", "http://test"}
    cookie = response.cookies.get("rs_eval_auth")
    assert cookie and cookie != "s3cret-token"

    def with_cookie() -> int:
        async def call() -> int:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
                cookies={"rs_eval_auth": cookie},
            ) as client:
                return (await client.get("/api/summary")).status_code

        return asyncio.run(call())

    assert with_cookie() == 200
    assert asyncio.run(request(app, "/", {"key": "wrong"})).status_code == 401


def test_no_token_means_no_guard(tmp_path: Path, monkeypatch) -> None:
    """Local development (and every existing test) stays unauthenticated."""
    write_run(tmp_path, run_payload())
    monkeypatch.setenv("RS_EVAL_PORTAL_TOKEN", "")
    app = create_app(tmp_path, ROOT / "web")
    assert asyncio.run(request(app, "/api/summary")).status_code == 200
