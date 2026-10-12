import asyncio
from pathlib import Path

import httpx
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
    assert downloads["resource_count"] == 10
    assert asyncio.run(request(app, "/")).status_code == 200


def test_portal_returns_404_for_unknown_run(tmp_path: Path) -> None:
    app = create_app(tmp_path, ROOT / "web")
    assert asyncio.run(request(app, "/api/runs/missing-run")).status_code == 404
    response = asyncio.run(request(app, "/api/downloads/missing/start", method="POST"))
    assert response.status_code == 404
