"""FastAPI application for browsing immutable experiment records."""

from __future__ import annotations

import os
from mimetypes import guess_type
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

from ..results import RunNotFoundError, RunRepository


def _default_web_root() -> Path:
    source_checkout = Path(__file__).resolve().parents[3] / "web"
    return Path(os.getenv("RS_EVAL_WEB_ROOT", source_checkout)).resolve()


def _read_file(path: Path) -> Response:
    media_type = guess_type(path.name)[0] or "application/octet-stream"
    return Response(content=path.read_bytes(), media_type=media_type)


def _safe_web_asset(web_root: Path, relative_path: str) -> Path:
    asset_root = (web_root / "assets").resolve()
    candidate = (asset_root / relative_path).resolve()
    if asset_root not in candidate.parents or not candidate.is_file():
        raise FileNotFoundError(relative_path)
    return candidate


def create_app(runs_root: Path, web_root: Path | None = None) -> FastAPI:
    repository = RunRepository(runs_root)
    web_root = (web_root or _default_web_root()).resolve()
    app = FastAPI(title="RS Backbone Eval Portal", version="0.1.0")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/summary")
    async def summary() -> dict:
        return repository.summary()

    @app.get("/api/runs")
    async def runs(
        task_id: str | None = Query(default=None),
        model_id: str | None = Query(default=None),
        protocol_id: str | None = Query(default=None),
        status: str | None = Query(default=None),
    ) -> dict:
        records, errors = repository.list_records()
        filtered = [
            record
            for record in records
            if (task_id is None or record.task_id == task_id)
            and (model_id is None or record.model_id == model_id)
            and (protocol_id is None or record.protocol_id == protocol_id)
            and (status is None or record.status == status)
        ]
        return {"items": [record.public_summary() for record in filtered], "errors": errors}

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str) -> dict:
        try:
            return repository.get(run_id).model_dump(mode="json")
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/api/artifacts/{run_id}/{artifact_path:path}")
    async def artifact(run_id: str, artifact_path: str) -> Response:
        try:
            return _read_file(repository.artifact_path(run_id, artifact_path))
        except RunNotFoundError as exc:
            raise HTTPException(status_code=404, detail="artifact not found") from exc

    if not web_root.is_dir():
        raise FileNotFoundError(f"Portal web directory not found: {web_root}")

    @app.get("/assets/{asset_path:path}")
    async def asset(asset_path: str) -> Response:
        try:
            return _read_file(_safe_web_asset(web_root, asset_path))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="asset not found") from exc

    @app.get("/")
    async def index() -> Response:
        return _read_file(web_root / "index.html")

    return app
