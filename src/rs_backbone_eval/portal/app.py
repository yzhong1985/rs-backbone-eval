"""FastAPI application for results and allow-listed external downloads."""

from __future__ import annotations

import os
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

from ..audit import stale_run_ids
from ..downloads import (
    DownloadConflictError,
    DownloadManager,
    DownloadNotFoundError,
    DownloadUnavailableError,
)
from ..matrix import build_matrix, model_names
from ..paths import RuntimePaths
from ..report import build_markdown
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


def _downloads_enabled() -> bool:
    return os.getenv("RS_EVAL_DOWNLOADS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}


def _default_download_manager() -> DownloadManager | None:
    paths = RuntimePaths.from_environment()
    catalog_path = paths.config_root / "downloads" / "catalog.yaml"
    if not catalog_path.is_file():
        return None
    return DownloadManager(
        catalog_path,
        paths.data_root,
        paths.weights_root,
        paths.cache_root,
        enabled=_downloads_enabled(),
    )


def create_app(
    runs_root: Path,
    web_root: Path | None = None,
    download_manager: DownloadManager | None = None,
) -> FastAPI:
    repository = RunRepository(runs_root)
    web_root = (web_root or _default_web_root()).resolve()
    download_manager = download_manager or _default_download_manager()
    app = FastAPI(title="RS Backbone Eval Portal", version="0.1.0")

    if download_manager is not None:
        app.add_event_handler("shutdown", download_manager.shutdown)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/summary")
    async def summary() -> dict:
        return repository.summary()

    @app.get("/api/matrix")
    async def matrix() -> dict:
        paths = RuntimePaths.from_environment()
        records, _ = repository.list_records()
        # Same admission rule as the report: a run that predates a value-changing
        # fix must not appear here either, or the portal and the markdown
        # disagree about which cells exist (review 2026-10-22).
        stale = stale_run_ids(records, paths.config_root.parent)
        live = [record for record in records if record.run_id not in stale]
        return build_matrix(live, model_names(paths.config_root))

    @app.get("/api/report.md")
    async def report_markdown() -> Response:
        paths = RuntimePaths.from_environment()
        text = build_markdown(
            runs_root,
            paths.data_root,
            paths.config_root,
            paths.config_root.parent,
        )
        filename = f"rs-eval-report-{datetime.now().strftime('%Y%m%d-%H%M')}.md"
        return Response(
            content=text.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

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

    @app.get("/api/downloads")
    async def downloads() -> dict:
        if download_manager is None:
            return {
                "enabled": False,
                "catalog_version": None,
                "items": [],
                "state_counts": {},
                "resource_count": 0,
                "expected_size_bytes": 0,
                "downloaded_bytes": 0,
            }
        return download_manager.summary()

    @app.get("/api/downloads/{resource_id}")
    async def download_status(resource_id: str) -> dict:
        if download_manager is None:
            raise HTTPException(status_code=503, detail="download catalog is unavailable")
        try:
            return download_manager.status(resource_id)
        except DownloadNotFoundError as exc:
            raise HTTPException(status_code=404, detail="download resource not found") from exc

    @app.post("/api/downloads/{resource_id}/start")
    async def start_download(resource_id: str) -> dict:
        if download_manager is None:
            raise HTTPException(status_code=503, detail="download catalog is unavailable")
        try:
            return download_manager.start(resource_id)
        except DownloadNotFoundError as exc:
            raise HTTPException(status_code=404, detail="download resource not found") from exc
        except DownloadConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except DownloadUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/downloads/{resource_id}/pause")
    async def pause_download(resource_id: str) -> dict:
        if download_manager is None:
            raise HTTPException(status_code=503, detail="download catalog is unavailable")
        try:
            return download_manager.pause(resource_id)
        except DownloadNotFoundError as exc:
            raise HTTPException(status_code=404, detail="download resource not found") from exc
        except DownloadConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/downloads/{resource_id}/verify")
    async def verify_download(resource_id: str) -> dict:
        if download_manager is None:
            raise HTTPException(status_code=503, detail="download catalog is unavailable")
        try:
            return download_manager.verify(resource_id)
        except DownloadNotFoundError as exc:
            raise HTTPException(status_code=404, detail="download resource not found") from exc
        except DownloadConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except DownloadUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

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
