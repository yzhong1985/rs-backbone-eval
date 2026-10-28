"""FastAPI application for results and allow-listed external downloads."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from datetime import date, datetime
from mimetypes import guess_type
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse, Response

from ..audit import excluded_run_ids
from ..downloads import (
    DownloadConflictError,
    DownloadManager,
    DownloadNotFoundError,
    DownloadUnavailableError,
)
from ..matrix import (
    TRACK_NAMES,
    build_matrix,
    is_reduced_run,
    is_superseded,
    model_names,
)
from ..paths import RuntimePaths
from ..registry import Registry, TaskSpec
from ..report import build_markdown, manifest_summary, split_overlap
from ..results import RunNotFoundError, RunRepository

# Shared-token guard for the public (frp) deployment. The portal itself has no
# user model: a colleague without SSH access gets one long random token, sent
# once as ?key= (sets an HttpOnly cookie) or as HTTP Basic credentials.
AUTH_COOKIE = "rs_eval_auth"
AUTH_REALM = "rs-eval"
AUTH_OPEN_PATHS = {"/api/health"}


def _auth_token() -> str:
    return os.getenv("RS_EVAL_PORTAL_TOKEN", "").strip()


def _cookie_fingerprint(token: str) -> str:
    """The cookie holds a fingerprint, never the token itself."""
    return hashlib.sha256(f"rs-eval:{token}".encode()).hexdigest()


def _basic_password(header: str) -> str | None:
    scheme, _, payload = header.partition(" ")
    if scheme.lower() != "basic" or not payload:
        return None
    try:
        decoded = base64.b64decode(payload).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    _, _, password = decoded.partition(":")
    return password


def _install_auth(app: FastAPI) -> None:
    """Require RS_EVAL_PORTAL_TOKEN on every route when it is configured."""

    @app.middleware("http")
    async def guard(request, call_next):
        token = _auth_token()
        if not token or request.url.path in AUTH_OPEN_PATHS:
            return await call_next(request)
        expected = _cookie_fingerprint(token)
        if hmac.compare_digest(request.cookies.get(AUTH_COOKIE, ""), expected):
            return await call_next(request)
        password = _basic_password(request.headers.get("authorization", ""))
        if password is not None and hmac.compare_digest(password, token):
            return await call_next(request)
        key = request.query_params.get("key")
        if key is not None and hmac.compare_digest(key, token):
            response = RedirectResponse(
                url=str(request.url.remove_query_params("key")), status_code=303
            )
            response.set_cookie(
                AUTH_COOKIE,
                expected,
                httponly=True,
                samesite="lax",
                max_age=60 * 60 * 24 * 30,
            )
            return response
        return Response(
            content=b"authentication required",
            status_code=401,
            media_type="text/plain",
            headers={"WWW-Authenticate": f'Basic realm="{AUTH_REALM}"'},
        )


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


OUT_OF_SUITE = {"T1S", "T8"}
RECENT_RUNS = 20
NOTES_STALE_DAYS = 30
# Random-init and label-fraction runs are separate panels; they must never be
# presented as the default headline of a task (review 2026-10-27, P1).
DEFAULT_INIT = "pretrained"
DEFAULT_FRACTION = "1.0"


def _is_default_condition(cell: dict) -> bool:
    return (
        str(cell.get("init") or DEFAULT_INIT) == DEFAULT_INIT
        and str(cell.get("label_fraction") or DEFAULT_FRACTION) == DEFAULT_FRACTION
    )


def _pick_best(cells: list[dict]) -> dict | None:
    """Strongest default-condition cell, honouring the metric direction.

    T7 reports RMSE (minimize), so "max(mean)" would pick the worst model. The
    direction comes from the cell itself, which the matrix derives from the
    task's primary metric.
    """
    candidates = [cell for cell in cells if _is_default_condition(cell)]
    if not candidates:
        return None

    def score(cell: dict) -> float:
        return cell["mean"] if cell.get("direction", "maximize") == "maximize" else -cell["mean"]

    best = max(candidates, key=score)
    return {
        "track": best["track"],
        "model": best["model"],
        "mean": best["mean"],
        "sd": best["sd"],
        "n": best["n"],
        "direction": best.get("direction", "maximize"),
        "learning_rate": best.get("learning_rate"),
        "init": best.get("init"),
        "label_fraction": best.get("label_fraction"),
    }


def _run_flags(record, excluded: set[str]) -> list[str]:
    """Why a record is or is not a formal result, for the run lists."""
    flags: list[str] = []
    if record.run_id in excluded:
        flags.append("excluded")
    if is_superseded(record):
        flags.append("superseded")
    if is_reduced_run(record):
        flags.append("reduced")
    return flags


def _task_specs(config_root: Path) -> dict[str, TaskSpec]:
    """Task registry, tolerating a checkout without configs (tests, partial deploys)."""
    try:
        return {spec.task_id: spec for spec in Registry(config_root).tasks()}
    except (FileNotFoundError, ValueError):
        return {}


def _cells_by_task(matrix: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for cell in matrix.get("cells", []):
        grouped.setdefault(cell["task"], []).append(cell)
    return grouped


def _task_warnings(
    task_id: str,
    spec: TaskSpec | None,
    cells: list[dict],
    overlap: dict | None,
    excluded_here: int,
) -> list[dict]:
    """Warnings derived from data only: never hand-written, so nothing is forgotten."""
    warnings: list[dict] = []
    if task_id in OUT_OF_SUITE:
        warnings.append(
            {
                "kind": "out_of_suite",
                "text": "域外轨：不并入 Prithvi 官方套件汇总，也不参与主排名。",
            }
        )
    for cell in cells:
        if cell["n"] == 1:
            warnings.append(
                {
                    "kind": "single_seed",
                    "cell": f"{cell['model']}/{cell['track']}",
                    "text": "该单元只有 1 个种子，不能与 3 种子单元并列比较。",
                }
            )
    if overlap and overlap.get("shared_groups"):
        warnings.append(
            {
                "kind": "group_overlap",
                "text": (
                    f"训练与评测共享 {overlap['shared_groups']} 个 group"
                    f"（评测共 {overlap['eval_groups']} 个）：分数属同分布留出，"
                    "不支持跨区域泛化主张。"
                ),
            }
        )
    if excluded_here:
        warnings.append(
            {
                "kind": "excluded_runs",
                "count": excluded_here,
                "text": f"另有 {excluded_here} 条记录因审计或非正式预算被排除，未计入成绩。",
            }
        )
    non_default = [cell for cell in cells if not _is_default_condition(cell)]
    if non_default:
        examples = ", ".join(
            f"{cell['model']}/{cell['track']}"
            + (
                f"(init={cell['init']})"
                if str(cell.get("init") or "") not in {"", DEFAULT_INIT}
                else ""
            )
            + (
                f"(标注={cell['label_fraction']})"
                if str(cell.get("label_fraction") or "") not in {"", DEFAULT_FRACTION}
                else ""
            )
            for cell in non_default[:3]
        )
        warnings.append(
            {
                "kind": "non_default_cells",
                "count": len(non_default),
                "text": (
                    f"{len(non_default)} 个单元不是默认条件（随机初始化或少标注），"
                    f'不参与"最强单元"选择：{examples}。'
                ),
            }
        )
    if spec is not None and spec.notes is not None:
        age = (date.today() - spec.notes.updated).days
        if age > NOTES_STALE_DAYS:
            warnings.append(
                {
                    "kind": "stale_notes",
                    "age_days": age,
                    "text": (
                        f"结论写于 {spec.notes.updated.isoformat()}（{age} 天前），"
                        "可能未反映最新结果。"
                    ),
                }
            )
    if spec is not None and spec.notes is None:
        warnings.append(
            {
                "kind": "no_notes",
                "text": "本任务尚未撰写结论（configs/tasks 里的 notes 为空）。",
            }
        )
    return warnings


def _task_payload(
    task_id: str,
    spec: TaskSpec | None,
    cells: list[dict],
    records,
    excluded: set[str],
    data_root: Path,
    manifests: dict,
    repository: RunRepository,
) -> dict:
    overlap = split_overlap(data_root, task_id)
    manifest_entries = manifests.get(task_id, [])
    excluded_here = sum(
        1 for record in records if record.task_id == task_id and record.run_id in excluded
    )
    runs = sorted(
        (record for record in records if record.task_id == task_id),
        key=lambda record: record.updated_at,
        reverse=True,
    )[:RECENT_RUNS]
    return {
        "task_id": task_id,
        "name": (spec.application if spec else task_id),
        "spec": spec.model_dump(mode="json") if spec is not None else None,
        "dataset": {
            "manifest": manifest_entries,
            "overlap": overlap,
        },
        "cells": cells,
        "notes": spec.notes.model_dump(mode="json") if spec and spec.notes else None,
        "notes_age_days": (
            (date.today() - spec.notes.updated).days if spec and spec.notes else None
        ),
        "warnings": _task_warnings(task_id, spec, cells, overlap, excluded_here),
        "runs": [
            {
                **record.public_summary(),
                "excluded": record.run_id in excluded,
                "flags": _run_flags(record, excluded),
            }
            for record in runs
        ],
    }


def create_app(
    runs_root: Path,
    web_root: Path | None = None,
    download_manager: DownloadManager | None = None,
) -> FastAPI:
    repository = RunRepository(runs_root)
    web_root = (web_root or _default_web_root()).resolve()
    download_manager = download_manager or _default_download_manager()
    app = FastAPI(title="RS Backbone Eval Portal", version="0.1.0")
    _install_auth(app)

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
        # Same admission rule as the report: pre-fix and unverifiable records
        # must not appear here either, or the portal and the markdown disagree
        # about which cells exist (review 2026-10-22, second round).
        excluded = excluded_run_ids(records, paths.config_root.parent)
        live = [record for record in records if record.run_id not in excluded]
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

    @app.get("/api/tasks")
    async def tasks() -> dict:
        paths = RuntimePaths.from_environment()
        records, _ = repository.list_records()
        excluded = excluded_run_ids(records, paths.config_root.parent)
        live = [record for record in records if record.run_id not in excluded]
        matrix = build_matrix(live, model_names(paths.config_root))
        grouped = _cells_by_task(matrix)
        specs = _task_specs(paths.config_root)
        ordered = list(specs) + sorted(set(grouped) - set(specs))
        items = []
        for task_id in ordered:
            spec = specs.get(task_id)
            cells = grouped.get(task_id, [])
            best = _pick_best(cells)
            non_default = sum(1 for cell in cells if not _is_default_condition(cell))
            items.append(
                {
                    "task_id": task_id,
                    "name": spec.application if spec else task_id,
                    "application": spec.application if spec else None,
                    "dataset_id": spec.dataset_id if spec else None,
                    "sensor": spec.sensor if spec else None,
                    "metric": spec.primary_metric.name if spec else None,
                    "suite": "out_of_suite" if task_id in OUT_OF_SUITE else "in_suite",
                    "cells": len(cells),
                    "models": len({cell["model"] for cell in cells}),
                    "seeds_max": max((cell["n"] for cell in cells), default=0),
                    "has_notes": bool(spec and spec.notes),
                    "notes_updated": (
                        spec.notes.updated.isoformat() if spec and spec.notes else None
                    ),
                    "notes_age_days": (
                        (date.today() - spec.notes.updated).days if spec and spec.notes else None
                    ),
                    "best": best,
                    "non_default_cells": non_default,
                }
            )
        return {
            "items": items,
            "tracks": [{"id": key, "name": value} for key, value in TRACK_NAMES.items()],
        }

    @app.get("/api/tasks/{task_id}")
    async def task_detail(task_id: str) -> dict:
        paths = RuntimePaths.from_environment()
        records, _ = repository.list_records()
        excluded = excluded_run_ids(records, paths.config_root.parent)
        live = [record for record in records if record.run_id not in excluded]
        matrix = build_matrix(live, model_names(paths.config_root))
        cells = _cells_by_task(matrix).get(task_id, [])
        specs = _task_specs(paths.config_root)
        spec = specs.get(task_id)
        if spec is None and not cells:
            raise HTTPException(status_code=404, detail="task not found")
        return _task_payload(
            task_id,
            spec,
            cells,
            records,
            excluded,
            paths.data_root,
            manifest_summary(paths.data_root),
            repository,
        )

    @app.get("/api/compare")
    async def compare() -> dict:
        paths = RuntimePaths.from_environment()
        records, _ = repository.list_records()
        excluded = excluded_run_ids(records, paths.config_root.parent)
        live = [record for record in records if record.run_id not in excluded]
        matrix = build_matrix(live, model_names(paths.config_root))
        grouped = _cells_by_task(matrix)
        specs = _task_specs(paths.config_root)
        rows = []
        for task_id in sorted(grouped):
            spec = specs.get(task_id)
            per_track: dict[str, dict] = {}
            for track in {cell["track"] for cell in grouped[task_id]}:
                picked = _pick_best([cell for cell in grouped[task_id] if cell["track"] == track])
                if picked is None:
                    continue
                per_track[track] = {
                    "model": picked["model"],
                    "mean": picked["mean"],
                    "sd": picked["sd"],
                    "n": picked["n"],
                    "direction": picked["direction"],
                }
            rows.append(
                {
                    "task_id": task_id,
                    "name": spec.application if spec else task_id,
                    "metric": spec.primary_metric.name if spec else None,
                    "suite": "out_of_suite" if task_id in OUT_OF_SUITE else "in_suite",
                    "tracks": per_track,
                }
            )
        return {
            "tasks": rows,
            "tracks": [{"id": key, "name": value} for key, value in TRACK_NAMES.items()],
        }

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
