"""Catalog-driven downloads for external datasets and model weights."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DownloadKind = Literal["dataset", "model"]
DownloadMethod = Literal["http_files", "hf_snapshot", "gcs_rsync"]
DownloadRoot = Literal["data", "weights"]


class DownloadError(RuntimeError):
    """Base class for download control errors."""


class DownloadNotFoundError(DownloadError):
    """Raised when a resource ID does not exist in the catalog."""


class DownloadConflictError(DownloadError):
    """Raised when an action conflicts with the current resource state."""


class DownloadUnavailableError(DownloadError):
    """Raised when a required local download tool is missing."""


class CatalogModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _relative_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError("path must be a non-empty relative path without '..'")
    return path.as_posix()


class DownloadFileSpec(CatalogModel):
    url: str = Field(pattern=r"^https://")
    filename: str
    size_bytes: int | None = Field(default=None, gt=0)
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        return _relative_path(value)


class DownloadResourceSpec(CatalogModel):
    resource_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    kind: DownloadKind
    display_name: str
    description: str
    task_ids: list[str] = Field(default_factory=list)
    license: str
    source_page: str = Field(pattern=r"^https://")
    root: DownloadRoot
    target_dir: str
    method: DownloadMethod
    expected_size_bytes: int | None = Field(default=None, gt=0)
    size_is_estimate: bool = False
    files: list[DownloadFileSpec] = Field(default_factory=list)
    completion_paths: list[str] = Field(default_factory=list)
    repo_id: str | None = None
    repo_type: Literal["dataset", "model"] | None = None
    revision: str | None = None
    include: list[str] = Field(default_factory=list)
    gcs_uri: str | None = None
    notes: str | None = None

    @field_validator("target_dir")
    @classmethod
    def validate_target_dir(cls, value: str) -> str:
        return _relative_path(value)

    @field_validator("completion_paths")
    @classmethod
    def validate_completion_paths(cls, values: list[str]) -> list[str]:
        return [_relative_path(value) for value in values]

    @model_validator(mode="after")
    def validate_method_fields(self) -> DownloadResourceSpec:
        if self.method == "http_files" and not self.files:
            raise ValueError("http_files resources require files")
        if self.method == "hf_snapshot" and not (self.repo_id and self.repo_type and self.revision):
            raise ValueError("hf_snapshot resources require repo_id, repo_type, and revision")
        if self.method == "gcs_rsync" and not self.gcs_uri:
            raise ValueError("gcs_rsync resources require gcs_uri")
        return self


class DownloadCatalog(CatalogModel):
    schema_version: Literal["1.0"] = "1.0"
    resources: list[DownloadResourceSpec]

    @model_validator(mode="after")
    def unique_resource_ids(self) -> DownloadCatalog:
        ids = [resource.resource_id for resource in self.resources]
        if len(ids) != len(set(ids)):
            raise ValueError("download resource IDs must be unique")
        return self

    @classmethod
    def load(cls, path: Path) -> DownloadCatalog:
        with path.open("r", encoding="utf-8") as handle:
            return cls.model_validate(yaml.safe_load(handle))


@dataclass
class _Job:
    process: subprocess.Popen[bytes] | None
    state: str
    engine: str
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    bytes_downloaded: int = 0
    scan_at: float = field(default_factory=time.time)
    speed_bps: float = 0
    pause_requested: bool = False
    error: str | None = None


class DownloadManager:
    """Run allow-listed downloads and expose filesystem-derived progress."""

    active_states = {"queued", "downloading", "verifying"}

    def __init__(
        self,
        catalog_path: Path,
        data_root: Path,
        weights_root: Path,
        cache_root: Path,
        *,
        enabled: bool = True,
    ) -> None:
        self.catalog_path = catalog_path.resolve()
        self.catalog = DownloadCatalog.load(self.catalog_path)
        self.resources = {resource.resource_id: resource for resource in self.catalog.resources}
        self.roots = {"data": data_root.resolve(), "weights": weights_root.resolve()}
        self.state_root = cache_root.resolve() / "downloads"
        self.enabled = enabled
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.RLock()

    def _resource(self, resource_id: str) -> DownloadResourceSpec:
        try:
            return self.resources[resource_id]
        except KeyError as exc:
            raise DownloadNotFoundError(resource_id) from exc

    def _target(self, resource: DownloadResourceSpec) -> Path:
        root = self.roots[resource.root]
        target = (root / resource.target_dir).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"download target escapes {resource.root} root")
        return target

    def _marker(self, resource: DownloadResourceSpec) -> Path:
        return self._target(resource) / ".rs-download-complete.json"

    def _completion_paths_exist(self, resource: DownloadResourceSpec) -> bool:
        target = self._target(resource)
        return bool(resource.completion_paths) and all(
            (target / relative).exists() for relative in resource.completion_paths
        )

    def _scan_bytes(self, resource: DownloadResourceSpec) -> int:
        target = self._target(resource)
        if not target.exists():
            return 0
        if resource.method == "http_files":
            total = 0
            for item in resource.files:
                path = target / item.filename
                if path.is_file():
                    size = path.stat().st_size
                    total += min(size, item.size_bytes) if item.size_bytes else size
            return total
        total = 0
        for path in target.rglob("*"):
            if not path.is_file() or ".cache" in path.parts:
                continue
            if path.name == ".rs-download-complete.json" or path.suffix == ".aria2":
                continue
            total += path.stat().st_size
        return total

    def _engine(self, resource: DownloadResourceSpec) -> tuple[str | None, str | None]:
        if resource.method == "http_files":
            executable = shutil.which("aria2c")
            return ("aria2c", executable) if executable else ("python-http", sys.executable)
        if resource.method == "hf_snapshot":
            executable = shutil.which("hf") or shutil.which("huggingface-cli")
            return (Path(executable).name, executable) if executable else (None, None)
        executable = shutil.which("gsutil") or shutil.which("gcloud")
        return (Path(executable).name, executable) if executable else (None, None)

    def _inferred_state(self, resource: DownloadResourceSpec, downloaded: int) -> str:
        if self._marker(resource).is_file() and self._completion_paths_exist(resource):
            return "complete"
        if self._completion_paths_exist(resource):
            return "present"
        return "partial" if downloaded else "missing"

    @staticmethod
    def _disk_free(path: Path) -> int:
        candidate = path
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return shutil.disk_usage(candidate).free

    def status(self, resource_id: str) -> dict:
        resource = self._resource(resource_id)
        with self._lock:
            downloaded = self._scan_bytes(resource)
            job = self._jobs.get(resource_id)
            engine, executable = self._engine(resource)
            state = self._inferred_state(resource, downloaded)
            error = None
            speed = 0.0
            if job is not None:
                state = job.state
                error = job.error
                engine = job.engine
                now = time.time()
                elapsed = now - job.scan_at
                if job.state == "downloading" and elapsed >= 0.5:
                    instantaneous = max(0, downloaded - job.bytes_downloaded) / elapsed
                    job.speed_bps = (
                        instantaneous
                        if not job.speed_bps
                        else (job.speed_bps * 0.65 + instantaneous * 0.35)
                    )
                    job.bytes_downloaded = downloaded
                    job.scan_at = now
                    job.updated_at = now
                speed = job.speed_bps if job.state == "downloading" else 0.0
            total = resource.expected_size_bytes
            progress = min(downloaded / total, 1.0) if total else None
            eta = (
                (total - downloaded) / speed if total and speed > 0 and downloaded < total else None
            )
            available = executable is not None
            disk_free = self._disk_free(self._target(resource))
            remaining = max(0, total - downloaded) if total else None
            space_sufficient = remaining is None or disk_free >= remaining
            if not available and state not in {"complete", "present"}:
                error = error or self._missing_tool_message(resource)
                if job is None:
                    state = "unavailable"
            elif not space_sufficient and state not in self.active_states and state != "complete":
                error = error or "目标磁盘空间不足，请更换外置根目录或释放空间。"
            target = self._target(resource)
            return {
                "resource_id": resource.resource_id,
                "kind": resource.kind,
                "display_name": resource.display_name,
                "description": resource.description,
                "task_ids": resource.task_ids,
                "license": resource.license,
                "source_page": resource.source_page,
                "method": resource.method,
                "engine": engine,
                "state": state,
                "enabled": self.enabled,
                "available": available,
                "can_start": self.enabled
                and available
                and space_sufficient
                and state not in self.active_states
                and state != "complete",
                "can_pause": self.enabled and state == "downloading",
                "can_verify": self.enabled and state == "present",
                "bytes_downloaded": downloaded,
                "bytes_total": total,
                "size_is_estimate": resource.size_is_estimate,
                "progress": progress,
                "speed_bps": speed,
                "eta_seconds": eta,
                "disk_free_bytes": disk_free,
                "space_sufficient": space_sufficient,
                "destination": str(target),
                "error": error,
                "notes": resource.notes,
            }

    def summary(self) -> dict:
        items = [self.status(resource.resource_id) for resource in self.catalog.resources]
        counts: dict[str, int] = {}
        for item in items:
            counts[item["state"]] = counts.get(item["state"], 0) + 1
        return {
            "enabled": self.enabled,
            "catalog_version": self.catalog.schema_version,
            "items": items,
            "state_counts": counts,
            "resource_count": len(items),
            "expected_size_bytes": sum(item["bytes_total"] or 0 for item in items),
            "downloaded_bytes": sum(item["bytes_downloaded"] for item in items),
        }

    def _missing_tool_message(self, resource: DownloadResourceSpec) -> str:
        if resource.method == "hf_snapshot":
            return "缺少 hf CLI；安装项目 downloads extra 后重启 Portal。"
        if resource.method == "gcs_rsync":
            return "缺少 gsutil/gcloud；安装 Google Cloud CLI 后重启 Portal。"
        return "没有可用下载引擎。"

    def _write_aria2_input(self, resource: DownloadResourceSpec, target: Path) -> Path:
        self.state_root.mkdir(parents=True, exist_ok=True)
        path = self.state_root / f"{resource.resource_id}.aria2.txt"
        lines: list[str] = []
        for item in resource.files:
            lines.extend([item.url, f"  dir={target}", f"  out={item.filename}"])
            if item.sha256:
                lines.append(f"  checksum=sha-256={item.sha256}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _write_python_spec(self, resource: DownloadResourceSpec, target: Path) -> Path:
        self.state_root.mkdir(parents=True, exist_ok=True)
        path = self.state_root / f"{resource.resource_id}.http.json"
        payload = {
            "target": str(target),
            "files": [item.model_dump(mode="json") for item in resource.files],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _command(
        self, resource: DownloadResourceSpec, target: Path, engine: str, executable: str
    ) -> list[str]:
        if resource.method == "http_files" and engine == "aria2c":
            input_path = self._write_aria2_input(resource, target)
            return [
                executable,
                f"--input-file={input_path}",
                "--continue=true",
                "--allow-overwrite=true",
                "--auto-file-renaming=false",
                "--file-allocation=none",
                "--max-concurrent-downloads=3",
                "--max-connection-per-server=8",
                "--split=8",
                "--min-split-size=16M",
                "--check-integrity=true",
                "--summary-interval=1",
                "--console-log-level=warn",
                "--download-result=hide",
            ]
        if resource.method == "http_files":
            spec_path = self._write_python_spec(resource, target)
            return [executable, "-m", "rs_backbone_eval.download_worker", str(spec_path)]
        if resource.method == "hf_snapshot":
            if engine == "hf":
                command = [executable, "download", resource.repo_id or ""]
            else:
                command = [executable, "download", resource.repo_id or ""]
            command.extend(
                [
                    "--repo-type",
                    resource.repo_type or resource.kind,
                    "--revision",
                    resource.revision or "main",
                    "--local-dir",
                    str(target),
                ]
            )
            for pattern in resource.include:
                command.extend(["--include", pattern])
            return command
        if engine == "gsutil":
            return [executable, "-m", "rsync", "-r", resource.gcs_uri or "", str(target)]
        return [
            executable,
            "storage",
            "rsync",
            "--recursive",
            resource.gcs_uri or "",
            str(target),
        ]

    def start(self, resource_id: str) -> dict:
        resource = self._resource(resource_id)
        with self._lock:
            if not self.enabled:
                raise DownloadUnavailableError("Portal download control is disabled")
            current = self._jobs.get(resource_id)
            if current and current.state in self.active_states:
                raise DownloadConflictError("resource already has an active operation")
            if self.status(resource_id)["state"] == "complete":
                raise DownloadConflictError("resource is already complete")
            if not self.status(resource_id)["space_sufficient"]:
                raise DownloadUnavailableError("目标磁盘空间不足，请更换外置根目录或释放空间。")
            engine, executable = self._engine(resource)
            if engine is None or executable is None:
                raise DownloadUnavailableError(self._missing_tool_message(resource))
            target = self._target(resource)
            target.mkdir(parents=True, exist_ok=True)
            self.state_root.mkdir(parents=True, exist_ok=True)
            log_path = self.state_root / f"{resource_id}.log"
            command = self._command(resource, target, engine, executable)
            with log_path.open("ab") as log:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt",
                )
            downloaded = self._scan_bytes(resource)
            job = _Job(
                process=process,
                state="downloading",
                engine=engine,
                bytes_downloaded=downloaded,
            )
            self._jobs[resource_id] = job
            thread = threading.Thread(
                target=self._monitor,
                args=(resource_id, process, log_path),
                name=f"download-{resource_id}",
                daemon=True,
            )
            thread.start()
        return self.status(resource_id)

    def _monitor(self, resource_id: str, process: subprocess.Popen[bytes], log_path: Path) -> None:
        return_code = process.wait()
        resource = self._resource(resource_id)
        with self._lock:
            job = self._jobs.get(resource_id)
            if job is None or job.process is not process:
                return
            job.updated_at = time.time()
            if job.pause_requested:
                job.state = "paused"
                return
            if return_code != 0:
                job.state = "failed"
                job.error = self._log_tail(log_path) or f"download exited with code {return_code}"
                return
            job.state = "verifying"
        try:
            self._verify_resource(resource)
        except (OSError, ValueError) as exc:
            with self._lock:
                job = self._jobs[resource_id]
                job.state = "failed"
                job.error = str(exc)
        else:
            with self._lock:
                job = self._jobs[resource_id]
                job.state = "complete"
                job.speed_bps = 0
                job.error = None

    @staticmethod
    def _log_tail(path: Path, limit: int = 12) -> str:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-limit:])[-2000:]

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def _verify_resource(self, resource: DownloadResourceSpec) -> None:
        target = self._target(resource)
        if not self._completion_paths_exist(resource):
            raise ValueError("download finished but required paths are missing")
        if resource.method == "http_files" and not resource.size_is_estimate:
            for item in resource.files:
                path = target / item.filename
                if item.size_bytes and path.stat().st_size != item.size_bytes:
                    raise ValueError(f"size mismatch: {item.filename}")
                if item.sha256 and self._sha256(path).lower() != item.sha256.lower():
                    raise ValueError(f"SHA-256 mismatch: {item.filename}")
        marker = {
            "schema_version": "1.0",
            "resource_id": resource.resource_id,
            "completed_at": datetime.now(UTC).isoformat(),
            "catalog": str(self.catalog_path),
            "bytes_on_disk": self._scan_bytes(resource),
        }
        marker_path = self._marker(resource)
        temporary = marker_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        temporary.replace(marker_path)

    def verify(self, resource_id: str) -> dict:
        resource = self._resource(resource_id)
        with self._lock:
            if not self.enabled:
                raise DownloadUnavailableError("Portal download control is disabled")
            current = self._jobs.get(resource_id)
            if current and current.state in self.active_states:
                raise DownloadConflictError("resource already has an active operation")
            if not self._completion_paths_exist(resource):
                raise DownloadConflictError("required paths are not all present")
            job = _Job(process=None, state="verifying", engine="local-verifier")
            self._jobs[resource_id] = job
            thread = threading.Thread(
                target=self._verify_existing,
                args=(resource_id,),
                name=f"verify-{resource_id}",
                daemon=True,
            )
            thread.start()
        return self.status(resource_id)

    def _verify_existing(self, resource_id: str) -> None:
        try:
            self._verify_resource(self._resource(resource_id))
        except (OSError, ValueError) as exc:
            with self._lock:
                job = self._jobs[resource_id]
                job.state = "failed"
                job.error = str(exc)
        else:
            with self._lock:
                job = self._jobs[resource_id]
                job.state = "complete"
                job.error = None

    @staticmethod
    def _stop_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            pass

    def pause(self, resource_id: str) -> dict:
        self._resource(resource_id)
        with self._lock:
            job = self._jobs.get(resource_id)
            if job is None or job.state != "downloading" or job.process is None:
                raise DownloadConflictError("resource is not downloading")
            job.pause_requested = True
            job.state = "paused"
            self._stop_process(job.process)
        return self.status(resource_id)

    def shutdown(self) -> None:
        with self._lock:
            active = [
                job
                for job in self._jobs.values()
                if job.state == "downloading" and job.process is not None
            ]
            for job in active:
                job.pause_requested = True
                job.state = "paused"
                self._stop_process(job.process)
