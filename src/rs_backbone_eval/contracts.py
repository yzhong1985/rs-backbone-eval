"""Stable records exchanged by trainers, evaluators, and the result portal."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RunStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
MetricDirection = Literal["maximize", "minimize"]
WorkerAction = Literal["probe", "extract_features", "train", "evaluate", "export_previews"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetRecord(StrictModel):
    dataset_id: str
    version: str | None = None
    manifest_sha256: str
    split: str


class ManifestSample(StrictModel):
    task_id: str
    sample_id: str
    group_id: str
    split: str
    image_paths: list[str] = Field(min_length=1)
    label_path: str
    sensor: str
    bands: list[str] = Field(min_length=3)
    timestamps: list[str | int]
    temporal_mask: list[bool]
    crs: str | None = None
    transform: list[float] | None = None
    gsd_m: float = Field(gt=0)
    checksum: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @model_validator(mode="after")
    def temporal_lengths_must_match(self) -> ManifestSample:
        count = len(self.image_paths)
        if len(self.timestamps) != count or len(self.temporal_mask) != count:
            raise ValueError("image_paths, timestamps, and temporal_mask must have equal lengths")
        return self


class InputRecord(StrictModel):
    sensor: str
    bands: list[str]
    time_steps: int = Field(ge=1)
    gsd_m: float | None = Field(default=None, gt=0)
    preprocessing_id: str


class TrainingRecord(StrictModel):
    regime: Literal["reference", "full_finetune", "frozen_probe", "native_input"]
    head_id: str
    trainable_parameters: int = Field(ge=0)
    total_parameters: int = Field(ge=0)
    epochs_completed: int = Field(default=0, ge=0)
    optimizer_steps: int = Field(default=0, ge=0)
    wall_time_seconds: float | None = Field(default=None, ge=0)
    peak_gpu_memory_mb: float | None = Field(default=None, ge=0)


class MetricRecord(StrictModel):
    name: str
    value: float
    split: str = "test"
    direction: MetricDirection
    unit: str | None = None
    class_name: str | None = None
    is_primary: bool = False


class CurvePoint(StrictModel):
    name: str
    split: str
    step: int = Field(ge=0)
    value: float


class ArtifactRecord(StrictModel):
    kind: Literal[
        "checkpoint",
        "config",
        "confusion_matrix",
        "log",
        "prediction",
        "preview",
        "report",
        "table",
    ]
    path: str
    title: str | None = None
    media_type: str | None = None

    @field_validator("path")
    @classmethod
    def path_must_be_relative(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact path must stay inside its run directory")
        return value


class RunRecord(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: RunStatus
    task_id: str
    model_id: str
    protocol_id: str
    seed: int
    git_commit: str | None = None
    model_checkpoint_sha256: str | None = None
    dataset: DatasetRecord
    input: InputRecord
    training: TrainingRecord
    metrics: list[MetricRecord] = Field(default_factory=list)
    curves: list[CurvePoint] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    tags: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None

    def primary_metric(self) -> MetricRecord | None:
        return next((metric for metric in self.metrics if metric.is_primary), None)

    def public_summary(self) -> dict[str, Any]:
        primary = self.primary_metric()
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "task_id": self.task_id,
            "model_id": self.model_id,
            "protocol_id": self.protocol_id,
            "seed": self.seed,
            "primary_metric": primary.model_dump(mode="json") if primary else None,
            "training": self.training.model_dump(mode="json"),
            "input": self.input.model_dump(mode="json"),
            "tags": self.tags,
        }


class WorkerRequest(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    action: WorkerAction
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
    task_config_path: str
    model_config_path: str
    protocol_config_path: str
    manifest_path: str
    data_root: str
    weights_root: str
    cache_root: str
    runs_root: str
    seed: int


class WorkerEvent(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    level: Literal["debug", "info", "warning", "error"]
    event: str
    message: str
    progress: float | None = Field(default=None, ge=0, le=1)
