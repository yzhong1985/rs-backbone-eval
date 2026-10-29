"""Load and validate task, model, and protocol registry files."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RegistryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricSpec(RegistryModel):
    name: str
    direction: Literal["maximize", "minimize"]


DEFAULT_ALLOWED_SPLITS = (
    "train",
    "validation",
    "test",
    "official_validation",
    "geographic_holdout",
    "temporal_holdout",
)
DEFAULT_REQUIRED_SPLITS = ("train",)


def load_yaml_spec(path: Path, model_type: type[RegistryModel]) -> RegistryModel:
    """Load and validate a single registry spec from its YAML file."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"registry file must contain a mapping: {path}")
    return model_type.model_validate(payload)


class TaskNotes(RegistryModel):
    """Human-written conclusion shown on the task's portal page.

    Kept next to the task definition so the prose and the configuration move in
    the same review, and so `updated` travels with the claim it dates.
    """

    updated: date
    summary: str
    findings: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class TaskSpec(RegistryModel):
    task_id: str
    application: str
    dataset_id: str
    # "detection" carries rotated boxes instead of a raster label; the
    # discriminator is ManifestSample.annotation_path (FAIR1M review 2026-10-29).
    family: Literal["segmentation", "classification", "regression", "detection"]
    sensor: str
    time_steps: int = Field(ge=1)
    bands: list[str] = Field(min_length=2)
    num_outputs: int = Field(ge=1)
    ignore_index: int | None = None
    # Declared per task, never inferred from the index: T5's class 0 is a real
    # crop class and T8's class 0 is the ignore class, so "exclude class 0"
    # would be wrong there (review 2026-10-27). None = no background class.
    background_class: int | None = None
    primary_metric: MetricSpec
    official_config_url: str
    allowed_splits: list[str] = Field(default_factory=lambda: list(DEFAULT_ALLOWED_SPLITS))
    required_splits: list[str] = Field(default_factory=lambda: list(DEFAULT_REQUIRED_SPLITS))
    notes: TaskNotes | None = None


class ModelSpec(RegistryModel):
    model_id: str
    display_name: str
    worker_id: str
    checkpoint_env: str
    upstream_url: str
    enabled: bool = True
    worker_command: list[str] | None = None


class ProtocolSpec(RegistryModel):
    protocol_id: str
    regime: Literal["reference", "full_finetune", "frozen_probe", "native_input"]
    train_backbone: bool
    train_head: bool
    train_input_adapter: bool
    seeds: list[int]
    # Workers that actually implement this protocol. A protocol with stage or
    # cache semantics must name them: otherwise a worker that ignores the
    # semantics still records the run under this protocol id and the cell is
    # mislabelled (review P1: two-stage accepted by every model).
    workers: list[str] | None = None
    # Tasks whose worker code implements this protocol. The two-stage
    # implementation lives in the AnySat T5 worker only: AnySat itself ignores
    # the stage semantics on T1-T4, so the worker allowlist alone still lets a
    # T3 run be recorded as B2S (review P1, second round).
    tasks: list[str] | None = None


class Registry:
    def __init__(self, config_root: Path) -> None:
        self.config_root = config_root

    @staticmethod
    def _read_yaml(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"registry file must contain a mapping: {path}")
        return payload

    def _load_group(self, group: str, model_type: type[RegistryModel]) -> list[RegistryModel]:
        directory = self.config_root / group
        if not directory.is_dir():
            raise FileNotFoundError(f"registry directory not found: {directory}")
        return [
            model_type.model_validate(self._read_yaml(path))
            for path in sorted(directory.glob("*.yaml"))
        ]

    def _file_map(
        self, group: str, model_type: type[RegistryModel], id_field: str
    ) -> dict[str, Path]:
        directory = self.config_root / group
        if not directory.is_dir():
            raise FileNotFoundError(f"registry directory not found: {directory}")
        mapping: dict[str, Path] = {}
        for path in sorted(directory.glob("*.yaml")):
            spec = model_type.model_validate(self._read_yaml(path))
            mapping[getattr(spec, id_field)] = path
        return mapping

    def tasks(self) -> list[TaskSpec]:
        return list(self._load_group("tasks", TaskSpec))

    def models(self) -> list[ModelSpec]:
        return list(self._load_group("models", ModelSpec))

    def protocols(self) -> list[ProtocolSpec]:
        return list(self._load_group("protocols", ProtocolSpec))

    def task_paths(self) -> dict[str, Path]:
        return self._file_map("tasks", TaskSpec, "task_id")

    def model_paths(self) -> dict[str, Path]:
        return self._file_map("models", ModelSpec, "model_id")

    def protocol_paths(self) -> dict[str, Path]:
        return self._file_map("protocols", ProtocolSpec, "protocol_id")
