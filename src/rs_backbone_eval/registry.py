"""Load and validate task, model, and protocol registry files."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RegistryModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricSpec(RegistryModel):
    name: str
    direction: Literal["maximize", "minimize"]


class TaskSpec(RegistryModel):
    task_id: str
    application: str
    dataset_id: str
    family: Literal["segmentation", "classification", "regression"]
    sensor: str
    time_steps: int = Field(ge=1)
    bands: list[str] = Field(min_length=3)
    num_outputs: int = Field(ge=1)
    ignore_index: int | None = None
    primary_metric: MetricSpec
    official_config_url: str


class ModelSpec(RegistryModel):
    model_id: str
    display_name: str
    worker_id: str
    checkpoint_env: str
    upstream_url: str
    enabled: bool = True


class ProtocolSpec(RegistryModel):
    protocol_id: str
    regime: Literal["reference", "full_finetune", "frozen_probe", "native_input"]
    train_backbone: bool
    train_head: bool
    train_input_adapter: bool
    seeds: list[int]


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

    def tasks(self) -> list[TaskSpec]:
        return list(self._load_group("tasks", TaskSpec))

    def models(self) -> list[ModelSpec]:
        return list(self._load_group("models", ModelSpec))

    def protocols(self) -> list[ProtocolSpec]:
        return list(self._load_group("protocols", ProtocolSpec))
