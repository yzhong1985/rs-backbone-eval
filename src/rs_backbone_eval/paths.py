"""External payload paths. No model or dataset path is stored in source control."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _path_from_env(name: str, fallback: str) -> Path:
    return Path(os.getenv(name, fallback)).expanduser().resolve()


@dataclass(frozen=True)
class RuntimePaths:
    data_root: Path
    weights_root: Path
    runs_root: Path
    cache_root: Path
    config_root: Path

    @classmethod
    def from_environment(cls) -> RuntimePaths:
        load_dotenv()
        return cls(
            data_root=_path_from_env("RS_EVAL_DATA_ROOT", "data"),
            weights_root=_path_from_env("RS_EVAL_WEIGHTS_ROOT", "weights"),
            runs_root=_path_from_env("RS_EVAL_RUNS_ROOT", "runs"),
            cache_root=_path_from_env("RS_EVAL_CACHE_ROOT", "cache"),
            config_root=_path_from_env("RS_EVAL_CONFIG_ROOT", "configs"),
        )
