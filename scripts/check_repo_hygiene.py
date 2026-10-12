"""Fail CI when large payloads or known model/data files enter the source tree."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
BLOCKED_SUFFIXES = {
    ".arrow",
    ".bin",
    ".ckpt",
    ".dbf",
    ".geojson",
    ".gpkg",
    ".grib",
    ".gz",
    ".h5",
    ".hdf5",
    ".jp2",
    ".nc",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pt",
    ".pth",
    ".safetensors",
    ".tif",
    ".tiff",
    ".tar",
    ".vrt",
    ".zip",
    ".7z",
}
SKIP_PARTS = {".git", ".venv", "node_modules", "__pycache__"}


def main() -> int:
    violations: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.parts):
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            violations.append(f"blocked payload type: {relative}")
        elif path.stat().st_size > MAX_BYTES:
            violations.append(f"file exceeds 5 MiB: {relative}")

    if violations:
        print("Repository hygiene check failed:")
        for violation in violations:
            print(f"- {violation}")
        return 1

    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
