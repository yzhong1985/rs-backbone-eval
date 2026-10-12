"""Small resumable HTTP fallback used when aria2c is unavailable."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from urllib.request import Request, urlopen


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, size_bytes: int | None, sha256: str | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    if size_bytes and existing == size_bytes and (not sha256 or _sha256(destination) == sha256):
        return
    headers = {"User-Agent": "rs-backbone-eval/0.1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=60) as response:  # noqa: S310 - URLs are catalog allow-listed.
        resumed = response.status == 206 and existing > 0
        mode = "ab" if resumed else "wb"
        with destination.open(mode) as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    if size_bytes and destination.stat().st_size != size_bytes:
        raise RuntimeError(f"size mismatch for {destination.name}")
    if sha256 and _sha256(destination).lower() != sha256.lower():
        raise RuntimeError(f"SHA-256 mismatch for {destination.name}")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m rs_backbone_eval.download_worker SPEC.json")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    target = Path(payload["target"])
    for item in payload["files"]:
        download(
            item["url"],
            target / item["filename"],
            item.get("size_bytes"),
            item.get("sha256"),
        )


if __name__ == "__main__":
    main()
