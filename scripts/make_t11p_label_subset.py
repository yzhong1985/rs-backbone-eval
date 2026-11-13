#!/usr/bin/env python3
"""Build the T11P 10% label-fraction manifest from the T11D 10% subset.

The T11D few-shot panel uses a fixed 146-patch training subset
(``T11D_subsets/10pct-groups.jsonl``). The T11P control reuses exactly those
sample ids, keeps the full validation/test folds, and writes a new manifest so
the main T11P manifest is never modified.

Usage::

    uv run python scripts/make_t11p_label_subset.py \
        --source /geodata/rs-eval/data/manifests/T11P/<sha>.jsonl \
        --ids-from /geodata/rs-eval/data/manifests/T11D_subsets/10pct-groups.jsonl \
        --out /geodata/rs-eval/data/manifests/T11P_subsets/10pct-groups-from-T11D.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_subset(source: Path, ids_from: Path, out: Path) -> dict[str, object]:
    ids_rows = read_rows(ids_from)
    train_ids = {row["sample_id"] for row in ids_rows if row.get("split") == "train"}
    if not train_ids:
        raise SystemExit(f"{ids_from} has no split=train rows")

    rows = read_rows(source)
    train_rows = [row for row in rows if row.get("split") == "train"]
    available = {row["sample_id"] for row in train_rows}
    missing = train_ids - available
    if missing:
        raise SystemExit(
            f"{len(missing)} ids from {ids_from} are missing in {source}: {sorted(missing)[:5]}"
        )

    kept = [row for row in rows if row.get("split") != "train" or row["sample_id"] in train_ids]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for row in kept:
        split = str(row.get("split"))
        counts[split] = counts.get(split, 0) + 1
    return {
        "source": str(source),
        "source_sha256": sha256(source),
        "ids_from": str(ids_from),
        "ids_sha256": sha256(ids_from),
        "output": str(out),
        "output_sha256": sha256(out),
        "train_ids": len(train_ids),
        "counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ids-from", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_subset(args.source, args.ids_from, args.out)
    for key in (
        "source_sha256",
        "ids_sha256",
        "output_sha256",
        "train_ids",
        "counts",
    ):
        print(f"{key}: {report[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
