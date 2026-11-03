#!/usr/bin/env python3
"""Build the T11 PASTIS-HD manifest from the official metadata.

Split (declared, not tuned): Fold 1-3 train, Fold 4 validation, Fold 5 test.
Label year is 2019 for the technical validation; the LPIS year still needs the
release documentation to confirm it.

    uv run --with numpy python scripts/t11_build_manifest.py \
        --data-root /home/yzhong/rs-eval/data-aef

Writes ``<data-root>/manifests/T11/pastis_hd_2019.jsonl``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

AUDIT = Path("/geodata/rs-eval/data/derived/aef-audit/PASTIS-HD/metadata.geojson")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/home/yzhong/rs-eval/data-aef"))
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    root = args.data_root
    pastis = root / "PASTIS-HD"
    meta = json.loads(AUDIT.read_text(encoding="utf-8"))
    out = args.out or root / "manifests/T11" / f"pastis_hd_{args.year}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for feature in meta["features"]:
        props = feature["properties"]
        pid = int(props["ID_PATCH"])
        fold = int(props["Fold"])
        split = "train" if fold <= 3 else ("validation" if fold == 4 else "test")
        spot = next((pastis / "DATA_SPOT/PASTIS_SPOT6_RVB_1M00_2019").glob(f"*_{pid}.tif"))
        rows.append(
            {
                "task_id": "T11",
                "sample_id": str(pid),
                "group_id": str(pid),
                "split": split,
                "fold": fold,
                "label_year": args.year,
                "spot_path": str(spot.relative_to(root)),
                "s2_path": f"PASTIS-HD/DATA_S2/S2_{pid}.npy",
                "label_path": f"PASTIS-HD/ANNOTATIONS/TARGET_{pid}.npy",
                "parcel_path": f"PASTIS-HD/ANNOTATIONS/ParcelIDs_{pid}.npy",
                "aef_path": f"features/{args.year}/{pid}.npz",
                "window": [0, 0, 1280, 1280],
                "aef_grid": [128, 128],
                "sensor": "spot-6/7+s2",
                "bands": ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
                "timestamps": sorted(int(v) for v in props["dates-S2"].values()),
                "valid_fraction": 1.0,
            }
        )
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(out, len(rows), Counter(row["split"] for row in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
