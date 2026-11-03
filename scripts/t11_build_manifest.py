#!/usr/bin/env python3
"""Build the T11 PASTIS-HD manifest from the official metadata.

Split (declared, not tuned): Fold 1-3 train, Fold 4 validation, Fold 5 test.
Label year is 2019 for the technical validation; the LPIS year still needs the
release documentation to confirm it.

Field conventions (review 2026-11-03, #4/5):

- ``sensor`` / ``bands`` describe the *source* raster modalities the sample
  carries (SPOT 1 m RGB + Sentinel-2 10-band time series), not the model input;
  the model input for the AEF track is the precomputed grid in ``aef_path``
  (64 bands, 10 m), declared by ``aef_encoding`` / ``aef_bands`` / ``aef_grid``.
  The registry registration adds ``input_kind`` so the contract audits this
  rather than trusting prose;
- ``encoding`` field name follows the extraction sidecar's ``quantization``
  (raw AEF int8 -> float16, NODATA -128); a worker must apply
  ``sign(x) * (x/127.5)**2`` and mask by the npz ``valid`` raster.

The official metadata path comes from ``RS_EVAL_AEF_AUDIT_META`` (default
``/geodata/rs-eval/data/derived/aef-audit/PASTIS-HD/metadata.geojson``); the
record paths under ``--data-root``.

    uv run --with numpy python scripts/t11_build_manifest.py \
        --data-root /home/yzhong/rs-eval/data-aef

Writes ``<data-root>/manifests/T11/pastis_hd_<year>.jsonl`` via tmp-file + rename
and prints its sha256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

AUDIT = Path(
    os.environ.get(
        "RS_EVAL_AEF_AUDIT_META",
        "/geodata/rs-eval/data/derived/aef-audit/PASTIS-HD/metadata.geojson",
    )
)
AEF_ENCODING = "raw_aef_int8_as_float16"


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
                "tile": props["TILE"],
                "label_year": args.year,
                "spot_path": str(spot.relative_to(root)),
                "s2_path": f"PASTIS-HD/DATA_S2/S2_{pid}.npy",
                "label_path": f"PASTIS-HD/ANNOTATIONS/TARGET_{pid}.npy",
                "parcel_path": f"PASTIS-HD/ANNOTATIONS/ParcelIDs_{pid}.npy",
                "aef_path": f"features/{args.year}/{pid}.npz",
                "aef_encoding": AEF_ENCODING,
                "aef_bands": 64,
                "aef_grid": [128, 128],
                "window": [0, 0, 1280, 1280],
                "sensor": "spot-6/7+s2",
                # Source gridded modalities; see the module docstring. RGB is
                # the SPOT mosaic, the six S2 bands listed are the Prithvi/HLS
                # subset the B0-family workers consume.
                "bands": ["BLUE", "GREEN", "RED", "NIR", "SWIR1", "SWIR2"],
                "timestamps": sorted(int(v) for v in props["dates-S2"].values()),
                "valid_fraction": 1.0,
            }
        )
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    digest = hashlib.sha256(payload.encode(encoding="utf-8")).hexdigest()
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, out)
    print(out, len(rows), dict(Counter(row["split"] for row in rows)), f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
