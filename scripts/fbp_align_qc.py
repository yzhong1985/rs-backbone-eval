#!/usr/bin/env python3
"""Visual QC for the FBP 10 m alignment (GF-2 | labels | AEF | OlmoEarth).

Same layout as ``scripts/olmoearth_align_qc.py``: one row per scene and four
panels — orthorectified GF-2 RGB, GF-2 with the 10 m labels, AEF 2017 PCA and
OlmoEarth 2017 PCA. Because the labels and both embeddings share the 10 m grid,
a misalignment shows up as parcel edges cutting through the wrong PCA colour
patches.

The GF-2 panel is produced by warping the scene again on the window extent with
the same ``-tap`` grid and ``RPC_HEIGHT`` as the extraction, so what you see is
exactly what the features were sampled on.

Run in the Prithvi worker environment (rasterio + PIL, no GPU)::

    workers/prithvi/.venv/bin/python scripts/fbp_align_qc.py \\
        --cache /home/yzhong/rs-eval/rs-eval-cache/fbp2017 \\
        --out /home/yzhong/rs-eval/audits/fbp2017-qc
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw

FBP = Path("/geodata/Five-Billion-Pixels")
WINDOW = 512
FILL = -128
TILE = 320  # rendered panel size


def stretch(rgb: np.ndarray) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for band in range(rgb.shape[-1]):
        values = rgb[..., band]
        low, high = np.percentile(values, (2, 98))
        out[..., band] = np.clip((values - low) / max(high - low, 1e-6), 0, 1)
    return out


def pca_rgb(features: np.ndarray) -> np.ndarray:
    """Three leading principal components, per-window normalised."""
    valid = (features != FILL).all(axis=0)
    flat = features[:, valid].T.astype(np.float32)
    out = np.zeros((features.shape[1], features.shape[2], 3), dtype=np.float32)
    if len(flat) < 100:
        return out
    mean = flat.mean(axis=0)
    centred = flat - mean
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    components = centred @ vh[:3].T
    components /= components.std(axis=0) + 1e-6
    components = (components + 2.0) / 4.0
    tmp = np.zeros((valid.sum(), 3), dtype=np.float32)
    tmp[:] = np.clip(components, 0, 1)
    out[valid] = tmp
    return out


def gf2_rgb(scene: str, window_row: tuple) -> np.ndarray:
    """GF-2 RGB for one window: bands 3,2,1 after a 10 m RPC warp."""
    _, _, _, x_left, y_top = window_row
    size = WINDOW * 10
    work = Path("/tmp/opencode") / f"qc_{scene[:24]}.tif"
    if not work.exists():
        vrt = Path("/tmp/opencode") / f"qc_{scene[:24]}.vrt"
        subprocess.run(
            [
                "python3",
                "scripts/fbp_rpc_vrt.py",
                str(FBP / "Image_16bit_BGRNir" / f"{scene}.tiff"),
                str(FBP / "Coordinate_files" / f"{scene}.rpb"),
                str(vrt),
            ],
            check=True,
            capture_output=True,
        )
        height = _rpc_height(scene)
        subprocess.run(
            [
                "gdalwarp",
                "-q",
                "-overwrite",
                "-rpc",
                "-t_srs",
                _srs(scene),
                "-tr",
                "10",
                "10",
                "-tap",
                "-te",
                str(x_left),
                str(y_top - size),
                str(x_left + size),
                str(y_top),
                "-r",
                "bilinear",
                "-to",
                f"RPC_HEIGHT={height}",
                "-of",
                "GTiff",
                str(vrt),
                str(work),
            ],
            check=True,
            capture_output=True,
        )
    with rasterio.open(work) as dataset:
        rgb = dataset.read([3, 2, 1]).transpose(1, 2, 0) / 65535.0
    return stretch(rgb)


def _rpc_height(scene: str) -> float:
    text = (FBP / "Coordinate_files" / f"{scene}.rpb").read_text(errors="ignore")
    for line in text.splitlines():
        if "heightOffset" in line:
            return float(line.split("=")[1].strip().rstrip(";"))
    return 0.0


def _srs(scene: str) -> str:
    text = (FBP / "Coordinate_files" / f"{scene}.rpb").read_text(errors="ignore")
    lon = next(
        float(line.split("=")[1].strip().rstrip(";"))
        for line in text.splitlines()
        if "longOffset" in line
    )
    return f"EPSG:326{int((lon + 180) // 6) + 1:02d}"


def label_rgb(labels: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],
            [200, 0, 0],
            [0, 200, 0],
            [150, 250, 0],
            [150, 200, 150],
            [200, 0, 200],
            [150, 0, 250],
            [150, 150, 250],
            [200, 150, 200],
            [250, 200, 0],
            [200, 200, 0],
            [0, 0, 200],
            [250, 0, 150],
            [0, 150, 200],
            [0, 200, 250],
            [150, 200, 250],
            [250, 250, 250],
            [200, 200, 200],
            [200, 150, 150],
            [250, 200, 150],
            [150, 150, 0],
            [250, 150, 150],
            [250, 150, 0],
            [250, 200, 250],
            [200, 150, 0],
        ],
        dtype=np.float32,
    )
    return palette[np.clip(labels, 0, 24)] / 255.0


def to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8)).resize(
        (TILE, TILE), Image.BILINEAR
    )


def panel(image: Image.Image, caption: str) -> Image.Image:
    frame = Image.new("RGB", (TILE, TILE + 18), "black")
    frame.paste(image, (0, 18))
    ImageDraw.Draw(frame).text((4, 4), caption, fill="white")
    return frame


def row_for(scene: str, cache: Path, index: int = 0) -> Image.Image | None:
    labels_file = cache / "labels" / f"{scene}.npz"
    if not labels_file.exists():
        return None
    meta = np.load(labels_file)
    if len(meta["windows"]) <= index:
        return None
    window = meta["windows"][index]
    top, left = int(window[0]), int(window[1])
    labels = meta["labels"][top : top + WINDOW, left : left + WINDOW]
    aef = np.load(cache / "aef" / f"{scene}.npz")["features"][index]
    olmo = np.load(cache / "olmo" / f"{scene}.npz")["features"][index]

    rgb = gf2_rgb(scene, window)
    labelled = rgb.copy()
    mask = labels > 0
    labelled[mask] = 0.45 * rgb[mask] + 0.55 * label_rgb(labels)[mask]

    panels = [
        panel(to_image(rgb), f"{scene[:22]} GF-2 (10 m)"),
        panel(to_image(labelled), "GF-2 + labels"),
        panel(to_image(pca_rgb(aef)), "AEF 2017 PCA"),
        panel(to_image(pca_rgb(olmo)), "OlmoEarth 2017 PCA"),
    ]
    row = Image.new("RGB", (TILE * len(panels), TILE + 18), "black")
    for position, item in enumerate(panels):
        row.paste(item, (position * TILE, 0))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache", type=Path, default=Path("/home/yzhong/rs-eval/rs-eval-cache/fbp2017")
    )
    parser.add_argument("--out", type=Path, default=Path("/home/yzhong/rs-eval/audits/fbp2017-qc"))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--index", type=int, default=0, help="window index per scene")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    scenes = (
        args.scenes
        or sorted(path.name[:-4] for path in (args.cache / "labels").glob("*.npz"))[: args.limit]
    )
    rows = []
    for scene in scenes:
        row = row_for(scene, args.cache, args.index)
        if row is None:
            continue
        target = args.out / f"{scene}_w{args.index}.png"
        row.save(target)
        rows.append(row)
        print(f"wrote {target}", flush=True)
    if rows:
        sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "black")
        offset = 0
        for row in rows:
            sheet.paste(row, (0, offset))
            offset += row.height
        sheet.save(args.out / "contact_sheet.png")
        print(f"wrote {args.out / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
