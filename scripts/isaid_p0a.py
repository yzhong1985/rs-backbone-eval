"""iSAID P0-a data audit: sizes, tile counts and the semantic colour map.

Reproducible replacement for the ad-hoc analysis quoted in the design doc.
It reads only the downloaded annotations/masks (no training, no GPU) and
writes a JSON report under ``data_root/raw/isaid/``.

Why polygons instead of bbox centres: the first pass sampled the colour at the
bounding-box centre, which is wrong for ring-shaped (ground track field) or
elongated (harbor, bridge) objects -- the centre often lands on background.
Here every instance polygon is rasterised and only the pixels *inside* it are
counted, which is what the colour map claim needs.

Usage:
    uv run --with pillow --with numpy python scripts/isaid_p0a.py \
        --data-root /geodata/rs-eval/data \
        --out /geodata/rs-eval/data/raw/isaid/p0a_report.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

TILE_CONFIGS = ((1024, 512), (1024, 1024), (512, 256), (512, 512))


def _tiles(width: int, height: int, crop: int, stride: int) -> int:
    if width < crop or height < crop:
        return 0
    nx = (width - crop) // stride + 1
    ny = (height - crop) // stride + 1
    return nx * ny


def _polygons(segmentation) -> list[list[float]]:
    """iSAID stores polygons; tolerate a single flat ring too."""
    if isinstance(segmentation, list):
        if segmentation and isinstance(segmentation[0], (int, float)):
            return [segmentation]
        return [poly for poly in segmentation if poly]
    return []


def colour_map(
    split_dir: Path, images: list[dict], annotations: dict[int, list[dict]], limit: int
) -> dict:
    import numpy as np
    from PIL import Image, ImageDraw

    stride = max(1, len(images) // limit) if limit else 1
    sample = images[::stride][:limit]

    votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    palette: set[tuple[int, int, int]] = set()
    polygon_pixels = 0

    for image in sample:
        mask_path = split_dir / "images" / image["seg_file_name"]
        if not mask_path.is_file():
            continue
        with Image.open(mask_path) as handle:
            mask = np.asarray(handle.convert("RGB"))
        height, width = mask.shape[:2]
        for ann in annotations.get(image["id"], []):
            name = ann["category_name"]
            x, y, w, h = (int(round(v)) for v in ann["bbox"])
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + w + 1), min(height, y + h + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            local = Image.new("1", (x1 - x0, y1 - y0), 0)
            draw = ImageDraw.Draw(local)
            for poly in _polygons(ann["segmentation"]):
                points = [((poly[i] - x0), (poly[i + 1] - y0)) for i in range(0, len(poly) - 1, 2)]
                if len(points) >= 3:
                    draw.polygon(points, fill=1)
            inside = np.asarray(local, dtype=bool)
            if not inside.any():
                continue
            patch = mask[y0:y1, x0:x1][inside]
            polygon_pixels += int(inside.sum())
            for colour in np.unique(patch, axis=0):
                votes[name][tuple(int(c) for c in colour)] += 1
            palette |= {tuple(int(c) for c in colour) for colour in np.unique(patch, axis=0)}

    mapping: dict[str, str] = {}
    shares: dict[str, float] = {}
    for name, counter in votes.items():
        colour, count = counter.most_common(1)[0]
        total = sum(counter.values())
        mapping[name] = "#{:02x}{:02x}{:02x}".format(*colour)
        shares[name] = round(count / total, 4) if total else 0.0

    by_colour: dict[str, list[str]] = collections.defaultdict(list)
    for name, colour in mapping.items():
        by_colour[colour].append(name)
    collisions = {colour: names for colour, names in by_colour.items() if len(names) > 1}

    return {
        "images_sampled": len(sample),
        "instance_pixels": polygon_pixels,
        "palette_size": len(palette),
        "palette": sorted("#{:02x}{:02x}{:02x}".format(*colour) for colour in palette),
        "mapping": dict(sorted(mapping.items())),
        "consistency": dict(sorted(shares.items())),
        "collisions": collisions,
        "weak": sorted(name for name, share in shares.items() if share < 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/geodata/rs-eval/data"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--color-images", type=int, default=200)
    args = parser.parse_args()

    from PIL import Image

    base = args.data_root / "raw" / "isaid"
    masks = base / "masks"
    out_path = args.out or base / "p0a_report.json"

    report: dict = {"data_root": str(args.data_root), "splits": {}}
    for split in ("train", "val"):
        source = base / split / f"iSAID_{split}.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        images = payload["images"]
        annotations: dict[int, list[dict]] = collections.defaultdict(list)
        for ann in payload["annotations"]:
            annotations[ann["image_id"]].append(ann)

        split_dir = masks / split
        if not split_dir.is_dir():
            split_dir = base / split
        sizes = []
        for image in images:
            path = split_dir / "images" / image["seg_file_name"]
            if path.is_file():
                with Image.open(path) as handle:
                    sizes.append(handle.size)

        tiles = {
            f"{crop}/{stride}": sum(_tiles(w, h, crop, stride) for w, h in sizes)
            for crop, stride in TILE_CONFIGS
        }
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        entry: dict = {
            "images": len(images),
            "annotations": len(payload["annotations"]),
            "categories": len(payload["categories"]),
            "masks_found": len(sizes),
            "width": {"min": min(widths), "max": max(widths), "median": statistics.median(widths)},
            "height": {
                "min": min(heights),
                "max": max(heights),
                "median": statistics.median(heights),
            },
            "tiles": tiles,
        }
        if split == "train":
            entry["colour"] = colour_map(split_dir, images, annotations, args.color_images)
        report["splits"][split] = entry

    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    report["annotation_sha256"] = {"train": checksum}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_path)
    train = report["splits"]["train"]
    print("tiles:", train["tiles"])
    colour = train.get("colour", {})
    print("palette:", colour.get("palette_size"), "collisions:", colour.get("collisions"))
    print("weak classes:", colour.get("weak"))


if __name__ == "__main__":
    main()
