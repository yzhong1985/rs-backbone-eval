"""iSAID P0-a data audit: provenance hashes, coverage, tile counts, colour map.

Reproducible replacement for the ad-hoc analysis quoted in the design doc.
Reads only the downloaded annotations/masks (no training, no GPU) and writes a
JSON report under ``data_root/derived/isaid/``. The raw downloads stay in
``data_root/raw/isaid/``.

Conventions that matter for the protocol:

* tiling **covers every pixel**: positions are ``ceil((dim - crop) / stride) + 1``
  and the edge tile is padded, so no image is dropped and no pixel is skipped.
  An image smaller than the crop gets a single fully padded tile. The earlier
  version dropped such images (418/1411 on train) and stopped at the last
  whole step, covering only ~80% of the pixels -- unusable for evaluation,
  where the official mIoU is computed on the whole image;
* the colour map is built from **instance polygons rasterised onto the
  semantic mask**, weighted by **pixels** (not per-instance vote counts), and
  the second most frequent colour per class is reported so overlap effects are
  visible.

Usage:
    uv run --with pillow --with numpy python scripts/isaid_p0a.py
    uv run --with pillow --with numpy python scripts/isaid_p0a.py --color-images 200 --skip-hash
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
import statistics
import zipfile
from pathlib import Path

from rs_backbone_eval.data.tiling import tiles_padded

TILE_CONFIGS = ((1024, 512), (1024, 1024), (512, 256), (512, 512))
MASK_ZIPS = {"train": "train/semantic_masks.zip", "val": "val/semantic_masks.zip"}
# The two archives hold exactly these many masks; anything else means the
# extraction was interrupted and the next run must redo it instead of skipping
# a half-populated directory (review 2026-10-29).
EXPECTED_MASKS = {"train": 1411, "val": 458}
DOTA_ZIPS = (
    "dota/train/part1.zip",
    "dota/train/part2.zip",
    "dota/train/part3.zip",
    "dota/val/part1.zip",
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def ensure_masks(raw: Path, derived: Path) -> Path:
    """Extract the semantic-mask zips into ``derived`` and verify the count.

    A directory that merely exists is not proof of a complete extraction: an
    interrupted unzip leaves a few hundred files behind and the old check
    would skip it forever. Each split must hold its exact mask count.
    """
    target = derived / "masks"
    for split, relative in MASK_ZIPS.items():
        out_dir = target / split
        images = out_dir / "images"
        expected = EXPECTED_MASKS[split]
        if _count_pngs(images) == expected:
            continue
        archive = raw / relative
        if not archive.is_file():
            raise SystemExit(f"missing mask archive: {archive}")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(out_dir)
        found = _count_pngs(images)
        if found != expected:
            raise SystemExit(
                f"{split}: extracted {found} masks, expected {expected}; re-run "
                f"after checking {archive}"
            )
    return target


def _count_pngs(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.png")) if directory.is_dir() else 0


def _polygons(segmentation) -> list[list[float]]:
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
            colours, counts = np.unique(patch, axis=0, return_counts=True)
            polygon_pixels += int(inside.sum())
            for colour, count in zip(colours, counts, strict=True):
                key = tuple(int(c) for c in colour)
                votes[name][key] += int(count)
                palette.add(key)

    def hexed(colour: tuple[int, int, int]) -> str:
        return "#{:02x}{:02x}{:02x}".format(*colour)

    mapping: dict[str, str] = {}
    consistency: dict[str, float] = {}
    runner_up: dict[str, dict] = {}
    for name, counter in votes.items():
        total = sum(counter.values())
        (colour, count), *rest = counter.most_common()
        mapping[name] = hexed(colour)
        consistency[name] = round(count / total, 4) if total else 0.0
        if rest:
            second, second_count = rest[0]
            runner_up[name] = {
                "colour": hexed(second),
                "share": round(second_count / total, 4) if total else 0.0,
            }

    by_colour: dict[str, list[str]] = collections.defaultdict(list)
    for name, colour in mapping.items():
        by_colour[colour].append(name)
    collisions = {colour: names for colour, names in by_colour.items() if len(names) > 1}

    return {
        "images_sampled": len(sample),
        "instance_pixels": polygon_pixels,
        "weighting": "pixels",
        "palette_size": len(palette),
        "palette": sorted(hexed(colour) for colour in palette),
        "mapping": dict(sorted(mapping.items())),
        "consistency": dict(sorted(consistency.items())),
        "runner_up": dict(sorted(runner_up.items())),
        "collisions": collisions,
        "weak": sorted(name for name, share in consistency.items() if share < 0.95),
    }


def main() -> None:
    from PIL import Image

    from rs_backbone_eval.paths import RuntimePaths

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--color-images", type=int, default=200)
    parser.add_argument("--skip-hash", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root or RuntimePaths.from_environment().data_root
    raw = data_root / "raw" / "isaid"
    derived = data_root / "derived" / "isaid"
    derived.mkdir(parents=True, exist_ok=True)
    out_path = args.out or derived / "p0a_report.json"

    masks = ensure_masks(raw, derived)

    report: dict = {"data_root": str(data_root), "masks": str(masks), "splits": {}}
    for split in ("train", "val"):
        payload = json.loads((raw / split / f"iSAID_{split}.json").read_text(encoding="utf-8"))
        images = payload["images"]
        annotations: dict[int, list[dict]] = collections.defaultdict(list)
        for ann in payload["annotations"]:
            annotations[ann["image_id"]].append(ann)

        split_dir = masks / split
        sizes = []
        for image in images:
            path = split_dir / "images" / image["seg_file_name"]
            if path.is_file():
                with Image.open(path) as handle:
                    sizes.append(handle.size)

        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        tiles = {
            f"{crop}/{stride}": sum(tiles_padded(w, h, crop, stride) for w, h in sizes)
            for crop, stride in TILE_CONFIGS
        }
        entry: dict = {
            "images": len(images),
            "annotations": len(payload["annotations"]),
            "categories": len(payload["categories"]),
            "masks_found": len(sizes),
            "images_smaller_than_1024": sum(1 for w, h in sizes if w < 1024 or h < 1024),
            "width": {"min": min(widths), "max": max(widths), "median": statistics.median(widths)},
            "height": {
                "min": min(heights),
                "max": max(heights),
                "median": statistics.median(heights),
            },
            "tiles": tiles,
            "tiling": "ceil((dim-crop)/stride)+1, edge tile padded, small images padded whole",
            "pixel_coverage": 1.0,
        }
        if split == "train":
            entry["colour"] = colour_map(split_dir, images, annotations, args.color_images)
        report["splits"][split] = entry

    if not args.skip_hash:
        hashes = {}
        for relative in ("train/iSAID_train.json", "val/iSAID_val.json", *MASK_ZIPS.values()):
            path = raw / relative
            if path.is_file():
                hashes[relative] = sha256_file(path)
        for relative in DOTA_ZIPS:
            path = raw / relative
            if path.is_file():
                hashes[relative] = sha256_file(path)
        report["sha256"] = hashes

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(out_path)
    train = report["splits"]["train"]
    print("tiles:", train["tiles"])
    print("small images:", train["images_smaller_than_1024"])
    colour = train.get("colour", {})
    print("palette:", colour.get("palette_size"), "collisions:", colour.get("collisions"))
    print("weak:", colour.get("weak"))


if __name__ == "__main__":
    main()
