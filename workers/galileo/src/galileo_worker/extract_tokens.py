"""Frozen Galileo-Nano tokens for T13 tiles, NIR only.

For each tile of one split: the NIR band (band 4, uint8 DN) is mapped to the B8 scale with
the moment-matched linear map in nir_to_b8.json (a documented heuristic, no official
GF-1 / WHU calibration was found), normalised with the upstream B8 mean and std
(config/normalization.json, 2*std clip as in upstream Normalizer), and fed as the only
visible space-time group (S2_NIR_10m); every other group is masked (mask 1 = missing).
The 512 px tile gives a 64x64 grid of 8 px patches; the NIR group token is average-pooled
2x2 to 32x32 and stored as fp16 [128, 32, 32] in OUT/<split>/<sample_id>.pt.

usage: python -m galileo_worker.extract_tokens --manifest M --data-root D --split train --out O
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from galileo_upstream import galileo as galileo_model
from galileo_upstream.data.dataset import (
    SPACE_BANDS,
    SPACE_TIME_BANDS,
    SPACE_TIME_BANDS_GROUPS_IDX,
    STATIC_BANDS,
    TIME_BANDS,
)
from rasterio.windows import Window

ENCODER_DIR = Path("/geodata/rs-eval/weights/galileo/nano")
MAP_JSON = Path("/home/yzhong/diagnostics/galileo_src/nir_to_b8.json")
NORM_JSON = Path("/home/yzhong/diagnostics/galileo_src/normalization.json")
RGB_MAP_JSON = Path("/home/yzhong/diagnostics/galileo_src/rgb_to_s2.json")
NIR_BAND = 3  # 0-based: R, G, B, NIR
PATCH = 8
MONTH = 6


def load_rows(manifest: Path, split: str) -> list[dict]:
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("task_id") == "T13" and row.get("split") == split:
                    rows.append(row)
    return rows


def read_nir(data_root: Path, row: dict) -> np.ndarray:
    x, y, w, h = (int(v) for v in row["window"])
    with rasterio.open(data_root / row["image_paths"][0]) as src:
        return src.read(NIR_BAND + 1, window=Window(x, y, w, h)).astype(np.float32)


def to_b8(dn: np.ndarray, mapping: dict, norm: dict) -> np.ndarray:
    """DN -> B8 scale -> upstream normalisation (shift = mu - 2 sd, div = 4 sd)."""
    b8 = mapping["a"] * dn + mapping["b"]
    mu, sd = norm["mean"][mapping["B8_index"]], norm["std"][mapping["B8_index"]]
    return (b8 - (mu - 2 * sd)) / (4 * sd)


def read_rgb(data_root: Path, row: dict) -> np.ndarray:
    x, y, w, h = (int(v) for v in row["window"])
    with rasterio.open(data_root / row["image_paths"][0]) as src:
        return src.read([1, 2, 3], window=Window(x, y, w, h)).astype(np.float32)


def to_s2(dn: np.ndarray, a: float, b: float, index: int, norm: dict) -> np.ndarray:
    """DN -> S2 scale (moment-matched a, b) -> upstream normalisation for that band."""
    s2 = a * dn + b
    mu, sd = norm["mean"][index], norm["std"][index]
    return (s2 - (mu - 2 * sd)) / (4 * sd)


def load_model(device: str) -> galileo_model.Encoder:
    cfg = json.loads((ENCODER_DIR / "config.json").read_text())["model"]["encoder"]
    kw = {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}
    enc = galileo_model.Encoder(**kw)
    sd = torch.load(ENCODER_DIR / "encoder.pt", map_location="cpu", weights_only=True)
    enc.load_state_dict(sd, strict=True)
    return enc.to(device).eval()


@torch.no_grad()
def tokens_for(
    enc,
    nir_norm: torch.Tensor,
    nir_group: int,
    device: str,
    *,
    pool: bool = True,
    rgb_norm: torch.Tensor | None = None,
    rgb_group: int | None = None,
) -> torch.Tensor:
    """nir_norm: [B, H, W] normalised B8. Returns [B, 128, H/16, W/16] (fp32)."""
    b, h, w = nir_norm.shape
    groups = list(SPACE_TIME_BANDS_GROUPS_IDX.keys())
    s_t_x = torch.zeros(b, h, w, 1, len(SPACE_TIME_BANDS), device=device)
    s_t_x[:, :, :, 0, SPACE_TIME_BANDS.index("B8")] = nir_norm
    s_t_m = torch.ones(b, h, w, 1, len(groups), device=device)
    s_t_m[..., nir_group] = 0
    if rgb_norm is not None:  # [B, 3, H, W] as (B2, B3, B4): RGB is visible with NIR
        for k, name in enumerate(("B2", "B3", "B4")):
            s_t_x[:, :, :, 0, SPACE_TIME_BANDS.index(name)] = rgb_norm[:, k]
        s_t_m[..., rgb_group] = 0
    sp_x = torch.zeros(b, h, w, len(SPACE_BANDS), device=device)
    sp_m = torch.ones(b, h, w, len(SPACE_BANDS), device=device)
    t_x = torch.zeros(b, 1, len(TIME_BANDS), device=device)
    t_m = torch.ones(b, 1, len(TIME_BANDS), device=device)
    st_x = torch.zeros(b, len(STATIC_BANDS), device=device)
    st_m = torch.ones(b, len(STATIC_BANDS), device=device)
    months = torch.full((b, 1), MONTH, dtype=torch.long, device=device)
    out = enc(s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months, patch_size=PATCH)
    s_t = out[0]  # [B, H/8, W/8, 1, groups, 128]
    tok = s_t[:, :, :, 0, nir_group, :]  # [B, H/8, W/8, 128]
    tok = tok.permute(0, 3, 1, 2)
    if not pool:
        return tok  # native 64x64 patch grid for a 512 px tile
    return F.avg_pool2d(tok, kernel_size=2)  # 512 px -> 64x64 -> 32x32


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="keep the native 64x64 grid (F2+F3)")
    ap.add_argument(
        "--groups",
        choices=("nir", "rgbnir"),
        default="nir",
        help="nir: only B8 visible; rgbnir: B2/B3/B4 from the RGB bands are visible too",
    )
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mapping = json.loads(MAP_JSON.read_text())
    norm_all = json.loads(NORM_JSON.read_text())
    norm = norm_all[str(len(SPACE_TIME_BANDS))]
    groups = list(SPACE_TIME_BANDS_GROUPS_IDX.keys())
    nir_group = groups.index("S2_NIR_10m")
    rgb_map = json.loads(RGB_MAP_JSON.read_text()) if args.groups == "rgbnir" else None
    rgb_group = groups.index("S2_RGB") if rgb_map is not None else None
    rows = load_rows(args.manifest, args.split)
    if args.limit:
        rows = rows[: args.limit]
    out_dir = args.out / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    enc = load_model(device)
    for start in range(0, len(rows), args.batch):
        chunk = rows[start : start + args.batch]
        dn = np.stack([read_nir(args.data_root, r) for r in chunk])
        x = torch.from_numpy(to_b8(dn, mapping, norm)).float().to(device)
        rgb_norm = None
        if rgb_map is not None:
            rgb = np.stack([read_rgb(args.data_root, r) for r in chunk])  # R, G, B
            chan = {"R": 0, "G": 1, "B": 2}
            rgb_norm = np.stack(
                [
                    to_s2(
                        rgb[:, chan[c]], rgb_map[c]["a"], rgb_map[c]["b"], rgb_map[c]["index"], norm
                    )
                    for c in ("B", "G", "R")  # B2, B3, B4
                ],
                axis=1,
            )
            rgb_norm = torch.from_numpy(rgb_norm).float().to(device)
        tok = tokens_for(
            enc, x, nir_group, device, pool=not args.full, rgb_norm=rgb_norm, rgb_group=rgb_group
        )
        for row, t in zip(chunk, tok.cpu(), strict=True):
            torch.save(t.half().clone(), out_dir / f"{row['sample_id']}.pt")
    print(f"wrote {len(rows)} token files to {out_dir} on {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
