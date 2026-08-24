"""Render the images the project showcase needs, straight from the real pipeline.

    python scripts/showcase_assets.py                       # default case, auto slice
    python scripts/showcase_assets.py --case BraTS2021_00495 --slice 80

Three sets of pictures, all produced by the same functions the training runs call, so the
showcase illustrates what actually happened rather than a redrawing of it:

  modality_<m>.png   the four MRI channels the model receives as input
  regions_<r>.png    the three nested target masks, and the combined overlay
  shift_<H>.png      one slice as each hospital's simulated scanner renders it

The scanner-shift panel is the one that carries the argument: the four images are the same
patient, and the visible difference between Site A and Site D is the entire reason a single
global model struggles. Rendering it from `apply_shift` rather than illustrating it by hand
keeps the picture honest.

Outputs land in artifacts/showcase/assets/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedbrats.config import MODALITIES, Config          # noqa: E402
from fedbrats.data import brain_bbox, labels_to_regions, load_case  # noqa: E402
from fedbrats.shift import HOSPITAL_SHIFTS, apply_shift  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "artifacts" / "showcase" / "assets"

# The demo server's region colours. Kept identical so the showcase, the paper figures and the
# live demo can never disagree about which colour means which region.
REGION_COLORS = {"wt": (16, 185, 129), "tc": (59, 130, 246), "et": (236, 72, 153)}
SIDE = 240  # rendered edge, px — large enough to read on a projector, small enough to inline


def to_gray(sl: np.ndarray) -> np.ndarray:
    """One slice -> uint8, windowed on the brain so the background stays black."""
    brain = sl > 0
    if not brain.any():
        return np.zeros(sl.shape, np.uint8)
    lo, hi = np.percentile(sl[brain], [1.0, 99.0])
    if hi <= lo:
        lo, hi = float(sl[brain].min()), float(sl[brain].max()) + 1e-6
    out = np.clip((sl - lo) / (hi - lo), 0, 1)
    return (np.where(brain, out, 0.0) * 255).astype(np.uint8)


def square(img: Image.Image) -> Image.Image:
    """Pad to square, then resize — so no panel is stretched relative to another."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new(img.mode, (side, side), 0 if img.mode == "L" else (0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((SIDE, SIDE), Image.LANCZOS)


def orient(a: np.ndarray) -> np.ndarray:
    """BraTS arrays are (X, Y, Z); show an axial slice the way a radiologist reads it."""
    return np.flipud(a.T)


def save_gray(a: np.ndarray, path: Path) -> None:
    square(Image.fromarray(orient(a), mode="L")).save(path, optimize=True)


def overlay(gray: np.ndarray, masks: dict[str, np.ndarray], path: Path) -> None:
    """Paint region masks over the greyscale slice, drawn largest-first so nesting shows."""
    rgb = np.stack([orient(gray)] * 3, axis=-1).astype(np.float32)
    for name in ("wt", "tc", "et"):          # WT contains TC contains ET
        if name not in masks:
            continue
        m = orient(masks[name]).astype(bool)
        colour = np.array(REGION_COLORS[name], np.float32)
        rgb[m] = 0.45 * rgb[m] + 0.55 * colour
    square(Image.fromarray(rgb.astype(np.uint8), mode="RGB")).save(path, optimize=True)


def pick_slice(regions: np.ndarray) -> int:
    """The axial slice with the most whole-tumour area — what a reader would choose by hand."""
    return int(regions[0].sum(axis=(0, 1)).argmax())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default=None, help="case id (default: first case in the partition)")
    ap.add_argument("--slice", type=int, default=None, help="axial index (default: largest WT)")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    cfg = Config()
    root = Path(args.data_root) if args.data_root else cfg.paths.data_root
    if not root.exists():
        print(f"data root not found: {root}", file=sys.stderr)
        return 1

    case = args.case
    if case is None:
        manifest = json.loads(cfg.paths.manifest.read_text())
        case = sorted(manifest["assignment"])[0]
    print(f"case {case}  root {root}")

    mods, seg = load_case(root, case)
    regions = labels_to_regions(seg)
    z = args.slice if args.slice is not None else pick_slice(regions)
    print(f"slice {z}  WT voxels {int(regions[0][:, :, z].sum())}")

    # Crop to the brain so the panels are not mostly black border.
    bb = brain_bbox(mods.sum(axis=0) > 0)
    mods_c = mods[:, bb[0], bb[1], bb[2]]
    reg_c = regions[:, bb[0], bb[1], bb[2]]
    z_c = z - bb[2].start

    OUT.mkdir(parents=True, exist_ok=True)

    # 1. the four input channels
    for i, m in enumerate(MODALITIES):
        save_gray(to_gray(mods_c[i, :, :, z_c]), OUT / f"modality_{m}.png")

    # 2. the three targets, each alone, plus all three nested
    flair = to_gray(mods_c[0, :, :, z_c])
    for i, r in enumerate(("wt", "tc", "et")):
        overlay(flair, {r: reg_c[i, :, :, z_c]}, OUT / f"region_{r}.png")
    overlay(flair, {r: reg_c[i, :, :, z_c] for i, r in enumerate(("wt", "tc", "et"))},
            OUT / "region_all.png")
    save_gray(flair, OUT / "region_none.png")

    # 3. the same slice as each hospital's scanner renders it. Shift the full volume (the bias
    # field is 3D and seeded per hospital), then take the slice — shifting a lone slice would
    # sample a different part of the field and understate the difference.
    for h in sorted(HOSPITAL_SHIFTS):
        shifted = apply_shift(mods[:, bb[0], bb[1], bb[2]], h, seed=cfg.seed)
        save_gray(to_gray(shifted[0, :, :, z_c]), OUT / f"shift_{h}.png")
        print(f"  {h}: {HOSPITAL_SHIFTS[h]}")

    meta = {"case_id": case, "slice": int(z), "modalities": list(MODALITIES),
            "shifts": {h: p.__dict__ for h, p in sorted(HOSPITAL_SHIFTS.items())},
            "wt_voxels_slice": int(regions[0][:, :, z].sum()),
            "volume_shape": list(mods.shape[1:])}
    (OUT / "assets.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(list(OUT.glob('*.png')))} images to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
