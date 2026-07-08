"""Data: load a case, preprocess it, cache it, and sample training units from it.

Pipeline order (see docs/data-pipeline.md §3):
    load -> brain mask/bbox (UNSHIFTED) -> shift -> re-mask -> crop -> z-norm -> clip -> regions

The brain mask and the crop bbox are derived from the **unshifted** volume, then the shifted
volume is re-masked with it. This is load-bearing: `apply_shift`'s Gaussian blur smears brain
intensity into BraTS's exactly-zero background, so a mask taken *after* the shift grows with the
hospital's blur sigma (H4: +57% voxels). That would make the crop shape hospital-dependent and
leak hospital identity as geometry -- contaminating the very H2/H3 claims we are testing.

Cache layout (see `cache_key`):
    <cache>/<key>/<case_id>/x.npy    (4, X, Y, Z) float16   z-normed, clipped modalities
    <cache>/<key>/<case_id>/y.npy    (3, X, Y, Z) uint8     WT / TC / ET masks
    <cache>/<key>/<case_id>/meta.json
    <cache>/<key>/index.json         assembled from the per-case meta files
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import MODALITIES, Config
from .shift import HOSPITAL_SHIFTS, apply_shift

CACHE_VERSION = "v1"


# --------------------------------------------------------------------------------------
# load + preprocess
# --------------------------------------------------------------------------------------

def load_case(root, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a case -> (modalities (4, X, Y, Z) float32, seg (X, Y, Z) uint8).

    Accepts either `.nii` or `.nii.gz` (nibabel handles both), so the same loader works on the
    local unzipped D: copy and the compressed source.
    """
    cdir = os.path.join(str(root), case_id)
    mods = []
    for m in MODALITIES:
        matches = glob.glob(os.path.join(cdir, f"*_{m}.nii*"))
        if not matches:
            raise FileNotFoundError(f"{case_id}: missing modality '{m}' in {cdir}")
        mods.append(np.asarray(nib.load(matches[0]).dataobj, dtype=np.float32))
    seg_matches = glob.glob(os.path.join(cdir, "*_seg.nii*"))
    if not seg_matches:
        raise FileNotFoundError(f"{case_id}: missing segmentation in {cdir}")
    seg = np.asarray(nib.load(seg_matches[0]).dataobj, dtype=np.uint8)
    return np.stack(mods, axis=0), seg


def brain_bbox(brain: np.ndarray) -> tuple[slice, slice, slice]:
    """Bounding box of a (X, Y, Z) boolean brain mask."""
    coords = np.argwhere(brain)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def labels_to_regions(seg: np.ndarray) -> np.ndarray:
    """seg {0,1,2,4} -> (3, X, Y, Z) uint8 masks: WT (1,2,4), TC (1,4), ET (4)."""
    wt = np.isin(seg, [1, 2, 4])
    tc = np.isin(seg, [1, 4])
    et = seg == 4
    return np.stack([wt, tc, et], axis=0).astype(np.uint8)


def preprocess(
    mods: np.ndarray,
    seg: np.ndarray,
    hospital: str | None = None,
    seed: int = 42,
    clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full per-case preprocessing.

    Returns (x, y, brain):
      x     : (4, X', Y', Z') float32 z-normalized, clipped, cropped modalities
      y     : (3, X', Y', Z') uint8   WT/TC/ET region masks
      brain : (X', Y', Z')    bool    brain mask

    The mask/bbox come from the unshifted volume -- see the module docstring.
    """
    brain_full = mods.sum(axis=0) > 0           # BEFORE the shift: the true skull-stripped brain
    bbox = brain_bbox(brain_full)               # hospital-independent crop

    if hospital is not None:
        mods = apply_shift(mods, hospital, seed) * brain_full   # re-mask: kill the blur halo

    mods = mods[(slice(None),) + bbox]
    seg = seg[bbox]
    brain = brain_full[bbox]

    x = np.zeros_like(mods, dtype=np.float32)
    for c in range(mods.shape[0]):
        vals = mods[c][brain]
        if vals.size:
            mu = float(vals.mean())
            sd = float(vals.std()) + 1e-8
            z = np.clip((mods[c] - mu) / sd, -clip, clip)
            x[c] = np.where(brain, z, 0.0)

    return x, labels_to_regions(seg), brain


# --------------------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------------------

def cache_key(cfg: Config) -> str:
    """Identifies a cache by everything that changes its *content*.

    Shift parameters and the clip threshold are hashed in, so changing the scanner shift
    invalidates the cache instead of silently reusing stale tensors.
    """
    payload = json.dumps(
        {
            "version": CACHE_VERSION,
            "clip": cfg.clip_sigma,
            "seed": cfg.seed,
            "shifts": {h: p.__dict__ for h, p in sorted(HOSPITAL_SHIFTS.items())},
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def case_cache_dir(cfg: Config, case_id: str) -> Path:
    return Path(cfg.paths.cache) / cache_key(cfg) / case_id


def _tumor_bbox(wt: np.ndarray) -> list[int] | None:
    """Bounding box of the whole-tumour mask, as [x0,x1,y0,y1,z0,z1]; None if no tumour."""
    coords = np.argwhere(wt)
    if coords.size == 0:
        return None
    lo, hi = coords.min(axis=0), coords.max(axis=0) + 1
    return [int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]), int(lo[2]), int(hi[2])]


def build_case_cache(task: tuple) -> str:
    """Preprocess ONE case and write it to the cache. Skip-safe and resumable.

    Top-level function taking a picklable tuple, so it works under Windows `spawn`
    as well as Linux `fork` (see docs/environments.md).
    """
    case_id, hospital, split, data_root, out_dir, seed, clip = task
    out = Path(out_dir)
    if (out / "meta.json").exists():
        return f"skip {case_id}"

    mods, seg = load_case(data_root, case_id)
    x, y, _ = preprocess(mods, seg, hospital=hospital, seed=seed, clip=clip)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "x.npy", x.astype(np.float16))
    np.save(out / "y.npy", y)                       # already uint8
    meta = {
        "case_id": case_id,
        "hospital": hospital,
        "split": split,
        "shape": list(x.shape[1:]),
        "tumor_z": np.flatnonzero(y[0].any(axis=(0, 1))).tolist(),   # z with any WT
        "tumor_bbox": _tumor_bbox(y[0]),
    }
    with (out / "meta.json").open("w") as f:
        json.dump(meta, f)
    return f"built {case_id}"


def assemble_index(cfg: Config) -> dict:
    """Collect the per-case meta.json files into one index (written after a cache build)."""
    root = Path(cfg.paths.cache) / cache_key(cfg)
    index = {}
    for meta_path in sorted(root.glob("*/meta.json")):
        with meta_path.open() as f:
            m = json.load(f)
        index[m["case_id"]] = m
    with (root / "index.json").open("w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
    return index


def load_index(cfg: Config) -> dict:
    path = Path(cfg.paths.cache) / cache_key(cfg) / "index.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No cache index at {path}. Build it first: python scripts/build_cache.py"
        )
    with path.open() as f:
        return json.load(f)


def load_cached_case(cfg: Config, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Memory-mapped (x, y) for one cached case. Read-only; caller must not hold handles."""
    d = case_cache_dir(cfg, case_id)
    x = np.load(d / "x.npy", mmap_mode="r")
    y = np.load(d / "y.npy", mmap_mode="r")
    return x, y


# --------------------------------------------------------------------------------------
# spatial helpers
# --------------------------------------------------------------------------------------

def pad_to_multiple(arr: np.ndarray, m: int, n_spatial: int) -> tuple[np.ndarray, tuple]:
    """Right-pad the trailing `n_spatial` axes up to a multiple of `m`.

    Returns (padded, crop_slices) where crop_slices undoes the padding.
    """
    pads = [(0, 0)] * (arr.ndim - n_spatial)
    crops = []
    for ax in range(arr.ndim - n_spatial, arr.ndim):
        n = arr.shape[ax]
        target = ((n + m - 1) // m) * m
        pads.append((0, target - n))
        crops.append(slice(0, n))
    return np.pad(arr, pads, mode="constant"), tuple(crops)


def _rand_start(extent: int, size: int, centre: int | None, g: torch.Generator) -> int:
    """A random start index in [0, extent-size], biased to contain `centre` when given."""
    if extent <= size:
        return 0
    if centre is None:
        return int(torch.randint(0, extent - size + 1, (1,), generator=g).item())
    lo = max(0, min(centre - size + 1, extent - size))
    hi = max(0, min(centre, extent - size))
    if hi <= lo:
        return lo
    return int(torch.randint(lo, hi + 1, (1,), generator=g).item())


def _fit_hw(a: np.ndarray, hw: int, g: torch.Generator | None) -> np.ndarray:
    """Crop/pad the last two axes of a (C, H, W) array to (hw, hw). Random crop if `g` given."""
    c, h, w = a.shape
    if h > hw:
        i = _rand_start(h, hw, None, g) if g is not None else (h - hw) // 2
        a = a[:, i:i + hw, :]
    if w > hw:
        j = _rand_start(w, hw, None, g) if g is not None else (w - hw) // 2
        a = a[:, :, j:j + hw]
    ph, pw = hw - a.shape[1], hw - a.shape[2]
    if ph or pw:
        a = np.pad(a, ((0, 0), (0, ph), (0, pw)), mode="constant")
    return a


# --------------------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------------------

class _CachedCases(Dataset):
    """Shared base: lazily opens memmaps per worker, never across a fork/spawn boundary."""

    def __init__(self, cfg: Config, case_ids: list[str], index: dict):
        self.cfg = cfg
        self.case_ids = list(case_ids)
        self.index = index
        self._handles: dict[str, tuple] = {}

    def _get(self, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        if case_id not in self._handles:
            self._handles[case_id] = load_cached_case(self.cfg, case_id)
        return self._handles[case_id]

    def _gen(self, idx: int) -> torch.Generator:
        """Per-item RNG: torch seeds each DataLoader worker's default generator distinctly."""
        g = torch.Generator()
        g.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()) + idx)
        return g


class SliceDataset(_CachedCases):
    """2D: tumour-biased axial slices.

    `tumor_frac` of samples come from slices containing whole tumour; the rest are uniform over
    z. Training on tumour slices ONLY makes the model hallucinate tumour on the empty slices it
    meets during full-volume evaluation -- hence the deliberate mix.
    """

    def __len__(self) -> int:
        return len(self.case_ids) * self.cfg.slices_per_case

    def __getitem__(self, i: int):
        case_id = self.case_ids[i // self.cfg.slices_per_case]
        g = self._gen(i)
        x, y = self._get(case_id)
        tumor_z = self.index[case_id]["tumor_z"]
        nz = x.shape[3]

        pick_tumor = bool(tumor_z) and torch.rand(1, generator=g).item() < self.cfg.tumor_frac
        if pick_tumor:
            z = tumor_z[int(torch.randint(0, len(tumor_z), (1,), generator=g).item())]
        else:
            z = int(torch.randint(0, nz, (1,), generator=g).item())

        xs = np.asarray(x[:, :, :, z], dtype=np.float32)     # (4, H, W)
        ys = np.asarray(y[:, :, :, z], dtype=np.float32)     # (3, H, W)
        hw = self.cfg.train_hw
        # crop both with the same offsets: concatenate, cut once, split back
        both = _fit_hw(np.concatenate([xs, ys], axis=0), hw, g)
        return torch.from_numpy(both[:4].copy()), torch.from_numpy(both[4:].copy())


class PatchDataset(_CachedCases):
    """3D: foreground-biased cubic patches."""

    def __len__(self) -> int:
        return len(self.case_ids) * self.cfg.patches_per_case

    def __getitem__(self, i: int):
        case_id = self.case_ids[i // self.cfg.patches_per_case]
        g = self._gen(i)
        x, y = self._get(case_id)
        p = self.cfg.patch_size
        shape = x.shape[1:]
        tb = self.index[case_id]["tumor_bbox"]

        centres: list[int | None] = [None, None, None]
        if tb and torch.rand(1, generator=g).item() < self.cfg.tumor_frac:
            centres = [
                int(torch.randint(tb[2 * a], max(tb[2 * a] + 1, tb[2 * a + 1]), (1,), generator=g).item())
                for a in range(3)
            ]

        starts = [_rand_start(shape[a], p, centres[a], g) for a in range(3)]
        sl = tuple(slice(s, s + p) for s in starts)
        xs = np.asarray(x[(slice(None),) + sl], dtype=np.float32)
        ys = np.asarray(y[(slice(None),) + sl], dtype=np.float32)

        pads = [(0, 0)] + [(0, p - xs.shape[a + 1]) for a in range(3)]
        if any(b for _, b in pads):
            xs = np.pad(xs, pads, mode="constant")
            ys = np.pad(ys, pads, mode="constant")
        return torch.from_numpy(xs), torch.from_numpy(ys)


def build_dataset(cfg: Config, case_ids: list[str], index: dict) -> Dataset:
    return SliceDataset(cfg, case_ids, index) if cfg.is_2d else PatchDataset(cfg, case_ids, index)


def select_cases(index: dict, hospital: str, split: str, limit: int | None = None) -> list[str]:
    """Cached case IDs for one hospital+split, deterministically ordered and optionally capped."""
    ids = sorted(c for c, m in index.items() if m["hospital"] == hospital and m["split"] == split)
    return ids[:limit] if limit else ids
