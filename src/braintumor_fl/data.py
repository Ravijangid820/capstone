"""BraTS 2D data pipeline.

We never pre-slice to disk (that would be tens of GB). Instead we build a small
index of (case, axial-slice-z) pairs that contain tumor, and load each 2D slice
lazily from the 3D .nii.gz at access time via nibabel's memory-mapped dataobj.
"""

from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from monai.transforms import (
    Compose,
    ConvertToMultiChannelBasedOnBratsClassesd,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ToTensord,
)
from torch.utils.data import DataLoader, Dataset

from . import MODALITIES


# ----------------------------------------------------------------------------
# Case discovery + tumor-slice index
# ----------------------------------------------------------------------------

def find_cases(data_root: str) -> list[str]:
    """Return sorted case directories under data_root (each has *_seg.nii.gz)."""
    segs = glob.glob(os.path.join(data_root, "**", "*_seg.nii.gz"), recursive=True)
    return sorted(os.path.dirname(p) for p in segs)


def _modality_path(case_dir: str, modality: str) -> str:
    case_id = os.path.basename(case_dir)
    return os.path.join(case_dir, f"{case_id}_{modality}.nii.gz")


def _seg_path(case_dir: str) -> str:
    case_id = os.path.basename(case_dir)
    return os.path.join(case_dir, f"{case_id}_seg.nii.gz")


def build_slice_index(
    cases: list[str],
    min_tumor_pixels: int = 100,
    cache_csv: str | None = None,
) -> list[tuple[str, int]]:
    """Index every axial slice that contains at least `min_tumor_pixels` of tumor.

    Reads only the (small) seg volumes. The CSV cache is keyed by case dir and is
    incremental: only cases missing from the cache are recomputed, so growing the
    case set (5 -> 100 -> 1251) never rescans work already done. Returns slices for
    exactly the requested `cases`.

    Note: the cache assumes a fixed `min_tumor_pixels`. If you change it, delete the
    cache file first.
    """
    requested = list(cases)
    cached: dict[str, list[int]] = {}
    if cache_csv and os.path.exists(cache_csv):
        with open(cache_csv, newline="") as f:
            for row in csv.reader(f):
                if row:
                    cached.setdefault(row[0], []).append(int(row[1]))

    missing = [c for c in requested if c not in cached]
    new_rows: list[tuple[str, int]] = []
    for case_dir in missing:
        seg = np.asarray(nib.load(_seg_path(case_dir)).dataobj)  # (H, W, D)
        per_slice = (seg > 0).sum(axis=(0, 1))  # tumor pixels per axial slice
        zs = [int(z) for z in np.where(per_slice >= min_tumor_pixels)[0]]
        cached[case_dir] = zs
        new_rows.extend((case_dir, z) for z in zs)

    if cache_csv and new_rows:
        os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
        with open(cache_csv, "a", newline="") as f:  # append only the new cases
            csv.writer(f).writerows(new_rows)

    return [(c, z) for c in requested for z in cached.get(c, [])]


# ----------------------------------------------------------------------------
# Synthetic per-hospital scanner shift (non-IID without FeTS)
# ----------------------------------------------------------------------------
#
# Each hospital gets a deterministic "scanner profile" that remaps its images.
# Site 0 is the reference scanner (identity); later sites drift progressively, so
# higher-numbered hospitals are bigger domain outliers — exactly the ones FedAvg
# should hurt (H2) and FedBN should recover (H3).
#
# The shifts are NON-LINEAR (gamma) or SPATIAL (bias field, blur) on purpose: the
# transform pipeline z-normalizes each channel (NormalizeIntensityd, nonzero,
# channel-wise), which would erase any plain affine `a*x+b` shift. Gamma changes
# the intensity histogram's SHAPE and the bias field / blur change local spatial
# statistics, so they survive normalization and land in the BatchNorm running
# stats that FedBN keeps local.

SITE_PROFILES = [
    dict(gamma=1.0,  bias=0.0,  blur=0.0),  # site-1: reference scanner (no-op)
    dict(gamma=1.5,  bias=0.0,  blur=0.0),  # darker mid-tones
    dict(gamma=0.65, bias=0.0,  blur=0.0),  # brighter mid-tones
    dict(gamma=1.0,  bias=0.0,  blur=1.1),  # low-resolution scanner (PSF blur)
    dict(gamma=1.35, bias=0.45, blur=0.6),  # field inhomogeneity + contrast
    dict(gamma=0.75, bias=0.35, blur=0.4),  # opposite contrast + field
    dict(gamma=1.7,  bias=0.5,  blur=0.9),  # strong outlier
    dict(gamma=0.55, bias=0.4,  blur=0.7),  # strong outlier (other direction)
]


def site_shift_params(site: int) -> dict:
    """Scanner profile for a hospital (cycles if more sites than profiles)."""
    return SITE_PROFILES[site % len(SITE_PROFILES)]


def _bias_field(h: int, w: int, site: int) -> np.ndarray:
    """Smooth low-frequency multiplicative field, deterministic per site — models
    MRI intensity inhomogeneity (each coil/scanner has its own smooth gain)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy = yy / max(h - 1, 1) - 0.5
    xx = xx / max(w - 1, 1) - 0.5
    a = (site * 1.7) % (2 * np.pi)               # tilt direction varies per site
    field = np.cos(a) * xx + np.sin(a) * yy      # tilted plane
    field = field + 0.5 * (xx * xx + yy * yy)    # + radial bowl (coil falloff)
    return (field - field.mean()).astype(np.float32)


def apply_site_shift(image: np.ndarray, site: int) -> np.ndarray:
    """Apply a hospital's deterministic scanner shift to a (C,H,W) raw image.

    Operates per channel on the brain (nonzero) voxels, keeping the background at
    0 so the downstream nonzero normalization still sees a clean brain mask."""
    p = site_shift_params(site)
    if p["gamma"] == 1.0 and p["bias"] == 0.0 and p["blur"] == 0.0:
        return image  # reference scanner: leave untouched
    out = np.empty_like(image)
    for c in range(image.shape[0]):
        x = image[c]
        mask = x > 0                     # brain voxels; background stays 0
        if not mask.any():
            out[c] = x
            continue
        lo, hi = np.percentile(x[mask], (1.0, 99.0))
        rng = max(float(hi - lo), 1e-6)
        xn = np.clip((x - lo) / rng, 0.0, 1.0)
        xn = xn ** p["gamma"]            # nonlinear contrast (survives z-norm)
        if p["bias"]:
            xn = xn * (1.0 + p["bias"] * _bias_field(x.shape[0], x.shape[1], site))
        y = xn * rng + lo               # back to the original intensity scale
        if p["blur"]:
            y = gaussian_filter(y, sigma=p["blur"])
        out[c] = (y * mask).astype(np.float32)  # re-zero background after blur
    return out


@dataclass
class SiteShift:
    """Maps each case dir to its hospital's scanner profile. Deterministic, so the
    same case gets the same shift in EVERY method (centralized/local/FL/finetune),
    keeping per-hospital comparisons apples-to-apples."""

    case_site: dict  # case_dir -> 0-based site index

    def __call__(self, image: np.ndarray, case_dir: str) -> np.ndarray:
        site = self.case_site.get(case_dir)
        return image if site is None else apply_site_shift(image, site)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class BratsSliceDataset(Dataset):
    """Yields {"image": (4,H,W) float32, "label": (H,W) raw seg} for one slice.

    Raw seg keeps BraTS labels {0,1,2,4} and is left WITHOUT a channel dim, because
    ConvertToMultiChannelBasedOnBratsClassesd builds the 3 region-channels (TC, WT,
    ET) itself — giving it a channel dim would produce a spurious extra axis.
    """

    def __init__(self, index: list[tuple[str, int]], transform=None, site_shift=None):
        self.index = index
        self.transform = transform
        self.site_shift = site_shift  # optional SiteShift for the non-IID experiment

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        case_dir, z = self.index[i]
        # Lazy per-slice reads: dataobj[..., z] only pulls that slice off disk.
        channels = [
            np.asarray(nib.load(_modality_path(case_dir, m)).dataobj[..., z], dtype=np.float32)
            for m in MODALITIES
        ]
        image = np.stack(channels, axis=0)  # (4, H, W)
        if self.site_shift is not None:  # per-hospital scanner shift (before z-norm)
            image = self.site_shift(image, case_dir)
        label = np.asarray(nib.load(_seg_path(case_dir)).dataobj[..., z], dtype=np.float32)  # (H, W)

        sample = {"image": image, "label": label}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


# ----------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------

def train_transforms(size: int = 192):
    return Compose([
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),  # (1,H,W)->(3,H,W)
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Resized(keys=["image", "label"], spatial_size=(size, size), mode=("bilinear", "nearest")),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "label"], prob=0.3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
        ToTensord(keys=["image", "label"]),
    ])


def eval_transforms(size: int = 192):
    return Compose([
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        Resized(keys=["image", "label"], spatial_size=(size, size), mode=("bilinear", "nearest")),
        ToTensord(keys=["image", "label"]),
    ])


# ----------------------------------------------------------------------------
# Splitting helper
# ----------------------------------------------------------------------------

def case_split(cases, val_frac: float = 0.2, seed: int = 42):
    """Deterministically split a case list into (train_cases, val_cases) at the
    CASE level. The single source of truth for splits, so every method (local,
    FL, centralized) sees the exact same per-hospital validation cases."""
    cs = sorted(set(cases))
    rng = np.random.default_rng(seed)
    rng.shuffle(cs)
    n_val = max(1, int(len(cs) * val_frac))
    val = set(cs[:n_val])
    return [c for c in cs if c not in val], sorted(val)


def loaders_for_cases(cases, batch_size=8, size=192, workers=0, index_cache=None, val_frac=0.2,
                      site_shift=None):
    """Build (train_loader, val_loader, split) for one set of cases (e.g. a
    hospital), splitting it into train/val by case. Reused by local-only, the
    FLARE client, and fine-tune."""
    train_cases, val_cases = case_split(cases, val_frac)
    return loaders_from_case_lists(train_cases, val_cases, batch_size, size, workers, index_cache,
                                   site_shift=site_shift)


def loaders_from_case_lists(train_cases, val_cases, batch_size=8, size=192, workers=0, index_cache=None,
                            site_shift=None):
    """Build (train_loader, val_loader, split) from explicit train/val case lists.
    Used by the centralized run (train on the union of all hospitals' train cases,
    val on the union of their val cases — no leakage)."""
    train_index = build_slice_index(train_cases, cache_csv=index_cache)
    val_index = build_slice_index(val_cases, cache_csv=index_cache)
    train_ds = BratsSliceDataset(train_index, train_transforms(size), site_shift)
    val_ds = BratsSliceDataset(val_index, eval_transforms(size), site_shift)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=workers)
    return train_loader, val_loader, Split(train=train_index, val=val_index)


@dataclass
class Split:
    train: list[tuple[str, int]]
    val: list[tuple[str, int]]


def split_by_case(index: list[tuple[str, int]], val_frac: float = 0.2, seed: int = 42) -> Split:
    """Split a slice index by CASE (not slice) so slices from one patient never
    leak across sets. Uses the same `case_split` as everything else."""
    _, val_cases = case_split([c for c, _ in index], val_frac, seed)
    val_set = set(val_cases)
    train = [(c, z) for c, z in index if c not in val_set]
    val = [(c, z) for c, z in index if c in val_set]
    return Split(train=train, val=val)
