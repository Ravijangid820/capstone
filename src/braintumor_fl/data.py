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
# Dataset
# ----------------------------------------------------------------------------

class BratsSliceDataset(Dataset):
    """Yields {"image": (4,H,W) float32, "label": (H,W) raw seg} for one slice.

    Raw seg keeps BraTS labels {0,1,2,4} and is left WITHOUT a channel dim, because
    ConvertToMultiChannelBasedOnBratsClassesd builds the 3 region-channels (TC, WT,
    ET) itself — giving it a channel dim would produce a spurious extra axis.
    """

    def __init__(self, index: list[tuple[str, int]], transform=None):
        self.index = index
        self.transform = transform

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


def loaders_for_cases(cases, batch_size=8, size=192, workers=0, index_cache=None, val_frac=0.2):
    """Build (train_loader, val_loader, split) for one set of cases (e.g. a
    hospital), splitting it into train/val by case. Reused by local-only, the
    FLARE client, and fine-tune."""
    train_cases, val_cases = case_split(cases, val_frac)
    return loaders_from_case_lists(train_cases, val_cases, batch_size, size, workers, index_cache)


def loaders_from_case_lists(train_cases, val_cases, batch_size=8, size=192, workers=0, index_cache=None):
    """Build (train_loader, val_loader, split) from explicit train/val case lists.
    Used by the centralized run (train on the union of all hospitals' train cases,
    val on the union of their val cases — no leakage)."""
    train_index = build_slice_index(train_cases, cache_csv=index_cache)
    val_index = build_slice_index(val_cases, cache_csv=index_cache)
    train_ds = BratsSliceDataset(train_index, train_transforms(size))
    val_ds = BratsSliceDataset(val_index, eval_transforms(size))
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
