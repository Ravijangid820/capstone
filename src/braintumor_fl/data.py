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
from torch.utils.data import Dataset

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

    Reads only the (small) seg volumes. Result is cached to CSV if given.
    """
    if cache_csv and os.path.exists(cache_csv):
        with open(cache_csv, newline="") as f:
            return [(row[0], int(row[1])) for row in csv.reader(f)]

    index: list[tuple[str, int]] = []
    for case_dir in cases:
        seg = np.asarray(nib.load(_seg_path(case_dir)).dataobj)  # (H, W, D)
        per_slice = (seg > 0).sum(axis=(0, 1))  # tumor pixels per axial slice
        for z in np.where(per_slice >= min_tumor_pixels)[0]:
            index.append((case_dir, int(z)))

    if cache_csv:
        os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
        with open(cache_csv, "w", newline="") as f:
            csv.writer(f).writerows(index)
    return index


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class BratsSliceDataset(Dataset):
    """Yields {"image": (4,H,W) float32, "label": (1,H,W) raw seg} for one slice.

    Raw seg keeps BraTS labels {0,1,2,4}; the transform pipeline converts them to
    the 3 overlapping regions (TC, WT, ET).
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
        seg = np.asarray(nib.load(_seg_path(case_dir)).dataobj[..., z], dtype=np.float32)
        label = seg[np.newaxis]  # (1, H, W)

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

@dataclass
class Split:
    train: list[tuple[str, int]]
    val: list[tuple[str, int]]


def split_by_case(index: list[tuple[str, int]], val_frac: float = 0.2, seed: int = 42) -> Split:
    """Split by CASE (not slice) so slices from one patient never leak across sets."""
    cases = sorted({c for c, _ in index})
    rng = np.random.default_rng(seed)
    rng.shuffle(cases)
    n_val = max(1, int(len(cases) * val_frac))
    val_cases = set(cases[:n_val])
    train = [(c, z) for c, z in index if c not in val_cases]
    val = [(c, z) for c, z in index if c in val_cases]
    return Split(train=train, val=val)
