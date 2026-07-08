"""Dice on the three BraTS regions, computed per **volume** (never per slice).

Two conventions that materially change the numbers:

1. **Per-case, not per-slice.** The mean of per-slice Dice is not the per-case Dice, and it
   inflates scores (empty slices score 1.0 for free). We predict every slice, stack into a
   volume, and score the volume. Then average across cases.

2. **Empty ground truth.** ET is genuinely absent in some cases, making Dice 0/0. The BraTS
   convention: an empty prediction on empty GT scores 1.0; any false positive scores 0.0.
   Ignoring this makes the ET column meaningless.
"""

from __future__ import annotations

import numpy as np

from .config import REGIONS


def dice_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice for one binary region of one case. Follows the BraTS empty-GT convention."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    p, g = int(pred.sum()), int(gt.sum())
    if g == 0:
        return 1.0 if p == 0 else 0.0
    inter = int(np.logical_and(pred, gt).sum())
    return 2.0 * inter / (p + g)


def dice_regions(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """(3, ...) predicted binary masks vs (3, ...) ground truth -> {'wt','tc','et'}."""
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    return {r: dice_binary(pred[i], gt[i]) for i, r in enumerate(REGIONS)}


def mean_regions(per_case: list[dict[str, float]]) -> dict[str, float]:
    """Average per-case Dice dicts into one dict. Empty input -> NaNs."""
    if not per_case:
        return {r: float("nan") for r in REGIONS}
    return {r: float(np.mean([d[r] for d in per_case])) for r in REGIONS}
