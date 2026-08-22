"""One local training loop and one full-volume evaluation loop.

Shared by every method: centralized, local-only, FedAvg and FedBN all call exactly this code for
the "train on some data / score some data" part. The only thing that differs between methods is
what happens *between* calls -- see federated.py.

Precision is fp32 throughout. AMP/fp16 corrupts BatchNorm running statistics on the strongly
shifted data (commit 854f5fe) -- and BN statistics are the entire mechanism of FedBN.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from torch.utils.data import DataLoader

from .config import Config
from .data import build_dataset, load_cached_case, pad_to_multiple
from .metrics import dice_regions, mean_regions


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DiceBCELoss(nn.Module):
    """Soft Dice + BCE on independent sigmoids (regions overlap; softmax would be wrong)."""

    def __init__(self):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice(logits, target) + self.bce(logits, target)


def make_loader(cfg: Config, case_ids: list[str], index: dict, shuffle: bool = True) -> DataLoader:
    ds = build_dataset(cfg, case_ids, index)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )


def train_epochs(model: nn.Module, loader: DataLoader, epochs: int, cfg: Config,
                 device: torch.device, lr: float | None = None) -> float:
    """Train in place for `epochs`. Returns mean loss over the final epoch.

    A fresh optimizer per call: FedAvg/FedBN do not transmit optimizer state between rounds,
    so each round's local training starts from a clean Adam. This is the standard formulation.

    `lr` is supplied per round by the caller rather than read from `cfg.lr`. Because the
    optimizer is rebuilt every round, a constant LR means the model never anneals -- it takes
    full-size steps around the optimum forever, and the resulting round-to-round swing is larger
    than the differences between the methods being compared. Decaying across rounds is the only
    place an FL schedule can live when no optimizer state survives a round.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr if lr is None else lr)
    criterion = DiceBCELoss()

    last = float("nan")
    for _ in range(epochs):
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * x.shape[0]
            n += x.shape[0]
        last = total / max(n, 1)
    return last


def _tta_flips(n_spatial: int) -> list[tuple[int, ...]]:
    """Flip combinations for test-time augmentation: identity plus each single spatial axis.

    Single-axis flips only. The full 2^n power set costs 8x in 3D for a gain that has long since
    saturated -- averaging 4 views already removes most of the variance that flip-TTA can remove.
    Axes are given relative to the spatial block and offset by the caller.
    """
    return [()] + [(a,) for a in range(n_spatial)]


@torch.no_grad()
def _probs_2d(model: nn.Module, padded: np.ndarray, device: torch.device,
              tta: bool, batch: int = 8) -> np.ndarray:
    """Sigmoid probabilities for a (Z, 4, Hp, Wp) stack, optionally averaged over flip views."""
    views = _tta_flips(2) if tta else [()]
    acc = None
    for flip in views:
        axes = tuple(2 + a for a in flip)                          # skip (Z, C)
        src = np.ascontiguousarray(np.flip(padded, axis=axes)) if axes else padded
        out = []
        for i in range(0, src.shape[0], batch):
            chunk = torch.from_numpy(src[i:i + batch]).to(device)
            out.append(torch.sigmoid(model(chunk)).cpu().numpy())
        probs = np.concatenate(out, axis=0)
        if axes:
            probs = np.flip(probs, axis=axes)                      # undo: back to input frame
        acc = probs if acc is None else acc + probs
    return acc / len(views)


@torch.no_grad()
def _probs_3d(model: nn.Module, xt: torch.Tensor, cfg: Config, tta: bool) -> torch.Tensor:
    """Sigmoid probabilities for a (1, 4, X, Y, Z) volume via sliding window, optional flip-TTA."""
    views = _tta_flips(3) if tta else [()]
    acc = None
    for flip in views:
        dims = tuple(2 + a for a in flip)                          # skip (N, C)
        src = torch.flip(xt, dims=dims) if dims else xt
        logits = sliding_window_inference(
            src, roi_size=(cfg.patch_size,) * 3, sw_batch_size=cfg.sw_batch_size,
            predictor=model, overlap=cfg.sw_overlap,
        )
        probs = torch.sigmoid(logits)
        if dims:
            probs = torch.flip(probs, dims=dims)
        acc = probs if acc is None else acc + probs
    return acc / len(views)


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Drop 3D connected components smaller than `min_voxels` from one binary region.

    A 2D model predicts each slice independently, so its false positives are typically a handful
    of voxels on one slice with nothing above or below them. Real tumour is contiguous in z.
    Scoring in 3D therefore lets a 3D connectivity test remove exactly the errors the 2D model
    is structurally prone to.

    Guarded against erasing everything: if no component survives the threshold, the largest is
    kept. Emptying a mask is scored 0.0 against non-empty ground truth, so an over-eager filter
    on a weak case costs more than the false positives it removes.
    """
    if min_voxels <= 0 or not mask.any():
        return mask
    from scipy import ndimage
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0                                                   # background is not a component
    keep = np.flatnonzero(sizes >= min_voxels)
    if keep.size == 0:
        keep = np.array([int(sizes.argmax())])
    return np.isin(labels, keep)


def postprocess(pred: np.ndarray, cfg: Config) -> np.ndarray:
    """Clean up a (3, X, Y, Z) prediction, then restore WT >= TC >= ET nesting.

    The three channels are predicted by independent sigmoids, so nothing in the model enforces
    the containment the BraTS regions are defined by. Filtering each channel separately can
    leave an ET voxel outside its own TC -- an anatomically impossible prediction. Intersecting
    downward afterwards costs nothing and keeps every output well formed.
    """
    if cfg.postproc_min_voxels <= 0:
        return pred
    out = np.stack([remove_small_components(pred[i].astype(bool), cfg.postproc_min_voxels)
                    for i in range(pred.shape[0])])
    out[1] &= out[0]                                               # TC within WT
    out[2] &= out[1]                                               # ET within TC
    return out


@torch.no_grad()
def predict_volume(model: nn.Module, x: np.ndarray, cfg: Config,
                   device: torch.device, tta: bool = False) -> np.ndarray:
    """Predict binary (3, X, Y, Z) masks for one whole case."""
    model.to(device).eval()

    if cfg.is_2d:
        # x: (4, H, W, Z). Move the slice axis to the front FIRST so that padding the two
        # trailing axes pads (H, W) -- the in-plane dims the U-Net downsamples -- and not (W, Z).
        slices = np.ascontiguousarray(np.moveaxis(np.asarray(x, dtype=np.float32), 3, 0))
        padded, crop = pad_to_multiple(slices, 16, n_spatial=2)    # (Z, 4, Hp, Wp)
        probs = _probs_2d(model, padded, device, tta, cfg.eval_batch_size)   # (Z,3,Hp,Wp)
        pred = (probs > 0.5)[(slice(None), slice(None)) + crop]    # (Z, 3, H, W)
        pred = np.moveaxis(pred, 0, 3)                             # (3, H, W, Z)
        return postprocess(pred, cfg)

    # 3D: sliding-window over the volume at the training patch size.
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))[None].to(device)   # (1,4,X,Y,Z)
    probs = _probs_3d(model, xt, cfg, tta)
    return postprocess((probs[0] > 0.5).cpu().numpy(), cfg)


def evaluate_cases(model: nn.Module, cfg: Config, case_ids: list[str], device: torch.device,
                   tta: bool = False) -> tuple[dict[str, float], list[dict]]:
    """Full-volume Dice over a set of cases. Returns (mean over cases, per-case dicts).

    The per-case list is what makes a difference between two methods testable: 62 paired
    observations per hospital support a confidence interval and a signed-rank test, where the
    mean alone supports only an assertion that one number is bigger than another.
    """
    per_case = []
    for cid in case_ids:
        x, y = load_cached_case(cfg, cid)
        pred = predict_volume(model, x, cfg, device, tta=tta)
        per_case.append({"case_id": cid, **dice_regions(pred, np.asarray(y))})
        del x, y                     # release the memmap: Windows locks open files
    return mean_regions(per_case), per_case
