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
                 device: torch.device) -> float:
    """Train in place for `epochs`. Returns mean loss over the final epoch.

    A fresh optimizer per call: FedAvg/FedBN do not transmit optimizer state between rounds,
    so each round's local training starts from a clean Adam. This is the standard formulation.
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
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


@torch.no_grad()
def predict_volume(model: nn.Module, x: np.ndarray, cfg: Config,
                   device: torch.device) -> np.ndarray:
    """Predict binary (3, X, Y, Z) masks for one whole case."""
    model.to(device).eval()

    if cfg.is_2d:
        # x: (4, H, W, Z). Move the slice axis to the front FIRST so that padding the two
        # trailing axes pads (H, W) -- the in-plane dims the U-Net downsamples -- and not (W, Z).
        slices = np.ascontiguousarray(np.moveaxis(np.asarray(x, dtype=np.float32), 3, 0))
        padded, crop = pad_to_multiple(slices, 16, n_spatial=2)    # (Z, 4, Hp, Wp)
        out = []
        for i in range(0, padded.shape[0], 8):
            batch = torch.from_numpy(padded[i:i + 8]).to(device)
            probs = torch.sigmoid(model(batch))
            out.append((probs > 0.5).cpu().numpy())
        pred = np.concatenate(out, axis=0)                         # (Z, 3, Hp, Wp)
        pred = pred[(slice(None), slice(None)) + crop]             # (Z, 3, H, W)
        return np.moveaxis(pred, 0, 3)                             # (3, H, W, Z)

    # 3D: sliding-window over the volume at the training patch size.
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32))[None].to(device)   # (1,4,X,Y,Z)
    logits = sliding_window_inference(
        xt, roi_size=(cfg.patch_size,) * 3, sw_batch_size=1, predictor=model, overlap=0.25
    )
    return (torch.sigmoid(logits)[0] > 0.5).cpu().numpy()


def evaluate_cases(model: nn.Module, cfg: Config, case_ids: list[str],
                   device: torch.device) -> tuple[dict[str, float], list[dict]]:
    """Full-volume Dice over a set of cases. Returns (mean over cases, per-case dicts)."""
    per_case = []
    for cid in case_ids:
        x, y = load_cached_case(cfg, cid)
        pred = predict_volume(model, x, cfg, device)
        per_case.append(dice_regions(pred, np.asarray(y)))
        del x, y                     # release the memmap: Windows locks open files
    return mean_regions(per_case), per_case
