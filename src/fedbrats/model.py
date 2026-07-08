"""Model: a dimension-parametric U-Net with BatchNorm, plus BN-key identification for FedBN.

BatchNorm is not incidental -- it is the thing FedBN keeps local. The three region channels
(WT, TC, ET) are **nested and overlapping** (ET subset of TC subset of WT), so this is a multi-label
problem: independent sigmoids, never a softmax. See docs/specs.md §2.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.layers import Norm
from monai.networks.nets import UNet

from .config import Config


class BratsUNet(nn.Module):
    """MONAI U-Net behind a stable, reconstructable class.

    The wrapper is deliberate: NVIDIA FLARE reconstructs the model from a class reference on the
    client side, and a bare `monai...UNet` with kwargs proved fragile there (commit f1bef2f). It
    costs nothing now and the FLARE port needs it.
    """

    def __init__(self, dim: str = "2d", in_channels: int = 4, out_channels: int = 3,
                 base: int = 32, num_res_units: int = 2):
        super().__init__()
        self.dim = dim
        self.base = base
        spatial_dims = 2 if dim == "2d" else 3
        self.net = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(base, base * 2, base * 4, base * 8),
            strides=(2, 2, 2),                 # 3 downsamples -> inputs must be /8; we pad to /16
            num_res_units=num_res_units,
            norm=Norm.BATCH,                   # <- what FedBN keeps local
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)     # raw logits; loss applies sigmoid


def build_model(cfg: Config) -> BratsUNet:
    """Deterministic construction: same seed -> identical init across every method."""
    torch.manual_seed(cfg.seed)
    return BratsUNet(dim=cfg.dim, base=cfg.base_channels)


def bn_keys(model: nn.Module) -> set[str]:
    """state_dict keys belonging to BatchNorm layers: affine (weight/bias) AND buffers
    (running_mean, running_var, num_batches_tracked).

    Identified by **module type**, not by name. MONAI emits keys like
    `net.model.0.conv.unit0.adn.N.weight` -- there is no "bn" substring to match on, so
    name-based matching fails silently and collapses FedBN into FedAvg.
    """
    keys: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            prefix = f"{name}." if name else ""
            keys.update(prefix + k for k in module.state_dict())
    return keys
