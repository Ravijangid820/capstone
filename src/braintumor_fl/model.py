"""Model, loss, and metric for 2D BraTS segmentation.

3 output channels = the 3 overlapping BraTS regions (TC, WT, ET), so this is a
multi-label problem: sigmoid activation + per-channel Dice (not softmax).
"""

from __future__ import annotations

import torch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet


def build_unet(in_channels: int = 4, out_channels: int = 3, norm: str = "batch") -> UNet:
    """2D U-Net sized to fit a 4GB GPU comfortably at 192x192.

    Defaults to BatchNorm (`norm="batch"`) because it's the setting FedBN targets:
    BatchNorm's running stats capture each scanner's intensity fingerprint, so
    keeping them local per hospital IS the personalization. All experiments
    (centralized, local, FedAvg, FedBN) use the same norm for a fair comparison.
    """
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=norm,
    )


def build_loss() -> DiceLoss:
    # sigmoid=True because regions overlap (multi-label), not mutually exclusive.
    return DiceLoss(sigmoid=True)


def build_metric() -> DiceMetric:
    # Mean Dice per region; get_not_nans lets us average safely over batches.
    return DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)


def logits_to_preds(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) >= threshold).float()
