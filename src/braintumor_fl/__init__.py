"""Personalized Federated Learning for 2D brain tumor segmentation (BraTS/FeTS).

Package layout:
    data.py               - BraTS slice index + on-the-fly 2D dataset + transforms
    model.py              - MONAI U-Net, Dice loss/metric, BraTS region helpers
    train_centralized.py  - Phase 1: single-GPU centralized baseline
"""

__version__ = "0.1.0"

MODALITIES = ("flair", "t1", "t1ce", "t2")  # fixed 4-channel input order
REGIONS = ("TC", "WT", "ET")  # BraTS eval regions: tumor core, whole tumor, enhancing
