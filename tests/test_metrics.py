"""Smoke tests for fedbrats.metrics.dice_binary."""

import numpy as np

from fedbrats.metrics import dice_binary


def test_perfect_prediction():
    """All 1s vs all 1s → Dice = 1.0."""
    pred = np.ones((4, 4), dtype=np.uint8)
    gt = np.ones((4, 4), dtype=np.uint8)
    assert dice_binary(pred, gt) == 1.0


def test_worst_prediction():
    """All 1s vs all 0s → Dice = 0.0."""
    pred = np.ones((4, 4), dtype=np.uint8)
    gt = np.zeros((4, 4), dtype=np.uint8)
    assert dice_binary(pred, gt) == 0.0


def test_both_empty():
    """Empty prediction and empty ground truth → Dice = 1.0 (BraTS convention)."""
    pred = np.zeros((4, 4), dtype=np.uint8)
    gt = np.zeros((4, 4), dtype=np.uint8)
    assert dice_binary(pred, gt) == 1.0


def test_partial_overlap():
    """Partial overlap yields Dice between 0 and 1."""
    pred = np.zeros((4, 4), dtype=np.uint8)
    gt = np.zeros((4, 4), dtype=np.uint8)
    # pred covers top-left 2x2, gt covers top-left 2x4
    pred[:2, :2] = 1  # 4 voxels
    gt[:2, :] = 1     # 8 voxels
    # intersection = 4, dice = 2*4 / (4+8) = 8/12 = 2/3
    expected = 2.0 * 4 / (4 + 8)
    assert abs(dice_binary(pred, gt) - expected) < 1e-7


def test_output_is_float():
    """dice_binary must return a Python float."""
    pred = np.ones((2, 2), dtype=np.uint8)
    gt = np.ones((2, 2), dtype=np.uint8)
    result = dice_binary(pred, gt)
    assert isinstance(result, float)
