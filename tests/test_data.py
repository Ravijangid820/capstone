"""Smoke tests for fedbrats.data.labels_to_regions."""

import numpy as np

from fedbrats.data import labels_to_regions


def test_label_1_maps_to_wt_only():
    """BraTS label 1 → WT=1, TC=1, ET=0."""
    seg = np.array([[[1]]], dtype=np.uint8)  # (1,1,1)
    regions = labels_to_regions(seg)
    # regions shape: (3, 1, 1, 1) — [WT, TC, ET]
    assert regions[0, 0, 0, 0] == 1  # WT includes label 1
    assert regions[1, 0, 0, 0] == 1  # TC includes label 1
    assert regions[2, 0, 0, 0] == 0  # ET does NOT include label 1


def test_label_4_maps_to_all_regions():
    """BraTS label 4 → WT=1, TC=1, ET=1."""
    seg = np.array([[[4]]], dtype=np.uint8)
    regions = labels_to_regions(seg)
    assert regions[0, 0, 0, 0] == 1  # WT includes label 4
    assert regions[1, 0, 0, 0] == 1  # TC includes label 4
    assert regions[2, 0, 0, 0] == 1  # ET includes label 4


def test_label_2_maps_to_wt_tc():
    """BraTS label 2 → WT=1, TC=0, ET=0."""
    seg = np.array([[[2]]], dtype=np.uint8)
    regions = labels_to_regions(seg)
    assert regions[0, 0, 0, 0] == 1  # WT includes label 2
    assert regions[1, 0, 0, 0] == 0  # TC does NOT include label 2
    assert regions[2, 0, 0, 0] == 0  # ET does NOT include label 2


def test_output_shape():
    """Output shape is (3, *input_shape)."""
    seg = np.zeros((10, 12, 14), dtype=np.uint8)
    regions = labels_to_regions(seg)
    assert regions.shape == (3, 10, 12, 14)
