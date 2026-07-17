"""Smoke tests for fedbrats.model.build_model and bn_keys."""

import torch.nn as nn

from fedbrats.config import Config
from fedbrats.model import bn_keys, build_model


def test_build_model_returns_module_2d():
    """build_model with 2D config returns a torch.nn.Module."""
    cfg = Config(dim="2d", device="cpu")
    model = build_model(cfg)
    assert isinstance(model, nn.Module)


def test_build_model_returns_module_3d():
    """build_model with 3D config returns a torch.nn.Module."""
    cfg = Config(dim="3d", device="cpu")
    model = build_model(cfg)
    assert isinstance(model, nn.Module)


def test_bn_keys_non_empty_2d():
    """bn_keys returns a non-empty set of strings for a 2D model."""
    cfg = Config(dim="2d", device="cpu")
    model = build_model(cfg)
    keys = bn_keys(model)
    assert len(keys) > 0
    assert all(isinstance(k, str) for k in keys)


def test_bn_keys_non_empty_3d():
    """bn_keys returns a non-empty set of strings for a 3D model."""
    cfg = Config(dim="3d", device="cpu")
    model = build_model(cfg)
    keys = bn_keys(model)
    assert len(keys) > 0
    assert all(isinstance(k, str) for k in keys)


def test_bn_keys_are_valid_state_dict_keys():
    """Every key returned by bn_keys must exist in the model's state_dict."""
    cfg = Config(dim="2d", device="cpu")
    model = build_model(cfg)
    sd_keys = set(model.state_dict().keys())
    for k in bn_keys(model):
        assert k in sd_keys, f"bn_key '{k}' not found in state_dict"
