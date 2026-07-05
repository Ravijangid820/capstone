"""Personalization strategies — the core research contribution.

Every strategy is implemented the same clean way: the server always does plain
FedAvg (averages whatever clients send), and personalization is enforced
*client-side* by choosing which parameters to KEEP LOCAL when the global model
arrives. A kept-local parameter is never overwritten by the global average, so it
specializes to that hospital across rounds.

Strategies:
- "fedavg"        -> keep nothing local (standard global model)
- "fedbn"         -> keep normalization layers local (scanner calibration stays home)
- "personal_head" -> keep the output/head conv local (shared body, private head)

The client model object persists across rounds (it lives in the while-loop), so
kept-local params naturally accumulate their local training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.GroupNorm, nn.LayerNorm,
)


def norm_param_names(model: nn.Module) -> set[str]:
    """State-dict keys of all normalization layers (params AND running buffers).

    Detected by module *type*, so it's robust to architecture/naming.
    """
    names: set[str] = set()
    for mod_name, module in model.named_modules():
        if isinstance(module, _NORM_TYPES):
            prefix = f"{mod_name}." if mod_name else ""
            for pname, _ in module.named_parameters(recurse=False):
                names.add(prefix + pname)
            for bname, _ in module.named_buffers(recurse=False):
                names.add(prefix + bname)  # running_mean / running_var / num_batches_tracked
    return names


def head_param_names(model: nn.Module, out_channels: int = 3) -> set[str]:
    """State-dict keys of the final segmentation head (the last conv that emits
    `out_channels`). Used for the shared-body / personal-head strategy."""
    last = None
    for mod_name, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and module.out_channels == out_channels:
            last = mod_name
    if last is None:
        return set()
    return {k for k in model.state_dict() if k == f"{last}.weight" or k == f"{last}.bias"}


def keep_local_keys(model: nn.Module, strategy: str, out_channels: int = 3) -> set[str]:
    """Which state-dict keys this client keeps local (not overwritten by global)."""
    if strategy == "fedavg":
        return set()
    if strategy == "fedbn":
        return norm_param_names(model)
    if strategy == "personal_head":
        return norm_param_names(model) | head_param_names(model, out_channels)
    raise ValueError(f"unknown personalization strategy: {strategy!r}")


def load_global(model: nn.Module, global_params: dict, keep_local: set[str]) -> None:
    """Load global params into `model`, but leave `keep_local` keys untouched.

    Robust to a global payload that omits some keys (e.g. num_batches_tracked):
    anything missing from `global_params` keeps its current local value.
    """
    own = model.state_dict()
    merged = {}
    for k, v in own.items():
        if k not in keep_local and k in global_params:
            g = global_params[k]
            merged[k] = g if torch.is_tensor(g) else torch.as_tensor(g)
        else:
            merged[k] = v
    model.load_state_dict(merged, strict=True)


@torch.no_grad()
def clone_state(model: nn.Module) -> dict:
    """CPU copy of the current weights (for the FedProx proximal term)."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
