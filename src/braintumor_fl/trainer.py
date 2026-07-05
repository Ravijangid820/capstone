"""Shared training/eval code used by BOTH the centralized baseline and the FLARE
federated client, so there is exactly one training code path.

- `train_one_epoch` — one pass, with AMP + NaN-skip + grad clipping (the Phase-1
  stabilizers). The centralized loop calls this per epoch.
- `local_train` — run several epochs; this is what a FLARE client does each round
  after receiving the global weights.
- `evaluate` — per-region Dice on a loader.
"""

from __future__ import annotations

import torch

from . import REGIONS
from .model import build_loss, build_metric, logits_to_preds


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prox_term(model, global_params, mu: float):
    """FedProx penalty: (mu/2) * sum ||w - w_global||^2 over current params."""
    if mu <= 0 or global_params is None:
        return 0.0
    penalty = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad and name in global_params:
            penalty = penalty + ((p - global_params[name]) ** 2).sum()
    return (mu / 2.0) * penalty


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, grad_clip: float = 1.0,
                    prox_mu: float = 0.0, global_params=None):
    """One epoch. Returns (avg_loss, n_batches_ok, n_skipped).

    If prox_mu > 0, adds the FedProx proximal term pulling weights toward the
    round's starting global weights (global_params: name -> tensor on device).
    """
    model.train()
    running, n_ok, skipped = 0.0, 0, 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss = loss_fn(model(images), labels)
            if prox_mu > 0:
                loss = loss + _prox_term(model, global_params, prox_mu)
        if not torch.isfinite(loss):  # never let a bad batch corrupt the weights
            skipped += 1
            continue
        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        running += loss.item()
        n_ok += 1
    return running / max(1, n_ok), n_ok, skipped


def local_train(model, loader, epochs: int, lr: float, device, grad_clip: float = 1.0,
                prox_mu: float = 0.0, global_params=None):
    """Train `model` in place for `epochs`. Used by a FLARE client each round.

    For FedProx, pass prox_mu > 0 and global_params (name -> tensor, moved to
    `device`) captured from the round's incoming global model.
    """
    model.to(device)
    loss_fn = build_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    if global_params is not None:
        global_params = {k: v.to(device) for k, v in global_params.items()}
    for _ in range(epochs):
        train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, grad_clip,
                        prox_mu=prox_mu, global_params=global_params)
    return model


@torch.no_grad()
def evaluate(model, loader, metric=None, device=None) -> dict[str, float]:
    """Per-region + mean Dice over a loader."""
    device = device or get_device()
    metric = metric or build_metric()
    model.to(device)
    model.eval()
    metric.reset()
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        metric(y_pred=logits_to_preds(logits), y=labels)
    per_region = metric.aggregate()
    scores = {r: float(per_region[i]) for i, r in enumerate(REGIONS)}
    scores["mean"] = float(per_region.mean())
    return scores
