"""Local-only baseline (the FLOOR): each hospital trains its OWN model on ONLY
its own data, no collaboration. Shows what a hospital gets without federation.

Pure PyTorch/MONAI (no FLARE) — runs anywhere.

    uv run python fl/run_local_baselines.py --data-root data/BraTS2021_Training_Data \
        --n-clients 6 --epochs 40           # even split
    uv run python fl/run_local_baselines.py --data-root data/BraTS2021_Training_Data \
        --fets-csv data/fets_partitioning.csv --epochs 40   # real hospitals

Set --epochs ~= (FL rounds x FL local-epochs) so the compute budget is comparable.
"""

from __future__ import annotations

import argparse
import os

import torch

from braintumor_fl.data import loaders_for_cases
from braintumor_fl.model import BratsUNet, build_loss, build_metric
from braintumor_fl.partition import get_partitions
from braintumor_fl.results import write_scores
from braintumor_fl.trainer import evaluate, get_device, train_one_epoch


def train_one_client(cases, args, device, site) -> dict:
    train_loader, val_loader, split = loaders_for_cases(
        cases, args.batch_size, args.size, args.workers,
        index_cache=os.path.join(args.results_dir, args.method, f"_index_{site}.csv"),
    )
    model = BratsUNet().to(device)
    loss_fn = build_loss()
    metric = build_metric()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best, best_scores = 0.0, None
    for _ in range(args.epochs):
        train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, args.grad_clip)
        scheduler.step()
        scores = evaluate(model, val_loader, metric, device)
        if scores["mean"] > best:
            best, best_scores = scores["mean"], scores
            torch.save(model.state_dict(), os.path.join(args.results_dir, args.method, f"{site}.pt"))

    write_scores(args.results_dir, args.method, site, best_scores,
                 len(split.train), len(split.val))
    return best_scores


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--n-clients", type=int, default=0)
    p.add_argument("--fets-csv", default="")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--method", default="local")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    device = get_device()
    parts = get_partitions(args.data_root, args.n_clients or None,
                           args.fets_csv or None, args.max_cases)
    os.makedirs(os.path.join(args.results_dir, args.method), exist_ok=True)
    print(f"[local] {len(parts)} hospitals | {args.epochs} epochs each | device={device}")

    for i, cases in enumerate(parts):
        site = f"site-{i + 1}"
        scores = train_one_client(cases, args, device, site)
        print(f"[local] {site}: Dice mean={scores['mean']:.4f} "
              f"(TC={scores['TC']:.3f} WT={scores['WT']:.3f} ET={scores['ET']:.3f})")


if __name__ == "__main__":
    main()
