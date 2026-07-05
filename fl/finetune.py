"""Fine-tune personalization: take the converged FedAvg GLOBAL model and train it
a few more epochs on each hospital's own data. A simple, strong personalization
baseline to compare against FedBN.

The FedAvg run saved the global model at results/fedavg/site-1.pt (for method
'fedavg' every site stores the same global weights). We load that and fine-tune.

    uv run python fl/finetune.py --data-root data/BraTS2021_Training_Data \
        --fets-csv data/fets_partitioning.csv --ft-epochs 5
"""

from __future__ import annotations

import argparse
import os

import torch

from braintumor_fl.data import loaders_for_cases
from braintumor_fl.model import build_loss, build_metric, build_unet
from braintumor_fl.partition import get_partitions
from braintumor_fl.results import write_scores
from braintumor_fl.trainer import evaluate, get_device, train_one_epoch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--n-clients", type=int, default=0)
    p.add_argument("--fets-csv", default="")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--global-model", default="results/fedavg/site-1.pt",
                   help="converged FedAvg global weights to start from")
    p.add_argument("--ft-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4, help="lower LR for fine-tuning")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--method", default="finetune")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    device = get_device()
    global_state = torch.load(args.global_model, map_location=device)
    parts = get_partitions(args.data_root, args.n_clients or None,
                           args.fets_csv or None, args.max_cases)
    os.makedirs(os.path.join(args.results_dir, args.method), exist_ok=True)
    print(f"[finetune] {len(parts)} hospitals from {args.global_model} | {args.ft_epochs} epochs")

    loss_fn = build_loss()
    metric = build_metric()
    for i, cases in enumerate(parts):
        site = f"site-{i + 1}"
        train_loader, val_loader, split = loaders_for_cases(
            cases, args.batch_size, args.size, args.workers,
            index_cache=os.path.join(args.results_dir, args.method, f"_index_{site}.csv"),
        )
        model = build_unet().to(device)
        model.load_state_dict(global_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

        best, best_scores = 0.0, evaluate(model, val_loader, metric, device)  # start = global
        for _ in range(args.ft_epochs):
            train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, args.grad_clip)
            scores = evaluate(model, val_loader, metric, device)
            if scores["mean"] > best:
                best, best_scores = scores["mean"], scores
                torch.save(model.state_dict(), os.path.join(args.results_dir, args.method, f"{site}.pt"))

        write_scores(args.results_dir, args.method, site, best_scores,
                     len(split.train), len(split.val))
        print(f"[finetune] {site}: Dice mean={best_scores['mean']:.4f}")


if __name__ == "__main__":
    main()
