"""Centralized training — the CEILING reference (train on all hospitals' data
pooled) and the Phase-1 pipeline check. Uses the shared `trainer` so the training
code is identical to the federated clients.

Quick Phase-1 check (any cases):
    uv run python -m braintumor_fl.train_centralized --data-root data/BraTS2021_Training_Data \
        --max-cases 100 --epochs 20

Final ceiling (partition-consistent: trains on the UNION of hospital train-cases,
never on any hospital's val cases -> no leakage into the comparison):
    uv run python -m braintumor_fl.train_centralized --data-root data/BraTS2021_Training_Data \
        --fets-csv data/fets_partitioning.csv --epochs 40 --out data/centralized_unet.pt
"""

from __future__ import annotations

import argparse
import os

import torch

from .data import SiteShift, case_split, find_cases, loaders_from_case_lists
from .model import BratsUNet, build_loss, build_metric
from .partition import case_site_map, get_partitions, partitioned_splits
from .trainer import evaluate, get_device, train_one_epoch


def build_training_loaders(args):
    """Centralized train/val loaders. If partition args are given, train on the
    union of hospital train-cases and validate on the union of their val-cases."""
    site_shift = None
    if args.fets_csv or args.n_clients:
        parts = get_partitions(
            os.path.abspath(args.data_root),
            args.n_clients or None,
            os.path.abspath(args.fets_csv) if args.fets_csv else None,
            args.max_cases,
        )
        all_train, all_val, _ = partitioned_splits(parts, args.val_frac)
        if args.synthetic_shift:  # each pooled case keeps ITS hospital's scanner shift
            site_shift = SiteShift(case_site_map(parts))
    else:
        if args.synthetic_shift:
            raise SystemExit("--synthetic-shift requires --n-clients or --fets-csv (needs hospitals)")
        cases = find_cases(args.data_root)
        if args.max_cases:
            cases = cases[: args.max_cases]
        all_train, all_val = case_split(cases, args.val_frac)
    return loaders_from_case_lists(
        all_train, all_val, args.batch_size, args.size, args.workers, args.index_cache,
        site_shift=site_shift,
    )


def train(args) -> None:
    device = get_device()
    print(f"[setup] device={device} | data_root={args.data_root}")

    if not find_cases(args.data_root):
        raise SystemExit(f"No BraTS cases under {args.data_root!r} (see DATA.md).")

    train_loader, val_loader, split = build_training_loaders(args)
    print(f"[data] slices: {len(split.train)} train / {len(split.val)} val")

    model = BratsUNet().to(device)
    loss_fn = build_loss()
    metric = build_metric()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        avg_loss, _, skipped = train_one_epoch(
            model, train_loader, optimizer, scaler, loss_fn, device, args.grad_clip
        )
        scheduler.step()  # cosine LR decay -> stable late training (no NaN blow-ups)
        scores = evaluate(model, val_loader, metric, device)
        skip_note = f" | skipped {skipped}" if skipped else ""
        print(f"[epoch {epoch}] loss={avg_loss:.4f} | Dice mean={scores['mean']:.4f} "
              f"(TC={scores['TC']:.3f} WT={scores['WT']:.3f} ET={scores['ET']:.3f}){skip_note}")
        if scores["mean"] > best:
            best = scores["mean"]
            torch.save(model.state_dict(), args.out)
            print(f"[epoch {epoch}] saved new best ({best:.4f}) -> {args.out}")

    print(f"[done] best val Dice = {best:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Centralized BraTS 2D segmentation (ceiling reference)")
    p.add_argument("--data-root", required=True)
    p.add_argument("--max-cases", type=int, default=0, help="0 = all cases")
    p.add_argument("--n-clients", type=int, default=0, help="partition-consistent training over N even hospitals")
    p.add_argument("--fets-csv", default="", help="partition-consistent training over real FeTS hospitals")
    p.add_argument("--synthetic-shift", action="store_true",
                   help="apply deterministic per-hospital scanner shift (synthetic non-IID)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4, help="cosine-decayed over epochs")
    p.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm; 0 disables")
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--index-cache", default="data/slice_index.csv")
    p.add_argument("--out", default="data/centralized_unet.pt")
    train(p.parse_args())


if __name__ == "__main__":
    main()
