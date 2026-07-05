"""Evaluate ONE model on every hospital's validation set — used to score the
centralized CEILING per hospital (the same model, tested on each hospital's data).

Uses each hospital's val cases (identical split to local/FL/finetune), so the
comparison is apples-to-apples.

    uv run python fl/evaluate.py --data-root data/BraTS2021_Training_Data \
        --fets-csv data/fets_partitioning.csv --model data/centralized_unet.pt --method centralized
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from braintumor_fl.data import BratsSliceDataset, build_slice_index, case_split, eval_transforms
from braintumor_fl.model import BratsUNet, build_metric
from braintumor_fl.partition import get_partitions
from braintumor_fl.results import write_scores
from braintumor_fl.trainer import evaluate, get_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--model", required=True, help="checkpoint to evaluate on every hospital")
    p.add_argument("--n-clients", type=int, default=0)
    p.add_argument("--fets-csv", default="")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--method", default="centralized", help="result tag")
    p.add_argument("--norm", default="batch", choices=["batch", "instance"],
                   help="must match how the checkpoint was trained")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    device = get_device()
    state = torch.load(args.model, map_location=device)
    model = BratsUNet(norm=args.norm).to(device)
    model.load_state_dict(state)
    metric = build_metric()

    parts = get_partitions(args.data_root, args.n_clients or None,
                           args.fets_csv or None, args.max_cases)
    os.makedirs(os.path.join(args.results_dir, args.method), exist_ok=True)
    print(f"[eval] {args.model} on {len(parts)} hospitals (norm={args.norm})")

    for i, cases in enumerate(parts):
        site = f"site-{i + 1}"
        _, val_cases = case_split(cases)  # same held-out val as every other method
        val_index = build_slice_index(
            val_cases, cache_csv=os.path.join(args.results_dir, args.method, f"_index_{site}.csv")
        )
        val_ds = BratsSliceDataset(val_index, eval_transforms(args.size))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

        scores = evaluate(model, val_loader, metric, device)
        write_scores(args.results_dir, args.method, site, scores, 0, len(val_index))
        print(f"[eval] {site}: Dice mean={scores['mean']:.4f} "
              f"(TC={scores['TC']:.3f} WT={scores['WT']:.3f} ET={scores['ET']:.3f})")


if __name__ == "__main__":
    main()
