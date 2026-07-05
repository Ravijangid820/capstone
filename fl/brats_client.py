"""FLARE client = one simulated hospital.

Each round: receive the global model, load it (respecting the personalization
strategy's kept-local params), train locally on THIS hospital's private cases,
evaluate, save the (personalized) model + score record, and send updated weights
back. Uses the shared `trainer` code — identical training to the centralized run.

Paths (data-root, results-dir) MUST be absolute: in the FLARE simulator the
client's working directory is the job workspace, not the project root.
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

import nvflare.client as flare
from braintumor_fl.data import (
    BratsSliceDataset,
    build_slice_index,
    eval_transforms,
    split_by_case,
    train_transforms,
)
from braintumor_fl.model import build_metric, build_unet
from braintumor_fl.partition import client_cases
from braintumor_fl.personalization import keep_local_keys, load_global
from braintumor_fl.results import write_scores
from braintumor_fl.trainer import evaluate, get_device, local_train


def build_loaders(cases, batch_size, size, workers, index_cache):
    index = build_slice_index(cases, cache_csv=index_cache)
    split = split_by_case(index)
    train_ds = BratsSliceDataset(split.train, train_transforms(size))
    val_ds = BratsSliceDataset(split.val, eval_transforms(size))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=workers)
    return train_loader, val_loader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="ABSOLUTE path to BraTS cases")
    p.add_argument("--results-dir", required=True, help="ABSOLUTE path for outputs")
    p.add_argument("--method", required=True, help="tag for result files, e.g. fedbn")
    p.add_argument("--n-clients", type=int, default=0)
    p.add_argument("--fets-csv", default="", help="ABSOLUTE path; overrides n-clients")
    p.add_argument("--client-index", type=int, required=True)
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--personalization", choices=["fedavg", "fedbn", "personal_head"], default="fedavg")
    p.add_argument("--prox-mu", type=float, default=0.0, help=">0 enables FedProx")
    p.add_argument("--epochs", type=int, default=1, help="local epochs per round")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fets_csv = args.fets_csv or None

    my_cases = client_cases(
        args.data_root, args.client_index,
        n_clients=args.n_clients or None, fets_csv=fets_csv, max_cases=args.max_cases,
    )

    device = get_device()
    model = build_unet().to(device)
    keep_local = keep_local_keys(model, args.personalization)

    index_cache = os.path.join(args.results_dir, f"_index_c{args.client_index}.csv")
    train_loader, val_loader = build_loaders(
        my_cases, args.batch_size, args.size, args.workers, index_cache
    )
    metric = build_metric()
    n_train, n_val = len(train_loader.dataset), len(val_loader.dataset)

    flare.init()
    site = flare.get_site_name()

    while flare.is_running():
        input_model = flare.receive()
        # Assemble this hospital's model: global params, EXCEPT kept-local ones.
        load_global(model, input_model.params, keep_local)

        # Evaluate BEFORE local training: this scores the true federated model on
        # this hospital (pure global for FedAvg; global-body + local-BN for FedBN),
        # with no local-adaptation contamination. This is the number we report.
        scores = evaluate(model, val_loader, metric, device)
        os.makedirs(os.path.join(args.results_dir, args.method), exist_ok=True)
        torch.save(model.state_dict(),
                   os.path.join(args.results_dir, args.method, f"{site}.pt"))
        write_scores(args.results_dir, args.method, site, scores, n_train, n_val)

        # Now train locally to contribute this round's update.
        global_ref = dict(input_model.params) if args.prox_mu > 0 else None  # FedProx anchor
        local_train(model, train_loader, args.epochs, args.lr, device,
                    prox_mu=args.prox_mu, global_params=global_ref)

        output = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"val_dice_mean": scores["mean"], "val_dice_WT": scores["WT"]},
            meta={"NUM_STEPS_CURRENT_ROUND": len(train_loader)},
        )
        model.to(device)
        flare.send(output)


if __name__ == "__main__":
    main()
