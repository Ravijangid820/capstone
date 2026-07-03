"""Phase 1 — centralized 2D BraTS segmentation baseline (single GPU).

Proves the whole pipeline (data -> model -> Dice metrics) before we federate.

Run:
    uv run python -m braintumor_fl.train_centralized \
        --data-root data/BraTS2021_TrainingData \
        --max-cases 100 --epochs 20 --batch-size 8

Start with --max-cases 50-100 on the 4GB card to iterate fast; scale up on Colab.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import REGIONS
from .data import (
    BratsSliceDataset,
    build_slice_index,
    eval_transforms,
    find_cases,
    split_by_case,
    train_transforms,
)
from .model import build_loss, build_metric, build_unet, logits_to_preds


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model, loader, metric, device) -> dict[str, float]:
    model.eval()
    metric.reset()
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
        metric(y_pred=logits_to_preds(logits), y=labels)
    per_region = metric.aggregate()  # tensor of shape (num_regions,)
    scores = {r: float(per_region[i]) for i, r in enumerate(REGIONS)}
    scores["mean"] = float(per_region.mean())
    return scores


def train(args) -> None:
    device = get_device()
    print(f"[setup] device={device} | data_root={args.data_root}")

    cases = find_cases(args.data_root)
    if not cases:
        raise SystemExit(
            f"No BraTS cases found under {args.data_root!r} "
            "(expected subject folders with *_seg.nii.gz). See DATA.md."
        )
    if args.max_cases:
        cases = cases[: args.max_cases]
    print(f"[data] {len(cases)} cases -> building tumor-slice index...")

    index = build_slice_index(cases, cache_csv=args.index_cache)
    split = split_by_case(index, val_frac=args.val_frac)
    print(f"[data] slices: {len(split.train)} train / {len(split.val)} val")

    train_ds = BratsSliceDataset(split.train, train_transforms(args.size))
    val_ds = BratsSliceDataset(split.val, eval_transforms(args.size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type == "cuda"))

    model = build_unet().to(device)
    loss_fn = build_loss()
    metric = build_metric()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        scores = evaluate(model, val_loader, metric, device)
        avg_loss = running / max(1, len(train_loader))
        print(f"[epoch {epoch}] loss={avg_loss:.4f} | "
              f"Dice mean={scores['mean']:.4f} "
              f"(TC={scores['TC']:.3f} WT={scores['WT']:.3f} ET={scores['ET']:.3f})")

        if scores["mean"] > best:
            best = scores["mean"]
            torch.save(model.state_dict(), args.out)
            print(f"[epoch {epoch}] saved new best ({best:.4f}) -> {args.out}")

    print(f"[done] best val Dice = {best:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Centralized BraTS 2D segmentation baseline")
    p.add_argument("--data-root", required=True, help="folder containing BraTS subject dirs")
    p.add_argument("--max-cases", type=int, default=0, help="0 = use all cases")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=192, help="square crop/resize side")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--index-cache", default="data/slice_index.csv")
    p.add_argument("--out", default="best_unet.pt")
    train(p.parse_args())


if __name__ == "__main__":
    main()
