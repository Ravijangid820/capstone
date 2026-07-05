"""Aggregate all experiment results into the headline comparison:
per-hospital Dice for each method, fairness metrics, and the personalization gain.

Reads results/<method>/<site>.json (written by every experiment) and produces:
  - results/per_hospital.csv   (rows=hospitals, cols=methods, values=mean Dice)
  - results/summary.csv        (per method: avg / worst-hospital / std)
  - results/comparison.png     (per-hospital Dice by method)
  - results/personalization_gain.png  (FedBN - FedAvg per hospital)

    uv run python analyze.py            # after running the experiments
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless (WSL / servers)
import matplotlib.pyplot as plt
import pandas as pd

from braintumor_fl.results import read_all

# floor -> ceiling ordering for display
METHOD_ORDER = ["local", "fedavg", "fedprox", "finetune", "personal_head", "fedbn", "centralized"]


def site_key(site: str) -> int:
    try:
        return int(site.split("-")[-1])
    except ValueError:
        return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--metric", default="dice_mean",
                   choices=["dice_mean", "dice_WT", "dice_TC", "dice_ET"])
    args = p.parse_args()

    records = read_all(args.results_dir)
    if not records:
        raise SystemExit(f"No results in {args.results_dir}/ — run some experiments first.")

    df = pd.DataFrame(records)
    table = df.pivot_table(index="site", columns="method", values=args.metric)
    table = table.reindex(sorted(table.index, key=site_key))
    cols = [m for m in METHOD_ORDER if m in table.columns] + \
           [m for m in table.columns if m not in METHOD_ORDER]
    table = table[cols]

    # Summary: average across hospitals, worst hospital (fairness), spread (fairness).
    summary = pd.DataFrame({
        "avg_hospital": table.mean(),
        "worst_hospital": table.min(),
        "std_across_hospitals": table.std(),
    }).reindex(cols)

    os.makedirs(args.results_dir, exist_ok=True)
    table.to_csv(os.path.join(args.results_dir, "per_hospital.csv"))
    summary.to_csv(os.path.join(args.results_dir, "summary.csv"))

    print(f"\n=== Per-hospital {args.metric} ===")
    print(table.round(3).to_string())
    print("\n=== Summary (higher avg + worst = better; lower std = fairer) ===")
    print(summary.round(3).to_string())

    # Personalization gain: FedBN - FedAvg per hospital (the core result).
    if {"fedbn", "fedavg"}.issubset(table.columns):
        gain = (table["fedbn"] - table["fedavg"]).sort_values()
        print("\n=== Personalization gain (FedBN - FedAvg), per hospital ===")
        print(gain.round(3).to_string())
        print(f"mean gain = {gain.mean():.3f} | worst-hospital gain = {gain.iloc[-1]:.3f}")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in gain.values]
        ax.bar(range(len(gain)), gain.values, color=colors)
        ax.axhline(0, color="#444", linewidth=0.8)
        ax.set_xticks(range(len(gain)))
        ax.set_xticklabels(gain.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Dice gain (FedBN - FedAvg)")
        ax.set_title("Personalization gain per hospital")
        fig.tight_layout()
        fig.savefig(os.path.join(args.results_dir, "personalization_gain.png"), dpi=150)

    # Grouped per-hospital comparison across methods.
    fig, ax = plt.subplots(figsize=(max(8, len(table) * 1.1), 5))
    table.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel(args.metric)
    ax.set_xlabel("hospital")
    ax.set_title("Per-hospital Dice by method")
    ax.legend(title="method", fontsize=8, ncol=2)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "comparison.png"), dpi=150)
    print(f"\nWrote CSVs + PNGs to {args.results_dir}/")


if __name__ == "__main__":
    main()
