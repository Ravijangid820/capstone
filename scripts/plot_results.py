"""Plot the results from metrics.jsonl into artifacts/figures/.

    python scripts/plot_results.py --dim 2d
    python scripts/plot_results.py --dim 3d --seed 7
    python scripts/plot_results.py --dim 2d --tag v2 --suffix _v2

Three figures:
  learning_curves_wt_<dim>.png   mean diagonal WT Dice vs round, per method
  per_hospital_wt_<dim>.png      reported WT per hospital, per method (outlier = H4)
  outlier_h4_wt_<dim>.png        the story in one panel: H4 WT vs round, per method

Curves plot `stage="round"` rows only. A run also emits `stage="final"` rows -- the selected
model re-scored, with TTA if it was enabled -- at whichever round was selected. Those share a
round number with an ordinary curve point, so averaging the two together would put a single
TTA-inflated point in the middle of a non-TTA curve and make it look like a spike in training.
The bar chart deliberately prefers those final rows: they are the reported result.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fedbrats.config import Config  # noqa: E402

METHODS = ["centralized", "local", "fedavg", "fedbn"]
COLORS = {"centralized": "#6b7280", "local": "#2563eb", "fedavg": "#f59e0b", "fedbn": "#10b981"}
HOS = ["H1", "H2", "H3", "H4"]


def load(runs_dir: Path, method: str, dim: str, seed: int) -> list[dict]:
    f = runs_dir / f"{method}_{dim}_{seed}" / "metrics.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def is_diag(r: dict) -> bool:
    return r["split"] == "test" and (r["model_hospital"] == "global"
                                     or r["model_hospital"] == r["test_hospital"])


def is_curve(r: dict) -> bool:
    """A per-round learning-curve point. Rows predating `stage` carry none and still qualify."""
    return is_diag(r) and r.get("stage", "round") == "round"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dim", default="2d", choices=("2d", "3d"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None, help="plot a tagged rerun: artifacts/runs/<tag>/")
    ap.add_argument("--runs-dir", type=str, default=None)
    ap.add_argument("--suffix", default="", help="appended to each filename, to avoid overwriting")
    args = ap.parse_args()

    dim = args.dim
    cfg = Config(dim=dim)
    runs = Path(args.runs_dir) if args.runs_dir else (
        cfg.paths.runs / args.tag if args.tag else cfg.paths.runs)
    out = cfg.paths.artifacts / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = {m: load(runs, m, dim, args.seed) for m in METHODS}
    present = [m for m in METHODS if data[m]]

    if not present:
        print(f"No runs found for dim={dim} seed={args.seed} under {runs}")
        return 1
    tail = f"{dim}{args.suffix}"

    # --- 1. learning curves: mean diagonal WT vs round -----------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in present:
        by_round: dict[int, list[float]] = {}
        for r in data[m]:
            if is_curve(r):
                by_round.setdefault(r["round"], []).append(r["dice_wt"])
        xs = sorted(by_round)
        ax.plot(xs, [statistics.fmean(by_round[x]) for x in xs], marker="o", ms=3,
                color=COLORS[m], label=m)
    ax.set_xlabel("round")
    ax.set_ylabel("mean diagonal WT Dice")
    ax.set_title(f"Learning curves ({dim.upper()}) — mean WT Dice across hospitals")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"learning_curves_wt_{tail}.png", dpi=130)
    plt.close(fig)

    # --- 2. reported per-hospital WT bars -------------------------------------------------
    def final_diag(m: str) -> dict[str, float]:
        """The reported number per hospital: the run's own final rows, else its last curve point."""
        finals = [r for r in data[m] if is_diag(r) and r.get("stage") == "final"]
        if finals:
            return {r["test_hospital"]: r["dice_wt"] for r in finals}
        curve = [r for r in data[m] if is_curve(r)]
        last = max(r["round"] for r in curve)
        return {r["test_hospital"]: r["dice_wt"] for r in curve if r["round"] == last}

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    import numpy as np
    w = 0.8 / len(present)
    for i, m in enumerate(present):
        d = final_diag(m)
        ax.bar(np.arange(len(HOS)) + i * w, [d.get(h, 0) for h in HOS], w,
               color=COLORS[m], label=m)
    ax.set_xticks(np.arange(len(HOS)) + 0.4 - w / 2)
    ax.set_xticklabels([h + (" *" if h == cfg.outlier_hospital else "") for h in HOS])
    ax.set_ylabel("reported WT Dice")
    ax.set_ylim(0.6, 0.95)
    ax.set_title(f"Reported per-hospital WT Dice ({dim.upper()})  (* = outlier)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"per_hospital_wt_{tail}.png", dpi=130)
    plt.close(fig)

    # --- 3. the outlier story: H4 WT vs round --------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in present:
        by_round = {}
        for r in data[m]:
            if is_curve(r) and r["test_hospital"] == cfg.outlier_hospital:
                by_round[r["round"]] = r["dice_wt"]
        xs = sorted(by_round)
        ax.plot(xs, [by_round[x] for x in xs], marker="o", ms=3, color=COLORS[m], label=m)
    ax.set_xlabel("round")
    ax.set_ylabel(f"{cfg.outlier_hospital} (outlier) WT Dice")
    ax.set_title(f"The outlier ({dim.upper()}): {cfg.outlier_hospital} WT Dice vs round")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"outlier_h4_wt_{tail}.png", dpi=130)
    plt.close(fig)

    print(f"wrote 3 figures to {out}  (source: {runs}, seed {args.seed})")
    for f in sorted(out.glob(f"*_{tail}.png")):
        print(" ", f.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
