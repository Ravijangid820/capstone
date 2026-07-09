"""Plot the 2D results from metrics.jsonl into artifacts/figures/.

    python scripts/plot_results.py

Three figures:
  learning_curves_wt.png   mean diagonal WT Dice vs round, per method
  per_hospital_wt.png      final-round WT per hospital, per method (outlier = H4)
  outlier_h4_wt.png        the story in one panel: H4 WT vs round, per method
"""
from __future__ import annotations

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


def load(runs_dir: Path, method: str) -> list[dict]:
    f = runs_dir / f"{method}_2d_42" / "metrics.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def is_diag(r: dict) -> bool:
    return r["split"] == "test" and (r["model_hospital"] == "global"
                                     or r["model_hospital"] == r["test_hospital"])


def main() -> int:
    cfg = Config()
    runs = cfg.paths.runs
    out = cfg.paths.artifacts / "figures"
    out.mkdir(parents=True, exist_ok=True)
    data = {m: load(runs, m) for m in METHODS}
    present = [m for m in METHODS if data[m]]

    # --- 1. learning curves: mean diagonal WT vs round -----------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in present:
        by_round: dict[int, list[float]] = {}
        for r in data[m]:
            if is_diag(r):
                by_round.setdefault(r["round"], []).append(r["dice_wt"])
        xs = sorted(by_round)
        ax.plot(xs, [statistics.fmean(by_round[x]) for x in xs], marker="o", ms=3,
                color=COLORS[m], label=m)
    ax.set_xlabel("round")
    ax.set_ylabel("mean diagonal WT Dice")
    ax.set_title("Learning curves — mean WT Dice across hospitals")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "learning_curves_wt.png", dpi=130)
    plt.close(fig)

    # --- 2. final per-hospital WT bars ---------------------------------------------------
    last = {m: max(r["round"] for r in data[m]) for m in present}

    def final_diag(m: str) -> dict[str, float]:
        d = {}
        for r in data[m]:
            if r["round"] == last[m] and is_diag(r):
                d[r["test_hospital"]] = r["dice_wt"]
        return d

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    import numpy as np
    w = 0.8 / len(present)
    for i, m in enumerate(present):
        d = final_diag(m)
        ax.bar(np.arange(len(HOS)) + i * w, [d.get(h, 0) for h in HOS], w,
               color=COLORS[m], label=m)
    ax.set_xticks(np.arange(len(HOS)) + 0.4 - w / 2)
    ax.set_xticklabels([h + (" *" if h == cfg.outlier_hospital else "") for h in HOS])
    ax.set_ylabel("final WT Dice")
    ax.set_ylim(0.6, 0.95)
    ax.set_title("Final per-hospital WT Dice  (* = outlier)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "per_hospital_wt.png", dpi=130)
    plt.close(fig)

    # --- 3. the outlier story: H4 WT vs round --------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for m in present:
        by_round = {}
        for r in data[m]:
            if is_diag(r) and r["test_hospital"] == cfg.outlier_hospital:
                by_round[r["round"]] = r["dice_wt"]
        xs = sorted(by_round)
        ax.plot(xs, [by_round[x] for x in xs], marker="o", ms=3, color=COLORS[m], label=m)
    ax.set_xlabel("round")
    ax.set_ylabel(f"{cfg.outlier_hospital} (outlier) WT Dice")
    ax.set_title(f"The outlier: {cfg.outlier_hospital} WT Dice vs round")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "outlier_h4_wt.png", dpi=130)
    plt.close(fig)

    print(f"wrote 3 figures to {out}")
    for f in sorted(out.glob("*.png")):
        print(" ", f.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
