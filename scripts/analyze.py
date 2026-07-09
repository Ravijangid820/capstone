"""Read metrics.jsonl from every run and decide H1/H2/H3.

    python scripts/analyze.py                 # all runs found under artifacts/runs/
    python scripts/analyze.py --dim 2d        # restrict to one backbone
    python scripts/analyze.py --round 20      # score at a specific round (default: each run's last)

Nothing here is a judgement call -- the hypotheses reduce to inequalities over the final-round
diagonal (docs/experiments.md §3):

    H1  collaboration helps on average   mean_dice(fedavg) >= mean_dice(local)
    H2  the global model fails outliers  dice(fedavg, H4)   <  dice(local, H4)
    H3  personalization recovers them    mean(fedbn) >= mean(fedavg)  AND
                                         dice(fedbn, H4) >= dice(fedavg, H4)

"Diagonal" means each hospital scored by the model that serves it. For centralized and FedAvg
there is only one model, so it is `model_hospital == "global"`; for local-only and FedBN each
hospital has its own, so it is `model_hospital == test_hospital`. Collapsing those two cases is
what `_diagonal` does -- and getting it wrong would silently score FedAvg's global model as if it
were four personalized ones.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedbrats.config import REGIONS, Config  # noqa: E402

METHOD_ORDER = ["centralized", "local", "fedavg", "fedbn"]


def load_rows(runs_dir: Path, dim: str | None) -> list[dict]:
    rows = []
    for f in sorted(runs_dir.glob("*/metrics.jsonl")):
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if dim is None or r.get("dim") == dim:
                    rows.append(r)
    return rows


def final_round(rows: list[dict], method: str) -> int | None:
    rs = [r["round"] for r in rows if r["method"] == method]
    return max(rs) if rs else None


def _diagonal(rows: list[dict], method: str, rnd: int) -> dict[str, dict[str, float]]:
    """hospital -> {wt,tc,et} for the model that actually serves that hospital."""
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["method"] != method or r["round"] != rnd or r["split"] != "test":
            continue
        mh, th = r["model_hospital"], r["test_hospital"]
        if mh == "global" or mh == th:            # one global model, or this hospital's own
            out[th] = {k: r[f"dice_{k}"] for k in REGIONS}
    return out


def mean_over_hospitals(diag: dict[str, dict[str, float]]) -> dict[str, float]:
    if not diag:
        return {r: float("nan") for r in REGIONS}
    return {r: statistics.fmean(d[r] for d in diag.values()) for r in REGIONS}


def _fmt(v: float) -> str:
    return "  –  " if v != v else f"{v:.4f}"       # NaN != NaN


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(header)]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(header), sep, *(line(r) for r in rows)])


def verdict(ok: bool) -> str:
    return "SUPPORTED" if ok else "NOT SUPPORTED"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dim", default=None, choices=("2d", "3d"))
    ap.add_argument("--round", type=int, default=None, help="score at this round (default: last)")
    ap.add_argument("--runs-dir", type=str, default=None)
    ap.add_argument("--json-out", type=str, default=None, help="also write the verdicts as JSON")
    args = ap.parse_args()

    cfg = Config()
    runs_dir = Path(args.runs_dir) if args.runs_dir else cfg.paths.runs
    rows = load_rows(runs_dir, args.dim)
    if not rows:
        print(f"no metrics found under {runs_dir}", file=sys.stderr)
        return 1

    present = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    outlier = cfg.outlier_hospital
    hospitals = cfg.hospital_ids()

    diag: dict[str, dict] = {}
    used_round: dict[str, int] = {}
    for m in present:
        rnd = args.round or final_round(rows, m)
        used_round[m] = rnd
        diag[m] = _diagonal(rows, m, rnd)

    print(f"runs: {runs_dir}")
    print("scored at round: " + ", ".join(f"{m}={used_round[m]}" for m in present) + "\n")

    # --- mean Dice across hospitals ------------------------------------------------------
    print("## Mean Dice across hospitals (diagonal)\n")
    print(_table(["Method", *(r.upper() for r in REGIONS)],
                 [[m, *(_fmt(mean_over_hospitals(diag[m])[r]) for r in REGIONS)] for m in present]))

    # --- per-hospital WT ------------------------------------------------------------------
    print(f"\n## Per-hospital WT Dice (outlier = {outlier})\n")
    print(_table(["Method", *hospitals],
                 [[m, *(_fmt(diag[m].get(h, {}).get("wt", float("nan"))) for h in hospitals)]
                  for m in present]))

    # --- local-only cross matrix ----------------------------------------------------------
    cross = defaultdict(dict)
    rnd_l = used_round.get("local")
    for r in rows:
        if r["method"] == "local" and r["round"] == rnd_l and r["split"] == "test":
            cross[r["model_hospital"]][r["test_hospital"]] = r["dice_wt"]
    if len(cross) > 1:
        print("\n## Local-only cross-hospital WT Dice (rows = model, cols = test set)\n")
        print(_table(["model \\ test", *hospitals],
                     [[mh, *(_fmt(cross[mh].get(th, float("nan"))) for th in hospitals)]
                      for mh in hospitals if mh in cross]))

    # --- hypotheses -----------------------------------------------------------------------
    results: dict[str, dict] = {}
    have = set(present)
    print("\n## Hypotheses\n")
    lines = []

    if {"fedavg", "local"} <= have:
        a = mean_over_hospitals(diag["fedavg"])["wt"]
        b = mean_over_hospitals(diag["local"])["wt"]
        ok = a >= b
        results["H1"] = {"supported": ok, "fedavg_mean_wt": a, "local_mean_wt": b}
        lines.append(["H1", "mean(fedavg) >= mean(local)",
                      f"{a:.4f} vs {b:.4f}", verdict(ok)])

        a4 = diag["fedavg"].get(outlier, {}).get("wt", float("nan"))
        b4 = diag["local"].get(outlier, {}).get("wt", float("nan"))
        ok2 = a4 < b4
        results["H2"] = {"supported": ok2, "fedavg_h4_wt": a4, "local_h4_wt": b4}
        lines.append(["H2", f"dice(fedavg,{outlier}) < dice(local,{outlier})",
                      f"{a4:.4f} vs {b4:.4f}", verdict(ok2)])

    if {"fedbn", "fedavg"} <= have:
        a = mean_over_hospitals(diag["fedbn"])["wt"]
        b = mean_over_hospitals(diag["fedavg"])["wt"]
        a4 = diag["fedbn"].get(outlier, {}).get("wt", float("nan"))
        b4 = diag["fedavg"].get(outlier, {}).get("wt", float("nan"))
        ok = (a >= b) and (a4 >= b4)
        results["H3"] = {"supported": ok, "fedbn_mean_wt": a, "fedavg_mean_wt": b,
                         "fedbn_h4_wt": a4, "fedavg_h4_wt": b4}
        lines.append(["H3", "mean(fedbn)>=mean(fedavg) AND outlier recovered",
                      f"{a:.4f}/{a4:.4f} vs {b:.4f}/{b4:.4f}", verdict(ok)])

    if lines:
        print(_table(["Hyp.", "Test", "Observed (WT)", "Verdict"], lines))
    else:
        print("(not enough methods run yet)")

    missing = [m for m in METHOD_ORDER if m not in have]
    if missing:
        print(f"\nnot yet run: {', '.join(missing)}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump({"round": used_round, "diagonal": diag, "hypotheses": results},
                      f, indent=2, sort_keys=True)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
