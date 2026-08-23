"""Paired per-case significance test between two methods -- the evidence behind Table V.

    python scripts/compute_significance.py --dim 2d
    python scripts/compute_significance.py --dim 3d --regions wt tc et
    python scripts/compute_significance.py --dim 2d --seed 42 7 123 --md-out docs/table5-2d.md

`compare_runs.py` answers "did the rerun beat the baseline?" -- the same run_id on two sides of a
config change. This answers a different question on the same per-case rows: **at one config, does
FedBN beat FedAvg?** That is the axis the headline claim lives on, and the reversal (FedBN
recovers the H4 outlier in 2D, loses to FedAvg in 3D) is only a claim about two point estimates
until the 62 test volumes behind each number are compared case by case.

Pairing is what buys the power. The same held-out volume is scored by both methods, so
differencing within a case removes case difficulty -- the dominant source of variance, since a
hard tumour is hard for both models -- and leaves the method effect. A 0.01 mean gap that looks
like rounding noise between two means can be decisively non-zero once paired.

Three numbers are reported per comparison, because no one of them is sufficient:

* **bootstrap 95% CI on the mean difference** -- an effect size with a range, making no normality
  assumption (Dice is bounded in [0,1] and skewed). This is the number a reader should quote.
* **Wilcoxon signed-rank p** -- distribution-free, and the count of tied cases is printed beside
  it so a "significant" result resting on a handful of non-tied volumes is visible rather than
  implied.
* **Holm-adjusted p** -- this script runs one test per hospital per region, so a table of 12
  comparisons will produce a p < 0.05 by chance alone. Holm-Bonferroni controls the family-wise
  error rate over the per-hospital tests within one backbone, which is the family a reviewer will
  count. Uncorrected p is kept in the output rather than replaced, so both are auditable.

Seeds are reported as separate rows, never pooled. Pooling would stack the same test volume three
times and hand the test 3x the sample size it earned, shrinking every interval by sqrt(3) on
replication that is not independent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fedbrats.config import REGIONS, Config          # noqa: E402
from compare_runs import (FINAL_STAGES, is_diagonal,  # noqa: E402
                          load_jsonl, paired_stats, stars, table)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def per_case_rows(run_dir: Path, stage: str | None = None) -> tuple[list[dict], str]:
    """Test-split, diagonal per-case rows for one run, plus which stage supplied them.

    The stage is returned rather than assumed because the same directory can hold rows from a
    plain rescore and from a TTA rescore. Scoring FedAvg from one and FedBN from the other would
    charge a TTA difference to the method comparison -- the exact confound this script exists to
    rule out -- so the caller checks that both sides used the same stage and refuses if not.
    """
    rows = [r for r in load_jsonl(run_dir / "per_case.jsonl")
            if r.get("split") == "test" and is_diagonal(r)]
    if not rows:
        return [], "none"
    stages = {r.get("stage", "round") for r in rows}
    pick = stage or next((s for s in FINAL_STAGES if s in stages), None)
    if pick is None:
        last = max(r["round"] for r in rows)
        return [r for r in rows if r["round"] == last], f"round {last}"
    return [r for r in rows if r.get("stage", "round") == pick], pick


def score_map(rows: list[dict], region: str) -> dict[tuple[str, str], float]:
    """(test_hospital, case_id) -> Dice for one region."""
    return {(r["test_hospital"], r["case_id"]): r[f"dice_{region}"] for r in rows}


# --------------------------------------------------------------------------------------
# multiple comparisons
# --------------------------------------------------------------------------------------

def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the order given.

    Step-down rather than plain Bonferroni: it controls the same family-wise error rate but is
    uniformly more powerful, so it does not throw away a real effect to buy a correction. NaNs
    (a comparison with no non-tied cases) sit out the correction and stay NaN instead of
    inflating the family size for the tests that did run.
    """
    idx = [i for i, p in enumerate(pvalues) if p == p]
    out = [float("nan")] * len(pvalues)
    m = len(idx)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        out[i] = running
    return out


# --------------------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------------------

def compare(a_rows: list[dict], b_rows: list[dict], region: str, hospital: str | None,
            n_boot: int, seed: int) -> dict | None:
    """Paired stats for one (hospital, region) cell. `hospital=None` pools every hospital.

    Only cases present on both sides are used, and the number dropped is reported: a silent
    intersection would let a method that crashed on the hardest 5 volumes look better than the one
    that scored them.
    """
    a, b = score_map(a_rows, region), score_map(b_rows, region)
    keys = sorted(set(a) & set(b))
    if hospital is not None:
        keys = [k for k in keys if k[0] == hospital]
    if not keys:
        return None
    a_vals, b_vals = [a[k] for k in keys], [b[k] for k in keys]
    st = paired_stats(a_vals, b_vals, n_boot=n_boot, seed=seed)
    st["mean_a"] = sum(a_vals) / len(a_vals)
    st["mean_b"] = sum(b_vals) / len(b_vals)
    st["n_dropped"] = len((set(a) ^ set(b)) if hospital is None
                          else {k for k in set(a) ^ set(b) if k[0] == hospital})
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cfg = Config()
    ap.add_argument("--runs", type=str, default=str(cfg.paths.artifacts / "snapshots" / "v1"),
                    help="directory holding the run directories with per_case.jsonl")
    ap.add_argument("--dim", default="2d", choices=("2d", "3d"))
    ap.add_argument("--seed", type=int, nargs="+", default=None,
                    help="seeds to report as separate rows (default: every seed found)")
    ap.add_argument("--a", default="fedavg", help="reference method (the subtrahend)")
    ap.add_argument("--b", default="fedbn", help="method under test; delta is B - A")
    ap.add_argument("--regions", nargs="+", default=["wt"], choices=list(REGIONS))
    ap.add_argument("--stage", default=None, help="per-case stage to score (default: auto)")
    ap.add_argument("--hospitals", nargs="+", default=None, help="default: every hospital found")
    ap.add_argument("--pooled", action="store_true",
                    help="also report a row pooling all hospitals (excluded from the Holm family)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0, help="seed for the bootstrap resampler")
    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--md-out", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.runs)
    if not root.exists():
        print(f"not found: {root}", file=sys.stderr)
        return 1

    # Which seeds have BOTH methods on disk. A seed with only one side cannot be paired, and
    # silently dropping it would let the table imply a coverage it does not have.
    def seeds_for(method: str) -> set[int]:
        out = set()
        for p in root.glob(f"{method}_{args.dim}_*/per_case.jsonl"):
            try:
                out.add(int(p.parent.name.rsplit("_", 1)[1]))
            except ValueError:
                continue
        return out

    have_a, have_b = seeds_for(args.a), seeds_for(args.b)
    seeds = sorted(have_a & have_b) if args.seed is None else sorted(args.seed)
    missing = [s for s in seeds if s not in have_a or s not in have_b]
    if missing:
        for s in missing:
            side = args.a if s not in have_a else args.b
            print(f"seed {s}: no per_case.jsonl for {side}_{args.dim}_{s} under {root}",
                  file=sys.stderr)
        return 1
    if not seeds:
        print(f"no seed under {root} has per_case.jsonl for both "
              f"{args.a}_{args.dim}_* and {args.b}_{args.dim}_*", file=sys.stderr)
        print("  (3D per-case rows come from: "
              f"python scripts/rescore.py --all --dim {args.dim} --seed 42 "
              "--out-dir artifacts/snapshots/v1)", file=sys.stderr)
        return 1

    out_lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        out_lines.append(s)

    emit(f"# {args.b.upper()} vs {args.a.upper()} — {args.dim.upper()}, paired per case")
    emit()
    emit(f"- **runs:** `{root}`")
    emit(f"- **delta:** {args.b} − {args.a} (positive favours {args.b})")
    emit(f"- **test:** Wilcoxon signed-rank on paired per-case Dice; "
         f"95% CI from {args.n_boot} paired bootstrap resamples")
    emit(f"- **seeds:** {', '.join(str(s) for s in seeds)} (reported separately, never pooled)")
    emit()

    rows, cells, family = [], [], []
    for seed in seeds:
        a_dir, b_dir = (root / f"{args.a}_{args.dim}_{seed}",
                        root / f"{args.b}_{args.dim}_{seed}")
        a_rows, a_stage = per_case_rows(a_dir, args.stage)
        b_rows, b_stage = per_case_rows(b_dir, args.stage)
        if a_stage != b_stage:
            print(f"seed {seed}: {args.a} scored at stage '{a_stage}' but {args.b} at "
                  f"'{b_stage}'. Pass --stage to score both the same way.", file=sys.stderr)
            return 1

        hospitals = args.hospitals or sorted({r["test_hospital"] for r in a_rows})
        targets = [(h, h) for h in hospitals] + ([(None, "all")] if args.pooled else [])
        for region in args.regions:
            for hospital, label in targets:
                st = compare(a_rows, b_rows, region, hospital, args.n_boot, args.boot_seed)
                if st is None:
                    continue
                cell = {"seed": seed, "hospital": label, "region": region,
                        "stage": a_stage, **st}
                cells.append(cell)
                if hospital is not None:          # pooled rows sit outside the Holm family
                    family.append(cell)

    for cell, p_adj in zip(family, holm([c["p_value"] for c in family])):
        cell["p_holm"] = p_adj
    for cell in cells:
        cell.setdefault("p_holm", float("nan"))

    fmt_p = lambda p: "  –  " if p != p else (f"{p:.2e}" if p < 1e-3 else f"{p:.4f}")
    for c in cells:
        lo, hi = c["ci95"]
        rows.append([
            str(c["seed"]), c["hospital"], c["region"].upper(), str(c["n"]),
            f"{c['mean_a']:.4f}", f"{c['mean_b']:.4f}", f"{c['mean_delta']:+.4f}",
            f"[{lo:+.4f}, {hi:+.4f}]",
            f"{c['n_improved']}/{c['n_worsened']}/{c['n_tied']}",
            fmt_p(c["p_value"]), fmt_p(c["p_holm"]), stars(c["p_holm"]),
        ])

    emit("## Table V — paired comparison, per hospital")
    emit()
    emit(table(["Seed", "Hosp", "Reg", "n", f"{args.a} mean", f"{args.b} mean",
                "Δ mean", "95% CI (bootstrap)", "up/down/tie", "p", "p (Holm)", ""], rows))
    emit()
    emit(f"Holm family = {len(family)} per-hospital tests"
         + (" (pooled rows excluded)" if args.pooled else "") + ". "
         "`***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` not significant, all after correction.")
    emit()

    dropped = sum(c["n_dropped"] for c in cells)
    if dropped:
        emit(f"> **{dropped} case(s)** were present for one method but not the other and were "
             f"excluded from the pairing.")
        emit()

    # The outlier hospital is the whole point of H3, so it gets stated in prose rather than left
    # for the reader to locate in the table.
    outlier = cfg.outlier_hospital
    head = [c for c in cells if c["hospital"] == outlier and c["region"] == "wt"]
    if head:
        emit(f"## Headline — {outlier} (the shifted hospital), WT")
        emit()
        for c in head:
            lo, hi = c["ci95"]
            direction = (f"{args.b} above {args.a}" if c["mean_delta"] > 0
                         else f"{args.b} below {args.a}")
            verdict = ("consistent with no difference" if not (lo > 0 or hi < 0)
                       else "excludes zero")
            emit(f"- **seed {c['seed']}:** {direction} by {abs(c['mean_delta']):.4f} Dice "
                 f"(95% CI [{lo:+.4f}, {hi:+.4f}], {verdict}; "
                 f"Holm p = {fmt_p(c['p_holm'])}, n = {c['n']}).")
        emit()

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump({"dim": args.dim, "a": args.a, "b": args.b, "runs": str(root),
                       "seeds": seeds, "n_boot": args.n_boot, "cells": cells},
                      f, indent=2, sort_keys=True)
        print(f"\nwrote {p}", file=sys.stderr)
    if args.md_out:
        p = Path(args.md_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"wrote {p}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
