"""Read metrics.jsonl from every run and decide H1/H2/H3 -- across seeds when present.

    python scripts/analyze.py                 # all runs found under artifacts/runs/
    python scripts/analyze.py --dim 2d        # restrict to one backbone
    python scripts/analyze.py --round 20      # score at a specific round (default: each run's last)

Each run_id is "<method>_<dim>_<seed>", so multiple seeds of the same method live in separate run
dirs. When more than one seed is present, the verdicts are reported per seed AND aggregated
(mean +/- std across seeds, plus "supported in N/M seeds") -- which is what turns a single-run
point estimate into a claim you can defend against run-to-run noise.

The hypotheses reduce to inequalities over the final-round diagonal (docs/experiments.md §3):

    H1  collaboration helps on average   mean_dice(fedavg) >= mean_dice(local)
    H2  the global model fails outliers  dice(fedavg, H4)   <  dice(local, H4)
    H3  personalization recovers them    mean(fedbn) >= mean(fedavg)  AND  dice(fedbn,H4) >= dice(fedavg,H4)

"Diagonal" = each hospital scored by the model that serves it: the single global model
(model_hospital == "global") for centralized/FedAvg, or the hospital's own model
(model_hospital == test_hospital) for local/FedBN.
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

try:                                    # the mean±std tables use non-ASCII; Windows consoles default
    sys.stdout.reconfigure(encoding="utf-8")   # to cp1252 and would mojibake the '±'
except (AttributeError, ValueError):
    pass

METHOD_ORDER = ["centralized", "local", "fedavg", "fedbn"]


def seed_of(run_id: str) -> str:
    return run_id.rsplit("_", 1)[-1]


def load_rows(runs_dir: Path, dim: str | None) -> list[dict]:
    rows = []
    for f in sorted(runs_dir.glob("*/metrics.jsonl")):
        for line in f.open():
            line = line.strip()
            if line:
                r = json.loads(line)
                if dim is None or r.get("dim") == dim:
                    rows.append(r)
    return rows


def _diagonal(rows: list[dict], method: str, seed: str, rnd: int | None) -> dict[str, dict[str, float]]:
    """hospital -> {wt,tc,et} for one (method, seed) at `rnd` (or that run's last round)."""
    sub = [r for r in rows if r["method"] == method and seed_of(r["run_id"]) == seed
           and r["split"] == "test"]
    if not sub:
        return {}
    use = rnd if rnd is not None else max(r["round"] for r in sub)
    out: dict[str, dict[str, float]] = {}
    for r in sub:
        if r["round"] != use:
            continue
        mh, th = r["model_hospital"], r["test_hospital"]
        if mh == "global" or mh == th:
            out[th] = {k: r[f"dice_{k}"] for k in REGIONS}
    return out


def mean_h(diag: dict[str, dict[str, float]], region: str = "wt") -> float:
    return statistics.fmean(d[region] for d in diag.values()) if diag else float("nan")


def _ms(vals: list[float]) -> str:
    vals = [v for v in vals if v == v]
    if not vals:
        return "  –  "
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.fmean(vals):.4f}±{statistics.pstdev(vals):.4f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(header)]
    line = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return "\n".join([line(header), sep, *(line(r) for r in rows)])


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

    outlier = cfg.outlier_hospital
    hospitals = cfg.hospital_ids()

    # method -> sorted list of seeds present
    seeds_by_method: dict[str, list[str]] = defaultdict(set)
    for r in rows:
        seeds_by_method[r["method"]].add(seed_of(r["run_id"]))
    seeds_by_method = {m: sorted(s, key=lambda x: (len(x), x)) for m, s in seeds_by_method.items()}
    present = [m for m in METHOD_ORDER if m in seeds_by_method]
    all_seeds = sorted({s for ss in seeds_by_method.values() for s in ss}, key=lambda x: (len(x), x))

    # diag[(method, seed)] = {hospital: {wt,tc,et}}
    diag: dict[tuple[str, str], dict] = {}
    for m in present:
        for s in seeds_by_method[m]:
            diag[(m, s)] = _diagonal(rows, m, s, args.round)

    print(f"runs: {runs_dir}")
    print(f"seeds present: {', '.join(all_seeds)}  "
          f"({'aggregating mean±std' if len(all_seeds) > 1 else 'single seed'})\n")

    # --- mean Dice across hospitals, per method (aggregated over seeds) -------------------
    print("## Mean Dice across hospitals (diagonal)\n")
    body = []
    for m in present:
        cells = [m]
        for reg in REGIONS:
            cells.append(_ms([mean_h(diag[(m, s)], reg) for s in seeds_by_method[m]]))
        body.append(cells)
    print(_table(["Method", *(r.upper() for r in REGIONS)], body))

    # --- per-hospital WT, per method (aggregated over seeds) ------------------------------
    print(f"\n## Per-hospital WT Dice (outlier = {outlier})\n")
    body = []
    for m in present:
        cells = [m]
        for h in hospitals:
            cells.append(_ms([diag[(m, s)].get(h, {}).get("wt", float("nan"))
                              for s in seeds_by_method[m]]))
        body.append(cells)
    print(_table(["Method", *hospitals], body))

    # --- hypotheses, per seed + aggregate ------------------------------------------------
    def wt_mean(m, s):
        return mean_h(diag.get((m, s), {}))

    def wt_h4(m, s):
        return diag.get((m, s), {}).get(outlier, {}).get("wt", float("nan"))

    tests = {
        "H1": ("mean(fedavg) >= mean(local)",
               lambda s: wt_mean("fedavg", s) >= wt_mean("local", s),
               lambda s: (wt_mean("fedavg", s), wt_mean("local", s)),
               ("fedavg", "local")),
        "H2": (f"dice(fedavg,{outlier}) < dice(local,{outlier})",
               lambda s: wt_h4("fedavg", s) < wt_h4("local", s),
               lambda s: (wt_h4("fedavg", s), wt_h4("local", s)),
               ("fedavg", "local")),
        "H3": ("mean(fedbn)>=mean(fedavg) AND outlier recovered",
               lambda s: wt_mean("fedbn", s) >= wt_mean("fedavg", s)
               and wt_h4("fedbn", s) >= wt_h4("fedavg", s),
               lambda s: (min(wt_mean("fedbn", s) - wt_mean("fedavg", s),
                              wt_h4("fedbn", s) - wt_h4("fedavg", s)),),
               ("fedbn", "fedavg")),
    }

    print("\n## Hypotheses\n")
    body = []
    results: dict[str, dict] = {}
    for hyp, (desc, verdict_fn, obs_fn, needed) in tests.items():
        usable = [s for s in all_seeds if all(s in seeds_by_method.get(m, []) for m in needed)]
        if not usable:
            continue
        verdicts = {s: verdict_fn(s) for s in usable}
        n_ok = sum(verdicts.values())
        agg = "SUPPORTED" if n_ok == len(usable) else ("mixed" if n_ok else "NOT SUPPORTED")
        per_seed = "  ".join(f"{s}:{'Y' if verdicts[s] else 'n'}" for s in usable)
        body.append([hyp, desc, f"{n_ok}/{len(usable)}", per_seed, agg])
        results[hyp] = {"supported_seeds": n_ok, "total_seeds": len(usable),
                        "per_seed": {s: bool(verdicts[s]) for s in usable}, "aggregate": agg,
                        "observed": {s: [round(float(x), 4) for x in obs_fn(s)] for s in usable}}
    print(_table(["Hyp.", "Test", "seeds", "per-seed", "verdict"], body))

    missing = [m for m in METHOD_ORDER if m not in present]
    if missing:
        print(f"\nnot run: {', '.join(missing)}")
    partial = {m: seeds_by_method[m] for m in present if len(seeds_by_method[m]) < len(all_seeds)}
    if partial and len(all_seeds) > 1:
        print("partial seed coverage: " + ", ".join(f"{m}={s}" for m, s in partial.items()))

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "seeds": all_seeds,
            "diagonal": {f"{m}_{s}": diag[(m, s)] for (m, s) in diag},
            "hypotheses": results,
        }
        with out.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
