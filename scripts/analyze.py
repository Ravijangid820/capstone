"""Read metrics.jsonl from every run and decide H1/H2/H3 -- across seeds when present.

    python scripts/analyze.py                          # all runs found under artifacts/runs/
    python scripts/analyze.py --dim 2d                 # restrict to one backbone
    python scripts/analyze.py --round 20               # score at a specific round
    python scripts/analyze.py --select last-k --last-k 5    # average the final K rounds
    python scripts/analyze.py --runs-dir artifacts/runs/v2  # a tagged rerun

Each run_id is "<method>_<dim>_<seed>", so multiple seeds of the same method live in separate run
dirs. When more than one seed is present, the verdicts are reported per seed AND aggregated
(mean +/- std across seeds, plus "supported in N/M seeds") -- which is what turns a single-run
point estimate into a claim you can defend against run-to-run noise.

**Choosing an estimator, and why it is not cosmetic.** These curves plateau by roughly round 15
and then oscillate. On the frozen baseline the swing between adjacent rounds reaches 0.12 WT
Dice, which is several times the gap between the methods under test -- so `--select last`, a
single round, reports one draw from that oscillation. It is not a wrong number, it is a noisy
estimator of the right one, and on the baseline it flips H1 from supported in 1/3 seeds to 3/3.

    last     the final round alone. What the baseline reported; kept as the default so old
             numbers stay reproducible.
    last-k   mean over the final K rounds. Cuts evaluation noise without touching the models.
    final    the run's own `stage="final"` rows -- the selected model re-scored with TTA.
             Runs made before selection existed have no such rows and fall back to `last`.

Pick the estimator before looking at the verdicts, and apply the same one to both sides of any
comparison. `compare_runs.py` enforces that; here it is on you.

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


def is_diagonal(r: dict) -> bool:
    """The cell that matters: each hospital scored by the model that actually serves it."""
    return r["model_hospital"] == "global" or r["model_hospital"] == r["test_hospital"]


def _diagonal(rows: list[dict], method: str, seed: str, rnd: int | None,
              select: str = "last", last_k: int = 5) -> dict[str, dict[str, float]]:
    """hospital -> {wt,tc,et} for one (method, seed) under the chosen estimator."""
    sub = [r for r in rows if r["method"] == method and seed_of(r["run_id"]) == seed
           and r["split"] == "test" and is_diagonal(r)]
    if not sub:
        return {}

    # Runs predating the stage field carry none; treat their rows as ordinary round rows.
    rounds_only = [r for r in sub if r.get("stage", "round") == "round"]
    finals = [r for r in sub if r.get("stage") == "final"]

    if rnd is not None:
        keep, pool = [rnd], rounds_only
    elif select == "final" and finals:
        keep, pool = sorted({r["round"] for r in finals}), finals
    else:
        pool = rounds_only or sub
        last = max(r["round"] for r in pool)
        k = last_k if select == "last-k" else 1
        keep = list(range(max(1, last - k + 1), last + 1))

    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in pool:
        if r["round"] in keep:
            for k_ in REGIONS:
                acc[r["test_hospital"]][k_].append(r[f"dice_{k_}"])
    return {h: {k_: statistics.fmean(v) for k_, v in d.items()} for h, d in acc.items()}


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
    ap.add_argument("--round", type=int, default=None, help="score at this round (overrides --select)")
    ap.add_argument("--select", default="last", choices=("last", "last-k", "final"),
                    help="estimator for the headline figure (see module docstring)")
    ap.add_argument("--last-k", type=int, default=5, help="rounds averaged when --select last-k")
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
            diag[(m, s)] = _diagonal(rows, m, s, args.round, args.select, args.last_k)

    est = (f"round {args.round}" if args.round is not None else
           f"mean of last {args.last_k} rounds" if args.select == "last-k" else
           "selected model, final evaluation" if args.select == "final" else "final round only")
    print(f"runs: {runs_dir}")
    print(f"estimator: {est}")
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
            "estimator": {"select": args.select, "last_k": args.last_k, "round": args.round},
            "runs_dir": str(runs_dir),
            "diagonal": {f"{m}_{s}": diag[(m, s)] for (m, s) in diag},
            "hypotheses": results,
        }
        with out.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
