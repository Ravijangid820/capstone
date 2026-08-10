"""Compare two sets of runs -- the frozen baseline against a rerun -- with paired statistics.

    python scripts/compare_runs.py --dim 2d
    python scripts/compare_runs.py --baseline artifacts/baseline --new artifacts/runs/v2 --dim 2d
    python scripts/compare_runs.py --dim 2d --md-out docs/results-comparison.md

Answers three questions, in the order a reader will ask them:

1. **What actually changed?** A field-by-field diff of the two sides' config.json. A results
   table means nothing until the reader can see that exactly one thing varied, and stated
   intentions are not evidence -- the runs' own recorded configs are.
2. **By how much?** Mean Dice per method and per hospital, before -> after, with the delta.
3. **Is the difference real?** Where both sides logged per-case Dice, the same test volumes are
   compared case by case: a paired bootstrap confidence interval on the mean difference and a
   Wilcoxon signed-rank test. 62 paired volumes per hospital make a small delta defensible or
   expose it as noise; two means alone can do neither.

Both sides are scored with the **same estimator**, passed once and applied everywhere. That is
the point of doing this in one tool instead of running analyze.py twice and pasting the numbers:
the comparison cannot be made with a different rule on each side.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedbrats.config import REGIONS, Config  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

METHOD_ORDER = ["centralized", "local", "fedavg", "fedbn"]

# Config fields that differ for uninteresting reasons (paths on another machine, a tag) and
# would bury the handful of fields that actually define the experiment.
IGNORE_FIELDS = {"paths", "tag", "device", "num_workers"}


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def runs_under(root: Path) -> dict[str, Path]:
    """run_id -> directory, for every run directory holding metrics or per-case rows."""
    out = {}
    for name in ("metrics.jsonl", "per_case.jsonl"):
        for p in sorted(root.glob(f"*/{name}")):
            out.setdefault(p.parent.name, p.parent)
    return out


def _coerce(v):
    """Old config.json files stringified every value; make 25 and '25' compare equal."""
    if not isinstance(v, str):
        return v
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"True": True, "False": False, "None": None}.get(v, v)


def read_config(run_dir: Path, order: tuple[str, ...]) -> dict:
    for name in order:
        p = run_dir / name
        if p.exists():
            return {k: _coerce(v) for k, v in json.loads(p.read_text()).items()}
    return {}


def config_diff(a_dir: Path, b_dir: Path) -> list[tuple[str, object, object]]:
    """Field-by-field difference between what the two sides actually recorded.

    Prefers the rescore config only when *both* sides have one -- that is the eval-only
    comparison (same weights, different inference), where the training config is identical by
    construction and describes neither side's question. Otherwise the training config wins, so
    a baseline that happens to have been backfilled is still compared as a training run.
    """
    both_rescored = all((d / "rescore_config.json").exists() for d in (a_dir, b_dir))
    order = ("rescore_config.json", "config.json") if both_rescored else \
            ("config.json", "rescore_config.json")
    a, b = read_config(a_dir, order), read_config(b_dir, order)
    keys = (set(a) | set(b)) - IGNORE_FIELDS
    return [(k, a.get(k, "—"), b.get(k, "—")) for k in sorted(keys) if a.get(k) != b.get(k)]


# --------------------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------------------

def is_diagonal(r: dict) -> bool:
    return r["model_hospital"] == "global" or r["model_hospital"] == r["test_hospital"]


FINAL_STAGES = ("final", "rescore")     # a selected model re-scored; produced by a run or rescore.py


def diagonal(rows: list[dict], select: str, last_k: int) -> tuple[dict[str, dict[str, float]], str]:
    """hospital -> {wt,tc,et} under the chosen estimator, plus a label for what was actually used.

    The label exists because the fallback is dangerous. Runs made before `stage` existed have no
    final rows, so `--select final` would quietly score them by their last round while scoring
    the other side by its selected model -- a different rule on each side of a comparison whose
    whole purpose is that the rule is identical. The caller compares labels and says so.
    """
    sub = [r for r in rows if r.get("split") == "test" and is_diagonal(r)]
    if not sub:
        return {}, "none"
    rounds_only = [r for r in sub if r.get("stage", "round") == "round"]
    finals = [r for r in sub if r.get("stage") in FINAL_STAGES]

    if select == "final" and finals:
        pool, label = finals, f"final ({sorted({r.get('stage') for r in finals})[0]})"
    else:
        pool = rounds_only or sub
        last = max(r["round"] for r in pool)
        k = last_k if select == "last-k" else 1
        keep = set(range(max(1, last - k + 1), last + 1))
        pool = [r for r in pool if r["round"] in keep]
        label = (f"last {k} rounds (up to {last})" if k > 1 else f"round {last}")
        if select == "final":
            label += " — NO FINAL ROWS, fell back"

    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in pool:
        for reg in REGIONS:
            acc[r["test_hospital"]][reg].append(r[f"dice_{reg}"])
    return {h: {reg: statistics.fmean(v) for reg, v in d.items()} for h, d in acc.items()}, label


def completion(run_dir: Path) -> tuple[int, int | None]:
    """(highest round with results, rounds the config asked for). `None` when unknowable."""
    rows = [r for r in load_jsonl(run_dir / "metrics.jsonl") if r.get("split") == "test"]
    seen = max((r["round"] for r in rows), default=0)
    cfg = read_config(run_dir, ("config.json", "rescore_config.json"))
    want = cfg.get("rounds")
    return seen, want if isinstance(want, int) else None


def is_incomplete(run_dir: Path) -> str | None:
    """Why this run is not finished, or None if it is. Guards against comparing a live run.

    A training run streams its rounds as it goes, so a directory being written *right now* looks
    exactly like a finished one with fewer rounds -- and a last-k estimator over rounds 1-3 of a
    25-round run reports a catastrophic regression that is really just an unfinished experiment.
    `summary.json` is written last and is the reliable signal; the round count covers runs made
    before it existed.
    """
    if (run_dir / "summary.json").exists():
        return None
    # A rescore directory holds a finished evaluation of a finished checkpoint. It has no
    # summary.json and no per-round metrics.jsonl to count, so a round-count test reads it as
    # "0 of 25 rounds" and throws away a perfectly complete side of the comparison.
    if any(r.get("stage") in FINAL_STAGES for r in load_jsonl(run_dir / "rescore_metrics.jsonl")):
        return None
    seen, want = completion(run_dir)
    if want and seen < want:
        return f"{seen}/{want} rounds"
    return None


def metrics_rows(run_dir: Path) -> list[dict]:
    """A run's mean-Dice rows, including any added later by rescore.py.

    rescore writes a separate file rather than appending to metrics.jsonl: that file is hashed in
    the baseline manifest, and a tool that mutates the frozen record to make a comparison work
    has destroyed the thing the comparison was evidence for.
    """
    return load_jsonl(run_dir / "metrics.jsonl") + load_jsonl(run_dir / "rescore_metrics.jsonl")


def per_case_map(rows: list[dict],
                 stage: str | None = None) -> tuple[dict[tuple[str, str], float], str]:
    """(hospital, case_id) -> WT Dice from the per-case rows, plus which stage supplied them.

    Prefers, in order: an explicit `stage`, then "final", then "rescore", then the highest round
    present. Mixing stages would pair a TTA score against a non-TTA one and charge the
    difference to whatever the comparison claims to be about, so the stage is returned and
    reported rather than assumed.
    """
    sub = [r for r in rows if r.get("split") == "test" and is_diagonal(r)]
    if not sub:
        return {}, "none"
    stages = {r.get("stage", "round") for r in sub}
    pick = stage or next((s for s in FINAL_STAGES if s in stages), None)
    if pick:
        sub = [r for r in sub if r.get("stage", "round") == pick]
        label = pick
    else:
        last = max(r["round"] for r in sub)
        sub = [r for r in sub if r["round"] == last]
        label = f"round {last}"
    return {(r["test_hospital"], r["case_id"]): r["dice_wt"] for r in sub}, label


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------

def paired_stats(before: list[float], after: list[float], n_boot: int = 10000,
                 seed: int = 0) -> dict:
    """Paired comparison of the same cases under two models.

    Paired, not two-sample: the hard cases are hard for both models, so per-case variance is
    mostly case difficulty rather than model quality. Differencing within a case removes it and
    leaves the effect, which is why a 0.01 mean shift can be significant here.

    Reports a bootstrap CI (no normality assumption -- Dice is bounded and skewed) alongside a
    Wilcoxon signed-rank test. Ties, common when both models score an empty ET at 1.0, are
    dropped by the signed-rank test; the count is reported so a "significant" result on a
    handful of non-tied cases is visible rather than implied.
    """
    d = np.asarray(after, dtype=float) - np.asarray(before, dtype=float)
    n = d.size
    out = {"n": int(n), "mean_delta": float(d.mean()) if n else float("nan"),
           "median_delta": float(np.median(d)) if n else float("nan"),
           "n_improved": int((d > 0).sum()), "n_worsened": int((d < 0).sum()),
           "n_tied": int((d == 0).sum())}
    if n < 2:
        return {**out, "ci95": (float("nan"), float("nan")), "p_value": float("nan")}

    rng = np.random.default_rng(seed)
    boot = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    out["ci95"] = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    try:
        from scipy.stats import wilcoxon
        nz = d[d != 0]
        out["p_value"] = float(wilcoxon(nz).pvalue) if nz.size >= 1 else float("nan")
    except (ImportError, ValueError):
        out["p_value"] = float("nan")
    return out


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------

def table(header: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "(no rows)"
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    line = lambda c: "| " + " | ".join(x.ljust(w) for x, w in zip(c, widths)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|",
                      *(line(r) for r in rows)])


def fmt_delta(x: float) -> str:
    return "  –  " if x != x else f"{x:+.4f}"


def stars(p: float) -> str:
    if p != p:
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cfg = Config()
    ap.add_argument("--baseline", type=str, default=str(cfg.paths.artifacts / "baseline"))
    ap.add_argument("--new", type=str, default=str(cfg.paths.runs / "v2"))
    ap.add_argument("--dim", default="2d", choices=("2d", "3d"))
    ap.add_argument("--seed", type=int, default=None, help="restrict to one seed")
    ap.add_argument("--select", default="last-k", choices=("last", "last-k", "final"),
                    help="estimator applied to BOTH sides (default: last-k)")
    ap.add_argument("--last-k", type=int, default=5)
    ap.add_argument("--json-out", type=str, default=None)
    ap.add_argument("--md-out", type=str, default=None, help="write the report as markdown")
    ap.add_argument("--include-partial", action="store_true",
                    help="compare runs that have not finished (they will look far worse than they are)")
    args = ap.parse_args()

    base_root, new_root = Path(args.baseline), Path(args.new)
    for p in (base_root, new_root):
        if not p.exists():
            print(f"not found: {p}", file=sys.stderr)
            return 1

    base_runs, new_runs = runs_under(base_root), runs_under(new_root)
    shared = sorted(set(base_runs) & set(new_runs),
                    key=lambda r: (METHOD_ORDER.index(r.split("_")[0])
                                   if r.split("_")[0] in METHOD_ORDER else 99, r))
    shared = [r for r in shared if f"_{args.dim}_" in r
              and (args.seed is None or r.endswith(f"_{args.seed}"))]

    partial = {r: why for r in shared
               if (why := is_incomplete(new_runs[r]) or is_incomplete(base_runs[r]))}
    if partial and not args.include_partial:
        shared = [r for r in shared if r not in partial]
    if not shared:
        print(f"no run_ids in common between {base_root} and {new_root} for dim={args.dim}",
              file=sys.stderr)
        print(f"  baseline has: {', '.join(sorted(base_runs)) or '(none)'}", file=sys.stderr)
        print(f"  new has:      {', '.join(sorted(new_runs)) or '(none)'}", file=sys.stderr)
        return 1

    out_lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        out_lines.append(s)

    est = (f"mean of last {args.last_k} rounds" if args.select == "last-k"
           else "selected model, final evaluation" if args.select == "final" else "final round only")
    emit(f"# Baseline vs rerun — {args.dim.upper()}")
    emit()
    emit(f"- **before:** `{base_root}`")
    emit(f"- **after:** `{new_root}`")
    emit(f"- **estimator (both sides):** {est}")
    emit(f"- **runs compared:** {', '.join(shared)}")
    if partial:
        state = "included anyway" if args.include_partial else "excluded"
        emit(f"- **unfinished, {state}:** "
             + ", ".join(f"`{r}` ({why})" for r, why in sorted(partial.items())))
    emit()

    # --- 1. what changed ------------------------------------------------------------------
    emit("## 1. What changed")
    emit()
    # Any recorded config on each side will do -- a run dir may hold the training config, a
    # rescore's eval config, or both. Requiring one specific filename silently reported "nothing
    # changed" for eval-only comparisons, which is the one case where the diff is the whole point.
    def has_config(d: Path) -> bool:
        return (d / "config.json").exists() or (d / "rescore_config.json").exists()

    probe = next((r for r in shared
                  if has_config(base_runs[r]) and has_config(new_runs[r])), None)
    diff = config_diff(base_runs[probe], new_runs[probe]) if probe else []
    if diff:
        emit(table(["Field", "before", "after"],
                   [[k, str(a), str(b)] for k, a, b in diff]))
    elif probe:
        emit("No config differences found — the two sides ran the same recipe.")
    else:
        emit("_Neither side recorded a config, so nothing could be diffed._")
    emit()

    # --- 2. how much ----------------------------------------------------------------------
    emit("## 2. Dice, before → after")
    emit()
    results: dict[str, dict] = {}
    body, labels = [], {}
    for rid in shared:
        b, lb = diagonal(metrics_rows(base_runs[rid]), args.select, args.last_k)
        n, ln = diagonal(metrics_rows(new_runs[rid]), args.select, args.last_k)
        labels[rid] = (lb, ln)
        if not b or not n:
            continue
        row = [rid]
        rec = {}
        for reg in REGIONS:
            mb = statistics.fmean(d[reg] for d in b.values())
            mn = statistics.fmean(d[reg] for d in n.values())
            row += [f"{mb:.4f}", f"{mn:.4f}", fmt_delta(mn - mb)]
            rec[reg] = {"before": mb, "after": mn, "delta": mn - mb}
        body.append(row)
        results[rid] = {"mean": rec, "per_hospital": {}}
        for h in sorted(set(b) & set(n)):
            results[rid]["per_hospital"][h] = {
                "before": b[h]["wt"], "after": n[h]["wt"], "delta": n[h]["wt"] - b[h]["wt"]}
    header = ["Run"] + [f"{r.upper()} {c}" for r in REGIONS for c in ("before", "after", "Δ")]
    emit(table(header, body))
    emit()

    mismatched = {rid: v for rid, v in labels.items() if v[0] != v[1]}
    if mismatched:
        emit("> **⚠ The two sides were not scored the same way.** A delta across different "
             "estimators measures the estimators as much as the models.")
        emit(">")
        for rid, (lb, ln) in mismatched.items():
            emit(f"> - `{rid}`: before = _{lb}_, after = _{ln}_")
        emit(">")
        emit("> Give the baseline comparable rows with `scripts/rescore.py` (no retraining), "
             "or compare with `--select last-k`, which both sides can always satisfy.")
        emit()

    emit("### Per-hospital WT Dice")
    emit()
    hospitals = sorted({h for r in results.values() for h in r["per_hospital"]})
    body = []
    for rid, rec in results.items():
        row = [rid]
        for h in hospitals:
            d = rec["per_hospital"].get(h)
            row.append(f"{d['before']:.4f} → {d['after']:.4f} ({d['delta']:+.4f})" if d else "—")
        body.append(row)
    emit(table(["Run", *hospitals], body))
    emit()

    # --- 3. is it real --------------------------------------------------------------------
    emit("## 3. Is the difference real? (paired, per case)")
    emit()
    body, paired_out, pc_labels = [], {}, {}
    for rid in shared:
        before, lb = per_case_map(load_jsonl(base_runs[rid] / "per_case.jsonl"))
        after, ln = per_case_map(load_jsonl(new_runs[rid] / "per_case.jsonl"))
        common = sorted(set(before) & set(after))
        if not common:
            continue
        pc_labels[rid] = (lb, ln)
        paired_out[rid] = {}
        for h in [*hospitals, "ALL"]:
            keys = common if h == "ALL" else [k for k in common if k[0] == h]
            if not keys:
                continue
            st = paired_stats([before[k] for k in keys], [after[k] for k in keys])
            paired_out[rid][h] = st
            body.append([rid, h, str(st["n"]), fmt_delta(st["mean_delta"]),
                         f"[{st['ci95'][0]:+.4f}, {st['ci95'][1]:+.4f}]",
                         f"{st['n_improved']}/{st['n_worsened']}",
                         f"{st['p_value']:.2g} {stars(st['p_value'])}".strip()])
    if body:
        emit(table(["Run", "Hospital", "n", "Δ mean WT", "95% CI (bootstrap)",
                    "better/worse", "Wilcoxon p"], body))
        emit()
        emit("`better/worse` counts cases that moved in each direction; the remainder tied. "
             "A CI excluding zero and a small p mean the improvement survives case-level noise.")
        emit()
        srcs = sorted({f"`{rid}`: {lb} → {ln}" for rid, (lb, ln) in pc_labels.items()
                       if lb != ln})
        used = sorted({lb for lb, _ in pc_labels.values()} | {ln for _, ln in pc_labels.values()})
        emit(f"Per-case rows read from stage: {', '.join(used)}. "
             "These can differ from section 2 — section 2 averages rounds of ordinary inference, "
             "while a `final` stage is the selected model re-scored with TTA. "
             "The two answer different questions and are expected to disagree in size.")
        if srcs:
            emit()
            emit("> **⚠ Different stages on each side:** " + "; ".join(srcs))
    else:
        emit("_No per-case rows on both sides, so no paired test could be run._")
        emit()
        emit("Backfill the baseline from its checkpoints (no retraining):")
        emit()
        emit("```")
        emit(f"python scripts/rescore.py --all --dim {args.dim} --out-dir {base_root}")
        emit("```")
    emit()

    payload = {"baseline": str(base_root), "new": str(new_root), "dim": args.dim,
               "estimator": {"select": args.select, "last_k": args.last_k},
               "config_diff": [[k, str(a), str(b)] for k, a, b in diff],
               "results": results, "paired": paired_out}
    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(f"wrote {p}")
    if args.md_out:
        p = Path(args.md_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
