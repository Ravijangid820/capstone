"""Build the self-contained project showcase from the frozen run logs.

    python scripts/showcase_assets.py --data-root data/BraTS2021_Training_Data   # images, once
    python scripts/build_showcase.py                                             # the page

Produces `artifacts/showcase/showcase.html`: one file, no external requests, every number read
out of `artifacts/snapshots/` at build time and every picture rendered by the real pipeline.

The point of generating it rather than writing it by hand is the same as `check_docs.py`'s: a
slide with a hand-typed Dice figure goes stale the moment a run is re-frozen, and nobody
re-derives a number on a slide before presenting it. Here the deck cannot disagree with the
logs, because it has no independent copy of them.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SNAP = REPO / "artifacts" / "snapshots"
SHOWCASE = REPO / "artifacts" / "showcase"

METHODS = ["centralized", "local", "fedavg", "fedbn"]
HOSPITALS = ["H1", "H2", "H3", "H4"]
SITES = {"H1": "Site A", "H2": "Site B", "H3": "Site C", "H4": "Site D"}
DIMS = ["2d", "3d"]
VERSIONS = ["v1", "v2", "v3", "v4", "v5"]

# The estimator window, pre-registered before the runs: mean over the final five rounds.
# v1-v3 ran 25 rounds, v4-v5 ran 40.
WINDOW = {"v1": (21, 25), "v2": (21, 25), "v3": (21, 25), "v4": (36, 40), "v5": (36, 40)}


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def is_diagonal(r: dict) -> bool:
    """Each hospital scored by the model that serves it — global, or its own for personalized."""
    return r["model_hospital"] == "global" or r["model_hospital"] == r["test_hospital"]


def curve_and_estimate(run: Path, tag: str) -> dict | None:
    """Per-round per-site WT Dice, plus the estimator value each site is reported at."""
    rows = [r for r in jsonl(run / "metrics.jsonl")
            if r.get("stage", "round") == "round" and r["split"] == "test" and is_diagonal(r)]
    if not rows:
        return None
    curve: dict[int, dict[str, float]] = {}
    for r in rows:
        curve.setdefault(r["round"], {})[r["test_hospital"]] = r["dice_wt"]

    lo, hi = WINDOW[tag]
    window = [rnd for rnd in curve if lo <= rnd <= hi]
    est = {h: round(statistics.fmean([curve[rnd][h] for rnd in window if h in curve[rnd]]), 4)
           for h in HOSPITALS}
    est["mean"] = round(statistics.fmean(est[h] for h in HOSPITALS), 4)
    return {
        "curve": [{"round": rnd,
                   **{h: round(curve[rnd][h], 4) for h in HOSPITALS if h in curve[rnd]},
                   "mean": round(statistics.fmean(curve[rnd].values()), 4)}
                  for rnd in sorted(curve)],
        "est": est,
    }


def per_case(run: Path) -> dict[str, dict[str, float]]:
    """case_id -> WT Dice, taking the run's reported stage (TTA final where one exists)."""
    rows = jsonl(run / "per_case.jsonl")
    if not rows:
        return {}
    stages = {r.get("stage", "round") for r in rows}
    stage = "final" if "final" in stages else ("rescore" if "rescore" in stages else "round")
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        if r.get("stage", "round") != stage or not is_diagonal(r):
            continue
        out.setdefault(r["test_hospital"], {})[r["case_id"]] = r["dice_wt"]
    return out


def paired(a: dict[str, float], b: dict[str, float], resamples: int = 10000,
           seed: int = 0) -> dict:
    """Paired difference b - a over the cases both sides scored: mean, bootstrap CI, sign split."""
    shared = sorted(set(a) & set(b))
    diffs = [b[c] - a[c] for c in shared]
    if not diffs:
        return {}
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.fmean(rng.choices(diffs, k=n)) for _ in range(resamples))
    return {
        "n": n,
        "delta": round(statistics.fmean(diffs), 4),
        "lo": round(means[int(0.025 * resamples)], 4),
        "hi": round(means[int(0.975 * resamples)], 4),
        "better": sum(1 for d in diffs if d > 0),
        "worse": sum(1 for d in diffs if d < 0),
    }


def collect_runs() -> dict:
    out: dict[str, dict] = {}
    for tag in VERSIONS:
        out[tag] = {}
        for dim in DIMS:
            for m in METHODS:
                run = SNAP / tag / f"{m}_{dim}_42"
                if not run.exists():
                    continue
                got = curve_and_estimate(run, tag)
                if got:
                    out[tag][f"{m}_{dim}"] = got
    return out


def collect_stats(runs: dict) -> dict:
    """The two comparisons the study actually makes, computed from per-case Dice."""
    recipe: dict[str, dict] = {}      # same method, v1 vs v5 -- did the recipe help?
    for dim in DIMS:
        for m in METHODS:
            old, new = per_case(SNAP / "v1" / f"{m}_{dim}_42"), per_case(SNAP / "v5" / f"{m}_{dim}_42")
            if not (old and new):
                continue
            flat_o = {f"{h}/{c}": v for h, cs in old.items() for c, v in cs.items()}
            flat_n = {f"{h}/{c}": v for h, cs in new.items() for c, v in cs.items()}
            recipe[f"{m}_{dim}"] = paired(flat_o, flat_n)

    personal: dict[str, dict] = {}    # FedBN vs FedAvg at v5 -- does personalization pay?
    for dim in DIMS:
        av, bn = per_case(SNAP / "v5" / f"fedavg_{dim}_42"), per_case(SNAP / "v5" / f"fedbn_{dim}_42")
        if not (av and bn):
            continue
        for h in HOSPITALS:
            if h in av and h in bn:
                personal[f"{dim}_{h}"] = paired(av[h], bn[h])
    return {"recipe": recipe, "personal": personal}


def coerce(v):
    """v1's configs were written before the values were typed — '25' there, 25 everywhere else."""
    if not isinstance(v, str):
        return v
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"True": True, "False": False, "None": None}.get(v, v)


def collect_configs() -> dict:
    """One config per iteration — what the run actually set, not what a preset table claims."""
    out = {}
    for tag in VERSIONS:
        for m in METHODS:
            p = SNAP / tag / f"{m}_2d_42" / "config.json"
            if p.exists():
                cfg = {k: coerce(v) for k, v in json.loads(p.read_text(encoding="utf-8")).items()}
                out[tag] = {k: cfg[k] for k in (
                    "rounds", "lr", "lr_schedule", "lr_min_factor", "lr_anneal_rounds",
                    "train_per_hospital", "val_per_hospital", "augment", "aug_flip_p",
                    "aug_rot90_p", "aug_intensity_p", "aug_noise_std", "tta",
                    "postproc_min_voxels", "select_by", "report_last_k", "eval_test_every",
                    "base_channels", "batch_size", "slices_per_case", "train_hw", "patch_size",
                ) if k in cfg}
                break
    return out


def encode_images() -> dict[str, str]:
    """Every picture the deck shows, inlined — the page must work with no network at all."""
    imgs: dict[str, str] = {}

    def add(key: str, path: Path) -> None:
        if path.exists():
            imgs[key] = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

    assets = SHOWCASE / "assets"
    for p in sorted(assets.glob("*.png")):
        add(p.stem, p)

    # The qualitative figures, from the frozen prediction exports.
    for key, d in (("pred2d", "v5_2d_42_H4_BraTS2021_01163"),
                   ("pred3d", "v5_3d_42_H4_BraTS2021_00104")):
        base = REPO / "artifacts" / "figures" / d
        for part in ("mri", "ground_truth", "fedavg", "fedbn"):
            add(f"{key}_{part}", base / f"{part}.png")
        cap = base / "caption.json"
        if cap.exists():
            imgs[f"{key}_caption"] = cap.read_text(encoding="utf-8")
    return imgs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(SHOWCASE / "showcase.html"))
    ap.add_argument("--template", default=str(Path(__file__).parent / "showcase_template.html"))
    args = ap.parse_args()

    if not SNAP.exists():
        print(f"no snapshots at {SNAP}", file=sys.stderr)
        return 1

    partition = json.loads((REPO / "artifacts" / "splits" / "partition.json").read_text())
    assets_meta_path = SHOWCASE / "assets" / "assets.json"
    assets_meta = json.loads(assets_meta_path.read_text()) if assets_meta_path.exists() else {}

    runs = collect_runs()
    data = {
        "sites": SITES,
        "partition": {"meta": partition["meta"], "per_hospital": partition["per_hospital"]},
        "assets_meta": assets_meta,
        "configs": collect_configs(),
        "runs": runs,
        "stats": collect_stats(runs),
        "images": encode_images(),
    }

    template = Path(args.template).read_text(encoding="utf-8")
    marker = "/*__DATA__*/"
    if marker not in template:
        print(f"template has no {marker} marker", file=sys.stderr)
        return 1
    blob = json.dumps(data, separators=(",", ":"))
    html = template.replace(marker, f"const DATA = {blob};")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  ({kb:.0f} KB, {len(data['images'])} images, "
          f"{sum(len(v) for k, v in runs.items())} runs)")
    print("serve it with: python scripts/demo_server.py  ->  http://localhost:8000/showcase")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
