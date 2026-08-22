"""Re-evaluate a finished run's checkpoint and write per-case Dice. No training.

    python scripts/rescore.py --run artifacts/runs/fedavg_2d_42 --out artifacts/baseline/fedavg_2d_42
    python scripts/rescore.py --all --dim 2d --seed 42        # every method for one backbone
    python scripts/rescore.py --all --dim 2d --tta --postproc-min-voxels 50 --stage tta

Two jobs, both of which avoid spending a GPU-hour to learn something already on disk.

**Backfilling the baseline.** `evaluate_cases` used to compute per-case Dice and discard it,
keeping only the mean. Without the per-case numbers a before/after comparison can say one mean
is larger than another but cannot say whether 62 test volumes actually moved -- no confidence
interval, no paired test. The checkpoints survived, so the per-case numbers can be recovered
exactly rather than approximated, and the comparison becomes paired at the case level.

**Isolating the inference-side gains.** Flip-TTA and component filtering change no weights. Run
this with `--tta --postproc-min-voxels 50` against the *baseline* checkpoint and the delta is
attributable to inference alone, separately from anything retraining later changes.

The written rows carry a `stage` label (default "rescore") so they never collide with the rows
the training run wrote.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedbrats.config import Config                            # noqa: E402
from fedbrats.data import load_index, select_cases            # noqa: E402
from fedbrats.federated import METHODS                        # noqa: E402
from fedbrats.logging_utils import MetricsWriter, RunExistsError, get_logger  # noqa: E402
from fedbrats.model import build_model                        # noqa: E402
from fedbrats.train import evaluate_cases                     # noqa: E402


def parse_run_id(run_id: str) -> tuple[str, str, int]:
    """'fedavg_2d_42' -> ('fedavg', '2d', 42)."""
    method, dim, seed = run_id.rsplit("_", 2)
    return method, dim, int(seed)


def load_states(ckpt_path: Path, method: str, hospitals: list[str]) -> dict[str, dict]:
    """hospital -> state_dict to evaluate that hospital with.

    Handles both checkpoint layouts: the original one saved the payload bare, the current one
    wraps it with the selected round and config. Distinguishing them by key ('bn' present means
    an aggregating method) keeps the old runs readable instead of stranding them.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    payload = ckpt["state"] if isinstance(ckpt, dict) and "state" in ckpt else ckpt

    if isinstance(payload, dict) and "bn" in payload and "global" in payload:
        glob, bn = payload["global"], payload["bn"] or {}
        if method == "fedbn":                       # global body + that hospital's own BN
            return {h: {**glob, **bn.get(h, {})} for h in hospitals}
        return {h: dict(glob) for h in hospitals}   # fedavg: one model serves everyone
    if "global" in payload:                         # centralized: one pooled model
        return {h: payload["global"] for h in hospitals}
    return {h: payload[h] for h in hospitals}       # local-only: one model each


OUTPUTS = ("per_case.jsonl", "rescore_metrics.jsonl")


def rescore(run_dir: Path, out_dir: Path, cfg: Config, method: str, stage: str,
            log, overwrite: bool = False) -> dict[str, dict[str, float]]:
    ckpt_path = run_dir / "checkpoints" / "final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")

    # These writers append, so a second rescore into the same directory would stack a duplicate
    # set of rows behind the first and double every case -- invisible in the file, fatal to a
    # paired test that then matches each case against itself.
    existing = [n for n in OUTPUTS if (out_dir / n).exists() and (out_dir / n).stat().st_size]
    if existing and not overwrite:
        raise RunExistsError(
            f"{out_dir} already contains {', '.join(existing)}.\n"
            f"Re-running would append duplicate rows. Pass --overwrite to replace them, "
            f"or --out-dir <other> to write elsewhere.")
    for n in OUTPUTS if overwrite else ():
        (out_dir / n).unlink(missing_ok=True)

    hospitals = cfg.hospital_ids()
    index = load_index(cfg)
    states = load_states(ckpt_path, method, hospitals)

    model = build_model(cfg)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case_log = MetricsWriter(out_dir / "per_case.jsonl")
    # A separate file, never metrics.jsonl: that one is hashed in the baseline manifest, and a
    # tool that edits the frozen record to make a later comparison work has destroyed the record.
    mean_log = MetricsWriter(out_dir / "rescore_metrics.jsonl")
    # Also never config.json -- writing that into a frozen baseline directory would silently
    # invalidate its manifest hash and rewrite the record of what the baseline actually ran.
    with (out_dir / "rescore_config.json").open("w") as f:
        json.dump(cfg.to_dict(), f, indent=2, sort_keys=True, default=str)

    means: dict[str, dict[str, float]] = {}
    for h in hospitals:
        cases = select_cases(index, h, "test", cfg.max_test_cases)
        model.load_state_dict(states[h])
        dice, per_case = evaluate_cases(model, cfg, cases, device, tta=cfg.tta)
        mh = "global" if method in ("fedavg", "centralized") else h
        common = {"run_id": run_dir.name, "method": method, "dim": cfg.dim, "round": -1,
                  "stage": stage, "model_hospital": mh, "test_hospital": h, "split": "test",
                  "tta": cfg.tta}
        mean_log.write(**common, dice_wt=dice["wt"], dice_tc=dice["tc"], dice_et=dice["et"])
        per_case_log.write_many([
            {**common, "case_id": c["case_id"],
             "dice_wt": c["wt"], "dice_tc": c["tc"], "dice_et": c["et"]}
            for c in per_case])
        means[h] = dice
        log.info(f"  {h}: {len(cases)} cases  "
                 f"WT={dice['wt']:.4f} TC={dice['tc']:.4f} ET={dice['et']:.4f}")
    return means


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=str, default=None, help="a single run directory")
    ap.add_argument("--out", type=str, default=None, help="where per_case.jsonl goes (default: --run)")
    ap.add_argument("--all", action="store_true", help="every method for --dim/--seed")
    ap.add_argument("--runs-dir", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None, help="parent for --all outputs")
    ap.add_argument("--dim", default="2d", choices=("2d", "3d"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", default="rescore", help="label written into each row")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--postproc-min-voxels", type=int, default=0)
    ap.add_argument("--sw-overlap", type=float, default=None)
    ap.add_argument("--max-test-cases", type=int, default=None)
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing rescore outputs instead of appending to them")
    args = ap.parse_args()

    log = get_logger("rescore")
    base = Config()
    runs_dir = Path(args.runs_dir) if args.runs_dir else base.paths.runs

    if args.all:
        targets = [(runs_dir / f"{m}_{args.dim}_{args.seed}") for m in sorted(METHODS)]
        targets = [t for t in targets if (t / "checkpoints" / "final.pt").exists()]
        out_parent = Path(args.out_dir) if args.out_dir else base.paths.artifacts / "baseline"
        pairs = [(t, out_parent / t.name) for t in targets]
    elif args.run:
        run = Path(args.run)
        pairs = [(run, Path(args.out) if args.out else run)]
    else:
        print("pass --run <dir> or --all", file=sys.stderr)
        return 1
    if not pairs:
        print(f"no runs with checkpoints under {runs_dir}", file=sys.stderr)
        return 1

    summary: dict[str, dict] = {}
    for run_dir, out_dir in pairs:
        method, dim, seed = parse_run_id(run_dir.name)
        cfg = Config(dim=dim, seed=seed, tta=args.tta,
                     postproc_min_voxels=args.postproc_min_voxels,
                     max_test_cases=args.max_test_cases,
                     **({"sw_overlap": args.sw_overlap} if args.sw_overlap else {}))
        if args.cache_dir:
            cfg.paths.cache = Path(args.cache_dir)
        log.info(f"{run_dir.name}  tta={cfg.tta} postproc={cfg.postproc_min_voxels} -> {out_dir}")
        try:
            summary[run_dir.name] = rescore(run_dir, out_dir, cfg, method, args.stage, log,
                                            overwrite=args.overwrite)
        except RunExistsError as e:
            print(f"\nrefusing to rescore: {e}", file=sys.stderr)
            return 1

    out = Path(args.out_dir or base.paths.artifacts) / f"rescore_{args.stage}_{args.dim}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({"stage": args.stage, "tta": args.tta,
                   "postproc_min_voxels": args.postproc_min_voxels, "means": summary},
                  f, indent=2, sort_keys=True)
    log.info(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
