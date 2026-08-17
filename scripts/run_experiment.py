"""Run one experiment: centralized (E0), local-only (E1), FedAvg (E2) or FedBN (E3).

    python scripts/run_experiment.py --method fedbn --dim 2d                  # baseline recipe
    python scripts/run_experiment.py --method fedbn --dim 2d --preset v2 --tag v2
    python scripts/run_experiment.py --method local --rounds 2 --max-train-cases 3 --max-test-cases 2

All four methods share the split manifest, the seed, the init and the cache -- only the training
*procedure* differs, which is what makes the H1/H2/H3 comparison clean. Results stream to
artifacts/runs/[<tag>/]<run_id>/metrics.jsonl.

**Presets.** Every default reproduces the frozen baseline, so an un-flagged run today still
produces the numbers in artifacts/baseline/. `--preset v2` turns on the improved recipe (cosine
LR across rounds, augmentation, flip-TTA, component filtering, validation-based selection).
Explicit flags beat the preset, and the resulting values -- not the preset name -- are what get
written to the run's config.json.

**Tags.** `--tag v2` writes to artifacts/runs/v2/<run_id>/ instead of artifacts/runs/<run_id>/.
Without it, a rerun of a method+seed that already exists is refused rather than appended to.
"""

from __future__ import annotations

import argparse
import sys
from argparse import BooleanOptionalAction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedbrats.config import PRESETS, apply_preset          # noqa: E402
from fedbrats.federated import METHODS, run                # noqa: E402
from fedbrats.logging_utils import RunExistsError          # noqa: E402

# argparse dest -> Config field. Anything not listed here is a script flag, not a config knob.
NON_CONFIG = {"method", "cache_dir", "preset", "overwrite"}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--preset", default="baseline", choices=sorted(PRESETS),
                    help="named recipe; explicit flags below override it (default: baseline)")
    ap.add_argument("--tag", default=None,
                    help="namespace the output dir: artifacts/runs/<tag>/<run_id>/")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing run dir instead of refusing to write into it")

    g = ap.add_argument_group("schedule / model")
    g.add_argument("--dim", default="2d", choices=("2d", "3d"))
    g.add_argument("--rounds", type=int, default=None, help="R communication rounds")
    g.add_argument("--local-epochs", type=int, default=None, help="E local epochs per round")
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--lr-schedule", default=None, choices=("constant", "cosine"),
                   help="per-round LR decay; a fresh Adam each round makes this the only schedule")
    g.add_argument("--lr-min-factor", type=float, default=None)
    g.add_argument("--lr-anneal-rounds", type=int, default=None,
                   help="rounds the cosine takes to reach its floor (default: the whole run)")
    g.add_argument("--eval-test-every", type=int, default=None,
                   help="score the full test set every Nth round; the estimator's rounds always run")
    g.add_argument("--batch-size", type=int, default=None)
    g.add_argument("--base-channels", type=int, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--num-workers", type=int, default=None)

    g = ap.add_argument_group("augmentation (train only)")
    g.add_argument("--augment", action=BooleanOptionalAction, default=None)
    g.add_argument("--aug-flip-p", type=float, default=None)
    g.add_argument("--aug-rot90-p", type=float, default=None)
    g.add_argument("--aug-intensity-p", type=float, default=None)
    g.add_argument("--aug-noise-std", type=float, default=None)

    g = ap.add_argument_group("inference")
    g.add_argument("--tta", action=BooleanOptionalAction, default=None,
                   help="flip test-time augmentation on the final evaluation")
    g.add_argument("--postproc-min-voxels", type=int, default=None,
                   help="drop connected components below this size; 0 = off")
    g.add_argument("--sw-overlap", type=float, default=None, help="3d sliding-window overlap")

    g = ap.add_argument_group("selection / data")
    g.add_argument("--val-per-hospital", type=int, default=None,
                   help="validation cases/hospital, taken AFTER the train cap; 0 = off")
    g.add_argument("--select-by", default=None, choices=("last", "best_val"))
    g.add_argument("--report-last-k", type=int, default=None,
                   help="declared analysis estimator: rounds averaged for the headline figure")
    g.add_argument("--train-per-hospital", type=int, default=None)
    g.add_argument("--max-train-cases", type=int, default=None, help="cap train cases/hospital")
    g.add_argument("--max-test-cases", type=int, default=None, help="cap test cases/hospital")
    g.add_argument("--cache-dir", type=str, default=None)
    g.add_argument("--device", default=None)
    return ap


def main() -> int:
    args = build_parser().parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None and k not in NON_CONFIG}
    cfg = apply_preset(args.preset, overrides)
    if args.cache_dir:
        cfg.paths.cache = Path(args.cache_dir)

    try:
        run(cfg, args.method, overwrite=args.overwrite)
    except RunExistsError as e:
        print(f"\nrefusing to run: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":       # required on Windows (spawn re-imports this module)
    raise SystemExit(main())
