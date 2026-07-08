"""Run one experiment: centralized (E0), local-only (E1), FedAvg (E2) or FedBN (E3).

    python scripts/run_experiment.py --method fedbn --dim 2d
    python scripts/run_experiment.py --method local --rounds 2 --max-train-cases 3 --max-test-cases 2

All four methods share the split manifest, the seed, the init and the cache -- only the training
*procedure* differs, which is what makes the H1/H2/H3 comparison clean. Results stream to
artifacts/runs/<run_id>/metrics.jsonl.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedbrats.config import Config          # noqa: E402
from fedbrats.federated import METHODS, run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--dim", default="2d", choices=("2d", "3d"))
    ap.add_argument("--rounds", type=int, default=None, help="R communication rounds")
    ap.add_argument("--local-epochs", type=int, default=None, help="E local epochs per round")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--max-train-cases", type=int, default=None, help="cap train cases/hospital")
    ap.add_argument("--max-test-cases", type=int, default=None, help="cap test cases/hospital")
    ap.add_argument("--cache-dir", type=str, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    overrides = {k: v for k, v in vars(args).items()
                 if v is not None and k not in ("method", "cache_dir")}
    cfg = Config(**overrides)
    if args.cache_dir:
        cfg.paths.cache = Path(args.cache_dir)

    run(cfg, args.method)
    return 0


if __name__ == "__main__":       # required on Windows (spawn re-imports this module)
    raise SystemExit(main())
