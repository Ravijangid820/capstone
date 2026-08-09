"""Run the four methods sequentially for one (dim, seed) and keep going if the session ends.

    python scripts/run_matrix.py --dim 2d --seed 42 --preset v2 --tag v2
    python scripts/run_matrix.py --dim 2d --seed 7 --preset v2 --tag v2 --methods fedavg fedbn

A full 2D matrix is several hours on the 3050, which is longer than a terminal reliably stays
open. This runs the methods in one detachable process, appends a progress line per method to
`<runs>/<tag>/matrix.log`, and **skips methods that already finished** -- so re-invoking after an
interruption resumes rather than restarting or, worse, colliding with the completed runs.

"Finished" means the run wrote `summary.json`, which happens last. A method killed mid-run leaves
a directory without one; that directory is *not* silently reused, because `run_experiment.py`
refuses to write into a populated run dir. Delete it or pass `--overwrite` -- an explicit choice,
which is the point.

Centralized runs first on purpose: it is the sanity gate. If the pooled ceiling does not climb
well past 0.7 WT the fault is in the data pipeline or the loss, not in federation, and the
remaining three runs would be hours spent measuring the same bug three more times.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedbrats.config import PRESETS, Config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ORDER = ["centralized", "local", "fedavg", "fedbn"]     # sanity gate first -- see the docstring


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dim", default="2d", choices=("2d", "3d"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preset", default="v2", choices=sorted(PRESETS))
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--methods", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="everything after this is forwarded to run_experiment.py verbatim")
    args = ap.parse_args()

    cfg = Config(dim=args.dim, seed=args.seed, tag=args.tag)
    root = cfg.run_dir("x").parent
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "matrix.log"

    def log(msg: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
        line = f"{stamp}  {msg}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    methods = [m for m in ORDER if m in args.methods]
    log(f"matrix start  dim={args.dim} seed={args.seed} preset={args.preset} tag={args.tag}")
    log(f"  methods: {', '.join(methods)}")

    t_all = time.time()
    for method in methods:
        run_dir = cfg.run_dir(method)
        if (run_dir / "summary.json").exists() and not args.overwrite:
            log(f"SKIP  {method:<12} already complete ({run_dir.name})")
            continue

        cmd = [sys.executable, str(REPO / "scripts" / "run_experiment.py"),
               "--method", method, "--dim", args.dim, "--seed", str(args.seed),
               "--preset", args.preset, "--tag", args.tag, *args.extra]
        if args.overwrite:
            cmd.append("--overwrite")

        log(f"START {method:<12} {' '.join(cmd[1:])}")
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=REPO).returncode
        mins = (time.time() - t0) / 60
        if rc != 0:
            log(f"FAIL  {method:<12} exit {rc} after {mins:.1f} min -- stopping")
            return rc
        log(f"DONE  {method:<12} {mins:.1f} min")

    log(f"matrix complete in {(time.time() - t_all) / 60:.1f} min -> {root}")
    log(f"next:  python scripts/compare_runs.py --dim {args.dim} --seed {args.seed} "
        f"--new {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
