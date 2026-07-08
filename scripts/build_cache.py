"""Build the preprocessing cache once, so training reads ready tensors.

Preprocessing is ~6 s/case (load 4 NIfTIs, shift, crop, z-norm), and every epoch of every round
of every method would otherwise redo it. Materialize the deterministic prefix once.

Resumable: a case with a meta.json is skipped, so a killed build (or a dropped Colab session)
picks up where it left off.

    python scripts/build_cache.py                      # everything in the manifest
    python scripts/build_cache.py --max-cases 3        # smoke: 3 train + 3 test per hospital
    python scripts/build_cache.py --workers 8          # parallel

Windows note: the worker is a top-level function taking a picklable tuple, so this works under
`spawn` as well as `fork`. The __main__ guard below is mandatory on Windows.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedbrats.config import Config                                    # noqa: E402
from fedbrats.data import assemble_index, build_case_cache, cache_key, case_cache_dir  # noqa: E402
from fedbrats.logging_utils import get_logger                         # noqa: E402
from fedbrats.partition import cases_for, load_manifest               # noqa: E402


def build_tasks(cfg: Config, manifest: dict, max_cases: int | None) -> list[tuple]:
    tasks = []
    for h in cfg.hospital_ids():
        for split in ("train", "test"):
            ids = cases_for(manifest, h, split)
            if max_cases:
                ids = ids[:max_cases]
            for cid in ids:
                tasks.append((cid, h, split, str(cfg.paths.data_root),
                              str(case_cache_dir(cfg, cid)), cfg.seed, cfg.clip_sigma))
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-cases", type=int, default=None,
                    help="cap cases per hospital per split (smoke runs)")
    ap.add_argument("--workers", type=int, default=0, help="processes (0 = serial)")
    ap.add_argument("--data-root", type=str, default=None, help="override the unzipped data root")
    ap.add_argument("--cache-dir", type=str, default=None, help="override the cache directory")
    args = ap.parse_args()

    cfg = Config()
    if args.data_root:
        cfg.paths.data_root = Path(args.data_root)
    if args.cache_dir:
        cfg.paths.cache = Path(args.cache_dir)

    log = get_logger("build_cache")
    if not cfg.paths.data_root.exists():
        log.error(f"data root not found: {cfg.paths.data_root} "
                  f"(set FEDBRATS_DATA_ROOT or pass --data-root)")
        return 1

    manifest = load_manifest(cfg.paths.manifest)
    tasks = build_tasks(cfg, manifest, args.max_cases)
    root = Path(cfg.paths.cache) / cache_key(cfg)
    log.info(f"cache {root}  ({len(tasks)} cases, key={cache_key(cfg)})")

    t0 = time.time()
    if args.workers and args.workers > 1:
        with mp.Pool(args.workers) as pool:
            for i, msg in enumerate(pool.imap_unordered(build_case_cache, tasks), 1):
                if i % 25 == 0 or i == len(tasks):
                    log.info(f"  {i}/{len(tasks)}  {msg}")
    else:
        for i, task in enumerate(tasks, 1):
            msg = build_case_cache(task)
            if i % 25 == 0 or i == len(tasks):
                log.info(f"  {i}/{len(tasks)}  {msg}")

    index = assemble_index(cfg)
    log.info(f"index: {len(index)} cases in {time.time() - t0:.1f}s -> {root / 'index.json'}")
    return 0


if __name__ == "__main__":       # required on Windows (spawn re-imports this module)
    raise SystemExit(main())
