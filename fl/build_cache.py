"""Build the preprocessing cache ONCE so training reads ready tensors instead of
re-decompressing + re-transforming every epoch/round (the pipeline is CPU-bound —
this makes it GPU-bound). Materializes the deterministic pipeline (per-hospital
shift -> label->regions -> z-norm -> resize) over the UNION of all hospitals' cases;
every downstream step then indexes into the same shared cache.

    uv run python fl/build_cache.py --data-root data/BraTS2021_Training_Data \
        --n-clients 6 --synthetic-shift --cache-dir data/cache

The cache is keyed by (shift profiles, size): change SITE_PROFILES -> new key ->
automatic rebuild, so a stale cache can never poison results.
"""

from __future__ import annotations

import argparse

from braintumor_fl.data import SiteShift, build_preprocess_cache, build_slice_index
from braintumor_fl.partition import case_site_map, get_partitions


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--n-clients", type=int, default=0)
    p.add_argument("--fets-csv", default="")
    p.add_argument("--max-cases", type=int, default=0)
    p.add_argument("--size", type=int, default=192)
    p.add_argument("--synthetic-shift", action="store_true")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--index-cache", default="data/slice_index.csv")
    args = p.parse_args()

    parts = get_partitions(args.data_root, args.n_clients or None,
                           args.fets_csv or None, args.max_cases)
    all_cases = [c for part in parts for c in part]
    site_shift = SiteShift(case_site_map(parts)) if args.synthetic_shift else None
    index = build_slice_index(all_cases, cache_csv=args.index_cache)
    print(f"[build_cache] {len(all_cases)} cases | {len(index)} slices | "
          f"shift={'on' if site_shift else 'off'} | size={args.size}")
    build_preprocess_cache(index, site_shift, args.size, args.cache_dir)


if __name__ == "__main__":
    main()
