"""Build (or rebuild) the committed split manifest.

Deterministic: same seed -> identical split, on any OS. The manifest is committed, so this
normally never needs running -- but re-running it is a free cross-platform determinism check:
`git status` must report no change to artifacts/splits/partition.json.

    python scripts/build_partition.py
    python scripts/build_partition.py --data-root D:/data/unzipped
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedbrats.config import Config  # noqa: E402
from fedbrats.logging_utils import get_logger  # noqa: E402
from fedbrats.partition import (  # noqa: E402
    build_partition,
    list_case_ids,
    save_manifest,
    summary_lines,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=str, default=None,
                    help="dir holding the BraTS2021_* case dirs (default: per-platform, see config)")
    args = ap.parse_args()

    cfg = Config()
    if args.data_root:
        cfg.paths.data_root = Path(args.data_root)

    log = get_logger("partition")

    ids = list_case_ids(cfg.paths.data_root)
    if not ids:
        log.error(f"no BraTS2021_* case dirs under {cfg.paths.data_root} "
                  f"(set FEDBRATS_DATA_ROOT or pass --data-root)")
        return 1
    log.info(f"found {len(ids)} cases under {cfg.paths.data_root}")

    manifest = build_partition(ids, cfg)
    save_manifest(manifest, cfg.paths.manifest)

    for line in summary_lines(manifest):
        log.info(line)
    log.info(f"manifest written -> {cfg.paths.manifest}")
    return 0


if __name__ == "__main__":       # required on Windows (spawn re-imports this module)
    raise SystemExit(main())
