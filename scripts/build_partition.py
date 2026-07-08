"""Build (or rebuild) the committed split manifest.

Deterministic: same seed -> identical split. Writes artifacts/splits/partition.json.

    .venv/bin/python scripts/build_partition.py
"""

from __future__ import annotations

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


def main() -> None:
    cfg = Config()
    log = get_logger("partition")

    # find case IDs: prefer the local compressed dir, fall back to the D: unzipped copy
    ids: list[str] = []
    for root in (cfg.paths.local_compressed, cfg.paths.local_unzipped):
        ids = list_case_ids(root)
        if ids:
            log.info(f"found {len(ids)} cases under {root}")
            break
    if not ids:
        log.error("no BraTS2021_* case dirs found (checked local compressed + D: unzipped)")
        raise SystemExit(1)

    manifest = build_partition(ids, cfg)
    out = cfg.paths.splits / "partition.json"
    save_manifest(manifest, out)

    for line in summary_lines(manifest):
        log.info(line)
    log.info(f"manifest written -> {out}")


if __name__ == "__main__":
    main()
