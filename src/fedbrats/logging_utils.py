"""Logging: a timestamped console+file logger and a JSONL metrics writer.

Every run produces a human `run.log` (mirrored to console) and a machine-readable
`metrics.jsonl` that plots and the H1/H2/H3 tables read directly. See docs/specs.md §8.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


def get_logger(name: str = "fedbrats", log_file: Path | None = None,
               level: int = logging.INFO) -> logging.Logger:
    """A logger writing INFO lines to stdout, and to `log_file` if given."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        logger.addHandler(fh)

    return logger


class MetricsWriter:
    """Append one JSON object per line to a metrics.jsonl file."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **row) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
