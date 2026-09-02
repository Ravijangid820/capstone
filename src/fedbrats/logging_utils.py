"""Logging: a timestamped console+file logger and a JSONL metrics writer.

Every run produces a human `run.log` (mirrored to console) and a machine-readable
`metrics.jsonl` that plots and the H1/H2/H3 tables read directly. See docs/code.md §3.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


def get_logger(name: str = "fedbrats", log_file: Path | None = None,
               level: int = logging.INFO, mode: str = "a") -> logging.Logger:
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
        fh = logging.FileHandler(log_file, mode=mode)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        logger.addHandler(fh)

    return logger


class RunExistsError(RuntimeError):
    """Raised when a run directory already holds results and would be written into."""


def guard_run_dir(run_dir: Path, overwrite: bool = False) -> None:
    """Refuse to write into a run directory that already has results.

    Without this, a second run of the same method+seed appends its rounds *underneath* the
    first run's inside one metrics.jsonl, and `analyze.py` -- which scores `max(round)` --
    silently averages the two together. The failure is invisible in the output: the file just
    quietly contains two experiments and reports their mean as one.

    The fix at the call site is a `--tag`, not a delete, so the earlier results survive.
    """
    run_dir = Path(run_dir)
    existing = [p.name for p in (run_dir / "metrics.jsonl", run_dir / "per_case.jsonl")
                if p.exists() and p.stat().st_size > 0]
    if not existing or overwrite:
        return
    raise RunExistsError(
        f"{run_dir} already contains {', '.join(existing)}.\n"
        f"Writing here would append a second experiment into the same file and corrupt both.\n"
        f"Use --tag <name> to write alongside it (recommended), or --overwrite to replace it."
    )


class MetricsWriter:
    """Append one JSON object per line to a JSONL file.

    Buffered and flushed per write: a killed Colab session should still leave every round it
    finished, since a partial run that can be read is worth more than a complete one that cannot.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **row) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def write_many(self, rows: list[dict]) -> None:
        if not rows:
            return
        with self.path.open("a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
