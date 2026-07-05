"""Uniform per-hospital result records, so every experiment (centralized, local,
FedAvg, FedBN, ...) writes the same schema and `analyze.py` can compare them.

One JSON file per (method, site): results/<method>/<site>.json
"""

from __future__ import annotations

import glob
import json
import os

from . import REGIONS


def write_scores(results_dir: str, method: str, site: str, scores: dict,
                 n_train: int = 0, n_val: int = 0, extra: dict | None = None) -> str:
    """Write one hospital's result record. `scores` has keys mean/TC/WT/ET."""
    out_dir = os.path.join(results_dir, method)
    os.makedirs(out_dir, exist_ok=True)
    record = {
        "method": method,
        "site": site,
        "n_train": n_train,
        "n_val": n_val,
        "dice_mean": scores["mean"],
        **{f"dice_{r}": scores[r] for r in REGIONS},
    }
    if extra:
        record.update(extra)
    path = os.path.join(out_dir, f"{site}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path


def read_all(results_dir: str) -> list[dict]:
    """Load every result record under results_dir/*/*.json."""
    records = []
    for p in sorted(glob.glob(os.path.join(results_dir, "*", "*.json"))):
        with open(p) as f:
            records.append(json.load(f))
    return records
