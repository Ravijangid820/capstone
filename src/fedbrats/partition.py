"""Deterministic partition: all cases -> K hospitals -> train/test.

Partition-then-split: each hospital owns a train AND a test set drawn from its own (shifted)
distribution, which the per-hospital H2/H3 claims require. The assignment is written once to a
committed manifest (`artifacts/splits/partition.json`) so every experiment uses the identical
split. The `train_per_hospital` knob is applied later at sampling time, NOT here — the manifest
stays the full, stable assignment. See docs/data-pipeline.md §1.
"""

from __future__ import annotations

import glob
import json
import os
import random
from pathlib import Path

from .config import Config


def list_case_ids(root: Path) -> list[str]:
    """Sorted `BraTS2021_XXXXX` case IDs found directly under `root`."""
    return sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(str(root), "BraTS2021_*"))
        if os.path.isdir(p)
    )


def build_partition(case_ids: list[str], cfg: Config) -> dict:
    """Assign each case to a hospital and a train/test split, deterministically."""
    ids = sorted(case_ids)                     # stable base order regardless of input order
    rng = random.Random(cfg.seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)

    K = cfg.num_hospitals
    n = len(shuffled)
    # near-equal contiguous chunks; remainder spread over the first hospitals
    sizes = [n // K + (1 if i < n % K else 0) for i in range(K)]
    hospitals = cfg.hospital_ids()

    assignment: dict[str, dict] = {}
    per_hospital: dict[str, dict] = {}
    idx = 0
    for i, (h, size) in enumerate(zip(hospitals, sizes)):
        chunk = shuffled[idx: idx + size]
        idx += size

        # deterministic per-hospital test split (index-based seed — NOT hash(), which is
        # randomized per process and would break reproducibility across sessions)
        hrng = random.Random(cfg.seed * 1000 + i)
        chunk_shuffled = chunk[:]
        hrng.shuffle(chunk_shuffled)
        test_ids = set(chunk_shuffled[: cfg.test_per_hospital])

        is_outlier = h == cfg.outlier_hospital
        for cid in chunk:
            assignment[cid] = {
                "hospital": h,
                "split": "test" if cid in test_ids else "train",
                "is_outlier": is_outlier,
            }
        per_hospital[h] = {
            "total": size,
            "train": size - len(test_ids),
            "test": len(test_ids),
            "is_outlier": is_outlier,
        }

    return {
        "meta": {
            "seed": cfg.seed,
            "num_hospitals": K,
            "outlier": cfg.outlier_hospital,
            "test_per_hospital": cfg.test_per_hospital,
            "train_per_hospital_knob": cfg.train_per_hospital,
            "n_cases": n,
        },
        "per_hospital": per_hospital,
        "assignment": assignment,
    }


def save_manifest(manifest: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_manifest(path: Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def cases_for(manifest: dict, hospital: str, split: str) -> list[str]:
    """Case IDs for one hospital + split (e.g. 'H4', 'train'), sorted."""
    return sorted(
        cid for cid, a in manifest["assignment"].items()
        if a["hospital"] == hospital and a["split"] == split
    )


def summary_lines(manifest: dict) -> list[str]:
    """Human-readable summary of a manifest."""
    m = manifest["meta"]
    lines = [
        f"{m['n_cases']} cases -> {m['num_hospitals']} hospitals "
        f"(seed {m['seed']}, outlier {m['outlier']}, {m['test_per_hospital']} test/hospital)"
    ]
    tot_train = tot_test = 0
    for h, s in manifest["per_hospital"].items():
        tag = "   <-- OUTLIER" if s["is_outlier"] else ""
        lines.append(f"  {h}: {s['total']:>4}  (train {s['train']:>4}, test {s['test']:>3}){tag}")
        tot_train += s["train"]
        tot_test += s["test"]
    lines.append(f"  TOTAL: train {tot_train}, test {tot_test}")
    return lines
