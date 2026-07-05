"""Map BraTS cases to federated clients ("hospitals").

Two modes:
- **even** round-robin split (roughly IID) — for validating the FL plumbing.
- **FeTS** real institutional partitioning from the challenge CSV — genuine
  non-IID hospitals, the setting our personalization method actually targets.

Both are deterministic given their inputs, so the job runner and each client
independently reconstruct the *same* partitions.
"""

from __future__ import annotations

import csv
import os

from .data import find_cases


def even_partition(cases: list[str], n_clients: int) -> list[list[str]]:
    """Round-robin the cases across n_clients (deterministic, ~balanced)."""
    return [cases[i::n_clients] for i in range(n_clients)]


def fets_partition(
    data_root: str,
    csv_path: str,
    id_col: str = "Subject_ID",
    part_col: str = "Partition_ID",
    min_cases: int = 5,
) -> list[list[str]]:
    """Group cases by FeTS institution from the partitioning CSV.

    The CSV maps each subject to an institution id. We match subject ids to case
    dirs by basename (exact or suffix), and drop institutions with < min_cases so
    a hospital always has enough data to train + validate.
    """
    by_base = {os.path.basename(c): c for c in find_cases(data_root)}

    def match(subject_id: str) -> str | None:
        if subject_id in by_base:
            return by_base[subject_id]
        for base, path in by_base.items():  # tolerate "00000" vs "BraTS2021_00000"
            if base.endswith(subject_id) or subject_id.endswith(base):
                return path
        return None

    groups: dict[str, list[str]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            case = match(str(row[id_col]).strip())
            if case is not None:
                groups.setdefault(str(row[part_col]).strip(), []).append(case)

    parts = [sorted(v) for _, v in sorted(groups.items()) if len(v) >= min_cases]
    if not parts:
        raise SystemExit(f"No FeTS partitions with >= {min_cases} cases from {csv_path!r}")
    return parts


def get_partitions(
    data_root: str,
    n_clients: int | None = None,
    fets_csv: str | None = None,
    max_cases: int = 0,
) -> list[list[str]]:
    """Unified entry: FeTS split if csv given, else even split into n_clients."""
    if fets_csv:
        return fets_partition(data_root, fets_csv)
    if not n_clients:
        raise ValueError("provide either fets_csv or n_clients")
    cases = find_cases(data_root)
    if max_cases:
        cases = cases[:max_cases]
    return even_partition(cases, n_clients)


def client_cases(
    data_root: str,
    client_index: int,
    n_clients: int | None = None,
    fets_csv: str | None = None,
    max_cases: int = 0,
) -> list[str]:
    return get_partitions(data_root, n_clients, fets_csv, max_cases)[client_index]


def case_site_map(partitions: list[list[str]]) -> dict[str, int]:
    """Flatten a partition into {case_dir: 0-based site index}. Used to drive the
    synthetic per-hospital scanner shift consistently across every method: because
    partitions are deterministic, each case maps to the same site everywhere."""
    return {case: i for i, cases in enumerate(partitions) for case in cases}


def partitioned_splits(partitions: list[list[str]], val_frac: float = 0.2, seed: int = 42):
    """Per-hospital case-level train/val splits, aggregated for the centralized run.

    Returns:
        all_train: union of every hospital's train cases (the centralized training set)
        all_val:   union of every hospital's val cases (held out from centralized training)
        per_site:  [(site_name, train_cases_i, val_cases_i), ...]
    Guarantees the centralized model never trains on any hospital's val cases.
    """
    from .data import case_split

    all_train, all_val, per_site = [], [], []
    for i, cases in enumerate(partitions):
        tr, va = case_split(cases, val_frac, seed)
        all_train += tr
        all_val += va
        per_site.append((f"site-{i + 1}", tr, va))
    return all_train, all_val, per_site
