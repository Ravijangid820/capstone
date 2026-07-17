"""Smoke tests for fedbrats.partition.build_partition."""

from fedbrats.config import Config
from fedbrats.partition import build_partition


def _make_case_ids(n: int = 100) -> list[str]:
    """Create n fake BraTS-style case IDs."""
    return [f"BraTS2021_{i:05d}" for i in range(n)]


def test_deterministic():
    """Same seed produces the same partition."""
    ids = _make_case_ids(100)
    cfg = Config(seed=42)
    p1 = build_partition(ids, cfg)
    p2 = build_partition(ids, cfg)
    assert p1["assignment"] == p2["assignment"]


def test_all_ids_appear_exactly_once():
    """Every case ID appears exactly once across all hospitals."""
    ids = _make_case_ids(100)
    cfg = Config(seed=42)
    partition = build_partition(ids, cfg)
    assigned_ids = sorted(partition["assignment"].keys())
    assert assigned_ids == sorted(ids)


def test_hospital_count():
    """Default config produces 4 hospitals."""
    ids = _make_case_ids(100)
    cfg = Config(seed=42)
    partition = build_partition(ids, cfg)
    assert len(partition["per_hospital"]) == 4
    assert set(partition["per_hospital"].keys()) == {"H1", "H2", "H3", "H4"}
