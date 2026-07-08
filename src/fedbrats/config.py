"""Central configuration: paths, the hospital split, and model/training defaults.

One `Config` drives every run. The same seed + the committed split manifest make runs
comparable across methods and across the 2D/3D backbones. See docs/specs.md.

Paths resolve per-platform (WSL2 and native Windows are both supported) and can be
overridden with `FEDBRATS_DATA_ROOT` / `FEDBRATS_CACHE_DIR`. See docs/environments.md.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

# The four modalities, in channel order. Fixed everywhere (cache, model input).
MODALITIES = ("flair", "t1", "t1ce", "t2")
REGIONS = ("wt", "tc", "et")


def _default_data_root() -> Path:
    """Unzipped BraTS cases (1251 dirs of `.nii`). Same physical D: drive from WSL2 and Windows.

    `load_case` reads `.nii` and `.nii.gz` alike, so pointing this at the compressed set
    (`D:/data/BraTS2021_Training_Data`, 13 GB) also works -- slower to decode, far less I/O.
    """
    if env := os.environ.get("FEDBRATS_DATA_ROOT"):
        return Path(env)
    if sys.platform == "win32":
        return Path("D:/data/unzipped")
    if Path("/content/drive/MyDrive/capstone/unzipped").exists():   # colab
        return Path("/content/drive/MyDrive/capstone/unzipped")
    return Path("/mnt/d/data/unzipped")


def _default_cache_dir() -> Path:
    """Preprocessed-tensor cache.

    Defaults are deliberately *small-run* friendly: a smoke cache lands in the repo's
    artifacts/. A full cache is ~28-46 GB and must NOT land on the WSL VHDX (it is backed
    by a nearly-full C:) -- point FEDBRATS_CACHE_DIR at D: or /content for those.
    """
    if env := os.environ.get("FEDBRATS_CACHE_DIR"):
        return Path(env)
    if Path("/content").exists():                                    # colab: local SSD
        return Path("/content/cache")
    return _REPO / "artifacts" / "cache"


def _default_workers() -> int:
    """DataLoader workers. Windows uses spawn (re-imports per worker) -- 0 is far faster."""
    return 0 if sys.platform == "win32" else 4


@dataclass
class Paths:
    """Where data lives and where outputs go (see docs/data.md §3)."""

    data_root: Path = field(default_factory=_default_data_root)
    cache: Path = field(default_factory=_default_cache_dir)
    artifacts: Path = _REPO / "artifacts"

    @property
    def splits(self) -> Path:
        return self.artifacts / "splits"

    @property
    def runs(self) -> Path:
        return self.artifacts / "runs"

    @property
    def manifest(self) -> Path:
        return self.splits / "partition.json"


@dataclass
class Config:
    # reproducibility
    seed: int = 42

    # hospital split (see docs/data-pipeline.md §1)
    num_hospitals: int = 4
    outlier_hospital: str = "H4"          # designated outlier (strongest scanner shift)
    test_per_hospital: int = 62           # held-out test cases per hospital
    train_per_hospital: int | None = 150  # runtime cap on train cases/hospital; None = use all

    # model / backbone
    dim: str = "2d"                       # "2d" | "3d"
    base_channels: int | None = None      # None -> 32 for 2d, 16 for 3d

    # preprocessing
    clip_sigma: float = 5.0

    # federated schedule (see docs/federated-learning.md)
    rounds: int = 20                      # R communication rounds
    local_epochs: int = 1                 # E local epochs per round
    # NOTE: local-only and centralized train R*E epochs total -- matched compute, so H1
    # tests collaboration rather than a longer training budget. See docs/experiments.md §3.

    # optimization
    lr: float = 1e-3
    batch_size: int | None = None         # None -> 8 for 2d, 1 for 3d
    num_workers: int = field(default_factory=_default_workers)

    # sampling
    slices_per_case: int = 8              # 2d: slices drawn per case per epoch
    patches_per_case: int = 2             # 3d: patches drawn per case per epoch
    tumor_frac: float = 0.7               # P(sample a tumour-bearing slice/patch)
    train_hw: int = 192                   # 2d: train crop (in-plane), multiple of 16
    patch_size: int = 96                  # 3d: cubic patch edge

    # smoke / scoping knobs
    max_train_cases: int | None = None    # cap train cases per hospital (smoke runs)
    max_test_cases: int | None = None     # cap test cases per hospital (smoke runs)

    device: str = "cuda"

    paths: Paths = field(default_factory=Paths)

    # ---- derived ---------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.dim not in ("2d", "3d"):
            raise ValueError(f"dim must be '2d' or '3d', got {self.dim!r}")
        if self.base_channels is None:
            self.base_channels = 32 if self.is_2d else 16
        if self.batch_size is None:
            self.batch_size = 8 if self.is_2d else 1

    @property
    def is_2d(self) -> bool:
        return self.dim == "2d"

    @property
    def total_epochs(self) -> int:
        """Matched compute budget: what one hospital sees across the whole FL run."""
        return self.rounds * self.local_epochs

    def hospital_ids(self) -> list[str]:
        return [f"H{i + 1}" for i in range(self.num_hospitals)]

    def run_id(self, method: str) -> str:
        return f"{method}_{self.dim}_{self.seed}"
