"""Central configuration: paths, the hospital split, and model/training defaults.

One `Config` drives every run. The same seed + the committed split manifest make runs
comparable across methods and across the 2D/3D backbones. See docs/specs.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


@dataclass
class Paths:
    """Where data lives and where outputs go (see docs/data.md §3)."""

    # local (this machine)
    local_compressed: Path = _REPO / "data" / "BraTS2021_Training_Data"  # 1251 .nii.gz case dirs
    local_unzipped: Path = Path("/mnt/d/capstone_data/unzipped")         # D: raw .nii
    # colab
    drive_unzipped: Path = Path("/content/drive/MyDrive/capstone/unzipped")
    # outputs
    artifacts: Path = _REPO / "artifacts"

    @property
    def splits(self) -> Path:
        return self.artifacts / "splits"

    @property
    def runs(self) -> Path:
        return self.artifacts / "runs"


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

    paths: Paths = field(default_factory=Paths)

    def hospital_ids(self) -> list[str]:
        return [f"H{i + 1}" for i in range(self.num_hospitals)]
