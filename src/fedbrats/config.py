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
    """BraTS cases: 1251 dirs of `.nii` (unzipped, 114 GB) or `.nii.gz` (as shipped, 13 GB).

    `load_case` reads both, so either layout works; unzipped decodes ~2x faster. Probe the
    known locations rather than hardcoding one -- the unzipped copy has lived under both
    `D:/data/` and the repo's own `data/`. Override with FEDBRATS_DATA_ROOT.
    """
    if env := os.environ.get("FEDBRATS_DATA_ROOT"):
        return Path(env)
    external = ([Path("D:/data/unzipped")] if sys.platform == "win32" else
                [Path("/content/drive/MyDrive/capstone/unzipped"),    # colab
                 Path("/content/drive/MyDrive/capstone/BraTS2021_Training_Data"),
                 Path("/mnt/d/data/unzipped")])                       # wsl2 view of D:
    candidates = [*external,
                  _REPO / "data" / "unzipped",                        # preferred: fast, local
                  _REPO / "data" / "BraTS2021_Training_Data"]         # fallback: compressed
    return next((c for c in candidates if c.exists()), candidates[-1])


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

    # run identity. `tag` namespaces the output directory (artifacts/runs/<tag>/<run_id>) so a
    # second run of the same method+seed lands beside the first instead of appending into it.
    # Every knob below is written to the run's config.json, so a run is described by its own
    # output rather than by whatever command someone remembers typing.
    tag: str | None = None

    # hospital split (see docs/data-pipeline.md §1)
    num_hospitals: int = 4
    outlier_hospital: str = "H4"          # designated outlier (strongest scanner shift)
    test_per_hospital: int = 62           # held-out test cases per hospital
    train_per_hospital: int | None = 150  # runtime cap on train cases/hospital; None = use all
    val_per_hospital: int = 0             # validation cases/hospital, taken from the cases AFTER
    #                                       the train cap so the training set is unchanged; 0 = off

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

    # Learning-rate schedule ACROSS rounds (not within an epoch). A fresh Adam is built every
    # round -- standard for FL, since optimizer state is not transmitted -- so with a constant LR
    # every round takes full-size steps forever and the plateau oscillates by more than the
    # effects under test. "cosine" decays the per-round LR from `lr` to `lr * lr_min_factor`.
    lr_schedule: str = "constant"         # "constant" | "cosine"
    lr_min_factor: float = 0.05
    # Rounds the cosine takes to reach its floor. None = the whole run (the usual formulation).
    # Setting it shorter than `rounds` decouples "how long we train" from "how fast we anneal":
    # v4 stretched a 25-round cosine over 40 rounds and lost 0.0099 to v3, because the late-curve
    # rise that motivated the longer run was the anneal consolidating the model, not headroom
    # from extra steps. With this set, the schedule bottoms out on v3's timetable and the extra
    # rounds are spent at the floor -- which is the actual question "does more low-LR training
    # help?" asked in isolation.
    lr_anneal_rounds: int | None = None

    # sampling
    slices_per_case: int = 8              # 2d: slices drawn per case per epoch
    patches_per_case: int = 2             # 3d: patches drawn per case per epoch
    tumor_frac: float = 0.7               # P(sample a tumour-bearing slice/patch)
    train_hw: int = 192                   # 2d: train crop (in-plane), multiple of 16
    patch_size: int = 96                  # 3d: cubic patch edge

    # augmentation (train only; never applied to the evaluation path)
    augment: bool = False
    aug_flip_p: float = 0.5               # P(flip) per spatial axis
    aug_rot90_p: float = 0.0              # 2d only: P(k*90 deg in-plane rotation)
    aug_intensity_p: float = 0.0          # P(per-channel scale+shift jitter)
    aug_intensity_scale: float = 0.1      # scale ~ U(1-s, 1+s)
    aug_intensity_shift: float = 0.1      # shift ~ U(-t, +t), in z-score units
    aug_noise_std: float = 0.0            # additive Gaussian sigma, in z-score units

    # inference
    tta: bool = False                     # flip test-time augmentation (final round only: 4x cost)
    postproc_min_voxels: int = 0          # drop connected components smaller than this; 0 = off
    sw_overlap: float = 0.25              # 3d: sliding-window overlap
    # Inference batch sizes. Evaluation is ~half of every round's wall clock and ran at batch 8
    # (2d) / 1 (3d) while using 0.28 GB of a 4 GB card -- the GPU was idling through the single
    # most expensive phase. Raising these is ~2.7x faster on the forward pass.
    #
    # These do NOT change what is computed: the model is in eval() mode, so BatchNorm reads its
    # running statistics and every slice's output is independent of what it shares a batch with.
    # cuDNN does pick different kernels per batch size, which moves probabilities by ~3e-4 and
    # per-case Dice by <=1.5e-4 -- the fourth decimal, far below any effect under study. Defaults
    # stay at the original values so previously published runs remain bit-reproducible; raise them
    # explicitly for new work.
    eval_batch_size: int = 8              # 2d: slices per forward pass
    sw_batch_size: int = 1                # 3d: windows per forward pass

    # model selection
    select_by: str = "last"               # "last" | "best_val" (needs val_per_hospital > 0)
    report_last_k: int = 1                # rounds averaged for the headline figure; 1 = final only

    # Test-evaluation cadence. Scoring all 248 test volumes every round is ~70% of a round's wall
    # clock, and most of those rounds are mid-curve points nobody reads a number off. Raising this
    # buys rounds -- which is what the models actually need -- at the cost of a sparser learning
    # curve. The rounds the analysis DOES read are always scored: the final `report_last_k` rounds
    # (the estimator) and the selected model's own final evaluation. Validation is unaffected and
    # runs every round, since checkpoint selection needs it.
    eval_test_every: int = 1              # 1 = every round (the original behaviour)

    # smoke / scoping knobs
    max_train_cases: int | None = None    # cap train cases per hospital (smoke runs)
    max_test_cases: int | None = None     # cap test cases per hospital (smoke runs)

    device: str = "cuda"

    paths: Paths = field(default_factory=Paths)

    # ---- derived ---------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.dim not in ("2d", "3d"):
            raise ValueError(f"dim must be '2d' or '3d', got {self.dim!r}")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(f"lr_schedule must be 'constant' or 'cosine', got {self.lr_schedule!r}")
        if self.select_by not in ("last", "best_val"):
            raise ValueError(f"select_by must be 'last' or 'best_val', got {self.select_by!r}")
        if self.select_by == "best_val" and self.val_per_hospital <= 0:
            raise ValueError("select_by='best_val' needs val_per_hospital > 0")
        if not 1 <= self.report_last_k <= self.rounds:
            raise ValueError(f"report_last_k must be in [1, rounds={self.rounds}], "
                             f"got {self.report_last_k}")
        if self.eval_test_every < 1:
            raise ValueError(f"eval_test_every must be >= 1, got {self.eval_test_every}")
        if self.base_channels is None:
            self.base_channels = 32 if self.is_2d else 16
        if self.batch_size is None:
            self.batch_size = 8 if self.is_2d else 1
        if not self.is_2d:
            # 3D never rotates: the three axes are anatomically distinct (axial / sagittal /
            # coronal), so a 90-degree turn between them produces an orientation no scanner in
            # the study emits. Zeroed here rather than just ignored downstream, so the run's
            # config.json describes what actually ran instead of implying rotations happened.
            self.aug_rot90_p = 0.0

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

    def run_dir(self, method: str) -> Path:
        """Where this run writes. `tag` namespaces it so reruns cannot collide with the baseline."""
        base = Path(self.paths.runs)
        return base / self.tag / self.run_id(method) if self.tag else base / self.run_id(method)

    def scores_test(self, rnd: int) -> bool:
        """Whether round `rnd` gets a full test evaluation.

        The final `report_last_k` rounds are always scored, because they are exactly the rounds the
        reported estimator averages -- thinning those would change the headline number rather than
        just the curve's resolution.
        """
        if rnd > self.rounds - self.report_last_k:
            return True
        return rnd % self.eval_test_every == 0 or rnd == 1

    def round_lr(self, rnd: int) -> float:
        """LR for communication round `rnd` (1-based). Cosine decays lr -> lr*lr_min_factor.

        The decay spans `lr_anneal_rounds` when set, otherwise the whole run; past that point the
        LR holds at the floor rather than rising again.
        """
        if self.lr_schedule == "constant" or self.rounds <= 1:
            return self.lr
        import math
        span = self.lr_anneal_rounds or self.rounds
        progress = min(1.0, (rnd - 1) / max(1, span - 1))
        floor = self.lr * self.lr_min_factor
        return floor + (self.lr - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def to_dict(self) -> dict:
        """JSON-serializable snapshot of every knob, for the run's config.json.

        Typed rather than stringified: the run-comparison tool diffs two of these to state what
        actually changed between a baseline and a rerun, which only works if 25 and "25" are not
        the same value.
        """
        out: dict = {}
        for k, v in vars(self).items():
            if k == "paths":
                out["paths"] = {n: str(p) for n, p in
                                (("data_root", v.data_root), ("cache", v.cache),
                                 ("artifacts", v.artifacts))}
            else:
                out[k] = v
        return out


# --------------------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------------------
# Every field above defaults to the value the *published baseline* ran with, so
# `run_experiment.py --method X --dim 2d` still reproduces the frozen numbers bit for bit.
# Improvements are opt-in through a named preset, which is what keeps "before" reproducible
# after "after" exists. A preset is recorded in the run's config.json field by field, so the
# comparison never depends on this table still saying what it said at run time.

PRESETS: dict[str, dict] = {
    "baseline": {},
    # v3 supersedes v2. Three changes, each traceable to a measurement rather than a hunch:
    #
    # 1. **Geometric augmentation only.** v2's intensity jitter and Gaussian noise perturb gamma,
    #    bias and blur -- the exact channels the synthetic scanner shift uses. Augmenting along
    #    the experimental variable dilutes the signal FedBN's BatchNorm layers exist to capture
    #    and forces FedAvg's one global model into a worse compromise. It shows in the numbers:
    #    FedAvg was the only 2D method to lose (-0.0025 on H1-H3, -0.0120 on the shifted H4),
    #    and in 3D -- where 2 patches/case makes an epoch tiny enough for noise to dominate --
    #    three of four methods lost. Flips and rotations are label-preserving and leave intensity
    #    statistics untouched, so they regularize without competing with the thing under test.
    #
    # 2. **Use the data that was there all along.** 150 of ~250 available training cases per
    #    hospital were being used. 230 train + 20 val = 250 fits every hospital (H4 has 250) and
    #    costs almost nothing in wall clock: training is ~10% of a round, evaluation dominates.
    #
    # 3. **Keep everything that measurably worked.** Cosine LR cut plateau oscillation 2.7-5.9x.
    #    TTA and component filtering, measured on frozen checkpoints with no retraining, gained
    #    +0.003..+0.008 WT and up to +0.014 ET on every 2D method with every CI excluding zero.
    #    Validation-based selection repeatedly picked a better round than the last one.
    "v3": {
        "lr_schedule": "cosine",
        "lr_min_factor": 0.05,
        "augment": True,
        "aug_flip_p": 0.5,
        "aug_rot90_p": 0.5,           # 2D only; zeroed for 3D in __post_init__
        "aug_intensity_p": 0.0,       # <- the regression: augmenting along the scanner-shift axis
        "aug_noise_std": 0.0,         # <-
        "tta": True,
        "postproc_min_voxels": 50,
        "train_per_hospital": 230,    # was 150; 40% of the training pool had never been used
        "val_per_hospital": 20,
        "select_by": "best_val",
        "report_last_k": 5,
    },
    # v4 = v3 trained longer, because the v3 runs proved they were not finished. Late-curve slope
    # (mean WT over rounds 21-25 minus rounds 16-20) was positive in all eight v3 runs -- +0.0051
    # to +0.0110 -- and validation was still rising in every one of them too, so this is
    # undertraining rather than test-set noise. Capacity and augmentation were not the binding
    # constraint; the round budget was.
    #
    # 40 rounds costs 1.6x the wall clock at v3's evaluation cadence, so the cadence pays for it:
    # scoring all 248 test volumes is ~70% of a round, and most rounds are mid-curve points no
    # number is read from. eval_test_every=3 thins those while still scoring round 1, every third
    # round, and every round the reported estimator averages -- about +15% wall clock for +60% more
    # training instead of +60%.
    "v4": {
        "lr_schedule": "cosine",
        "lr_min_factor": 0.05,
        "augment": True,
        "aug_flip_p": 0.5,
        "aug_rot90_p": 0.5,
        "aug_intensity_p": 0.0,
        "aug_noise_std": 0.0,
        "tta": True,
        "postproc_min_voxels": 50,
        "train_per_hospital": 230,
        "val_per_hospital": 20,
        "select_by": "best_val",
        "report_last_k": 5,
        "rounds": 40,
        "eval_test_every": 3,
    },
    "v2": {
        # stop the plateau from thrashing: decay the per-round LR instead of taking
        # full-size Adam steps forever
        "lr_schedule": "cosine",
        "lr_min_factor": 0.05,
        # the baseline trained with no augmentation of any kind
        "augment": True,
        "aug_flip_p": 0.5,
        "aug_rot90_p": 0.5,
        "aug_intensity_p": 0.3,
        "aug_noise_std": 0.02,
        # inference-side gains that cost no training time
        "tta": True,
        "postproc_min_voxels": 50,
        # methodology: a real validation set, and an estimator chosen before the run
        "val_per_hospital": 20,
        "select_by": "best_val",
        "report_last_k": 5,
    },
}


def apply_preset(name: str, overrides: dict | None = None) -> Config:
    """Build a Config from a named preset, with explicit CLI overrides winning."""
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; pick from {sorted(PRESETS)}")
    return Config(**{**PRESETS[name], **(overrides or {})})
