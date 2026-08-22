"""Export FedAvg vs FedBN prediction PNGs for one test case -- the qualitative figure.

    python scripts/export_predictions.py --dim 2d --seed 42 --runs artifacts/runs/v2
    python scripts/export_predictions.py --dim 3d --seed 42 --case BraTS2021_00017 --slice 78
    python scripts/export_predictions.py --dim 2d --seed 42 --runs artifacts/runs/v2 --pick best

Same models, same preprocessing, and the same cached tensors the scored runs used, so the picture
and the Dice in Table V describe the same event. The demo server renders this interactively;
this writes it to disk headlessly and reproducibly, which is what a paper figure needs.

**Which case gets shown matters.** The default is `--pick representative`: among the hospital's
test cases, the one whose FedBN − FedAvg gap sits closest to that hospital's *median* gap. A
figure is an illustration of the table, and picking the case with the biggest gap would illustrate
the tail rather than the effect -- the reader sees the best frame of a method and reads it as a
typical one. `--pick best` and `--pick worst` are available and are labelled as such in the
caption metadata, so a deliberately-chosen extreme is never silently passed off as typical.

The slice defaults to the one with the largest ground-truth whole-tumour area, which is both
reproducible and the slice a reader would choose by hand.

Outputs, into `artifacts/figures/<dim>_<seed>_<hospital>_<case>/`:

    mri.png · ground_truth.png · fedavg.png · fedbn.png    panels at native resolution
    panel.png                                              the four side by side, labelled
    caption.json                                           case, slice, per-case Dice, provenance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fedbrats.config import Config                     # noqa: E402
from fedbrats.data import load_cached_case, load_index, select_cases  # noqa: E402
from fedbrats.metrics import dice_regions              # noqa: E402
from fedbrats.model import build_model                 # noqa: E402
from fedbrats.train import predict_volume              # noqa: E402
from compute_significance import per_case_rows        # noqa: E402
from rescore import load_states                        # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# The demo server's region colours, kept identical so the figure in the paper and the view in the
# live demo cannot disagree about which colour means which region.
REGION_COLORS = {"wt": (16, 185, 129), "tc": (59, 130, 246), "et": (236, 72, 153)}
REGION_LABELS = {"wt": "Whole tumour", "tc": "Tumour core", "et": "Enhancing tumour"}
MODALITIES = ["flair", "t1", "t1ce", "t2"]
DISPLAY = {"fedavg": "FedAvg", "fedbn": "FedBN",
           "local": "Local-only", "centralized": "Centralized"}


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------

def to_uint8(sl: np.ndarray) -> np.ndarray:
    """One MRI slice to 0-255 greyscale, windowed to its own range."""
    lo, hi = float(sl.min()), float(sl.max())
    if hi <= lo:
        return np.zeros(sl.shape, dtype=np.uint8)
    return (255 * (sl - lo) / (hi - lo)).astype(np.uint8)


def overlay(mri: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend the three region masks over a greyscale slice, painted WT -> TC -> ET.

    The order is deliberate and matches the demo: the regions are nested (ET inside TC inside WT),
    so painting the largest first leaves each smaller region visible on top instead of buried.
    """
    rgb = np.stack([mri] * 3, axis=-1).astype(np.float32)
    for i, region in enumerate(("wt", "tc", "et")):
        colour = np.asarray(REGION_COLORS[region], dtype=np.float32)
        sel = mask[i][..., None].astype(bool)
        rgb = np.where(sel, rgb * (1 - alpha) + colour * alpha, rgb)
    return rgb.astype(np.uint8)


def orient(img: np.ndarray) -> np.ndarray:
    """Radiological convention: rows run superior->inferior, so flip the row axis for display."""
    return np.flipud(np.swapaxes(img, 0, 1))


def save_png(arr: np.ndarray, path: Path) -> None:
    from PIL import Image
    Image.fromarray(orient(arr)).save(path)


def save_panel(panels: list[tuple[str, str, np.ndarray]], path: Path, title: str) -> None:
    """The four views side by side with titles, per-panel Dice, and a region legend.

    Matplotlib only for the composition -- axes, ticks and grids are all off. The legend is
    non-negotiable rather than decorative: the region colours sit below 3:1 contrast against the
    surface, so identity has to be carried by a label as well as by the colour.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(panels)
    # constrained_layout, not tight_layout + bbox_inches="tight": the two disagree about how much
    # room the per-panel Dice line needs and the last panel's label was being clipped mid-number.
    fig, axes = plt.subplots(1, n, figsize=(2.9 * n, 3.5), dpi=200, constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, (heading, sub, img) in zip(axes, panels):
        ax.imshow(orient(img), interpolation="nearest")
        ax.set_title(heading, fontsize=11, pad=6)
        if sub:
            ax.set_xlabel(sub, fontsize=8, labelpad=5, linespacing=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles = [Patch(facecolor=np.array(REGION_COLORS[r]) / 255, label=REGION_LABELS[r])
               for r in ("wt", "tc", "et")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=10.5)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------------------

def gap_by_case(per_case_root: Path, dim: str, seed: int, hospital: str,
                a: str, b: str, region: str = "wt") -> dict[str, float]:
    """case_id -> (B − A) Dice for one hospital, from per-case rows already on disk.

    Read rather than recomputed: choosing the case from the same numbers that Table V reports is
    what makes "representative" mean representative *of the table* and not of a second, slightly
    different scoring pass. Which is also why the stage filter is shared with that script rather
    than reimplemented here: a run that logs per-case Dice every round (v5 does) holds thousands
    of intermediate rows alongside the 248 final ones, and selecting on split alone would pick the
    case using whichever round happened to be written last.
    """
    def scores(method: str) -> dict[str, float]:
        rows, _ = per_case_rows(per_case_root / f"{method}_{dim}_{seed}")
        return {r["case_id"]: r[f"dice_{region}"] for r in rows
                if r.get("test_hospital") == hospital}

    sa, sb = scores(a), scores(b)
    return {c: sb[c] - sa[c] for c in sorted(set(sa) & set(sb))}


def pick_case(gaps: dict[str, float], how: str) -> str:
    if how == "best":
        return max(gaps, key=lambda c: gaps[c])
    if how == "worst":
        return min(gaps, key=lambda c: gaps[c])
    median = float(np.median(list(gaps.values())))
    return min(gaps, key=lambda c: abs(gaps[c] - median))


def best_slice(y: np.ndarray) -> int:
    """Axial index with the most ground-truth whole-tumour voxels."""
    return int(np.asarray(y[0]).sum(axis=(0, 1)).argmax())


# --------------------------------------------------------------------------------------

def run_config(run_dir: Path, dim: str, seed: int) -> Config:
    """Config for inference, taking tta/postproc/overlap from the run's own config.json.

    Reading them rather than defaulting them keeps the picture on the same inference settings as
    the scored numbers; a figure rendered without the TTA its table row used is a different model.
    """
    cfg_path = run_dir / "config.json"
    saved = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    def get(key, default):
        v = saved.get(key, default)
        if isinstance(v, str) and not isinstance(default, str):   # old stringified configs
            v = json.loads(v.lower()) if isinstance(default, bool) else type(default)(v)
        return v

    return Config(dim=dim, seed=seed,
                  tta=get("tta", False),
                  postproc_min_voxels=get("postproc_min_voxels", 0),
                  sw_overlap=get("sw_overlap", 0.25))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    base = Config()
    ap.add_argument("--dim", default="2d", choices=("2d", "3d"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--runs", type=str, default=str(base.paths.runs),
                    help="root holding <method>_<dim>_<seed>/checkpoints/final.pt")
    ap.add_argument("--per-case", type=str, default=None,
                    help="root holding per_case.jsonl for --pick (default: --runs, then baseline)")
    ap.add_argument("--hospital", default=base.outlier_hospital)
    ap.add_argument("--methods", nargs="+", default=["fedavg", "fedbn"])
    ap.add_argument("--case", default=None, help="case id; default: chosen by --pick")
    ap.add_argument("--pick", default="representative",
                    choices=("representative", "best", "worst"))
    ap.add_argument("--slice", type=int, default=None, help="default: largest GT tumour area")
    ap.add_argument("--modality", default="flair", choices=MODALITIES)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    runs = Path(args.runs)
    a, b = args.methods[0], args.methods[-1]

    # Pick the case
    case_id, pick_note = args.case, "specified by hand"
    gaps: dict[str, float] = {}
    if case_id is None:
        pc_root = Path(args.per_case) if args.per_case else runs
        if not (pc_root / f"{a}_{args.dim}_{args.seed}" / "per_case.jsonl").exists():
            pc_root = base.paths.artifacts / "baseline"
        gaps = gap_by_case(pc_root, args.dim, args.seed, args.hospital, a, b)
        if not gaps:
            print(f"no per-case rows for {a}/{b} {args.dim} seed {args.seed} under {pc_root}; "
                  f"pass --case <id> to choose one directly", file=sys.stderr)
            return 1
        case_id = pick_case(gaps, args.pick)
        median = float(np.median(list(gaps.values())))
        pick_note = (f"--pick {args.pick} over {len(gaps)} {args.hospital} test cases "
                     f"(this case {b}−{a} WT {gaps[case_id]:+.4f}; "
                     f"hospital median {median:+.4f})")

    cfg = run_config(runs / f"{a}_{args.dim}_{args.seed}", args.dim, args.seed)
    index = load_index(cfg)
    if case_id not in select_cases(index, args.hospital, "test"):
        print(f"{case_id} is not a {args.hospital} test case", file=sys.stderr)
        return 1

    x, y = load_cached_case(cfg, case_id)
    x, y = np.asarray(x), np.asarray(y)
    z = args.slice if args.slice is not None else best_slice(y)
    if not 0 <= z < y.shape[-1]:
        print(f"slice {z} out of range 0..{y.shape[-1] - 1}", file=sys.stderr)
        return 1

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    mri = to_uint8(x[MODALITIES.index(args.modality), :, :, z])

    # The recipe goes in the directory name. Two recipes can pick the same case, and a figure
    # directory that says only "2d_42_H4_<case>" would let a v2 panel be captioned as a v5 result
    # with nothing on disk to contradict it.
    recipe = runs.name if runs.resolve() != base.paths.runs.resolve() else "baseline"
    out_dir = Path(args.out) if args.out else (
        base.paths.artifacts / "figures" /
        f"{recipe}_{args.dim}_{args.seed}_{args.hospital}_{case_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = [("MRI (" + args.modality + ")", "", np.stack([mri] * 3, axis=-1)),
              ("Ground truth", "", overlay(mri, y[:, :, :, z]))]
    save_png(np.stack([mri] * 3, axis=-1), out_dir / "mri.png")
    save_png(overlay(mri, y[:, :, :, z]), out_dir / "ground_truth.png")

    dice: dict[str, dict] = {}
    for method in (a, b):
        run_dir = runs / f"{method}_{args.dim}_{args.seed}"
        ckpt = run_dir / "checkpoints" / "final.pt"
        if not ckpt.exists():
            print(f"no checkpoint at {ckpt}", file=sys.stderr)
            return 1
        mcfg = run_config(run_dir, args.dim, args.seed)
        model = build_model(mcfg)
        model.load_state_dict(load_states(ckpt, method, [args.hospital])[args.hospital])
        pred = predict_volume(model, x, mcfg, device, tta=mcfg.tta)

        dice[method] = {"volume": dice_regions(pred, y),
                        "slice": dice_regions(pred[:, :, :, z], y[:, :, :, z]),
                        "tta": mcfg.tta, "postproc_min_voxels": mcfg.postproc_min_voxels}
        save_png(overlay(mri, pred[:, :, :, z]), out_dir / f"{method}.png")
        v = dice[method]["volume"]
        panels.append((DISPLAY.get(method, method),
                       f"volume Dice\nWT {v['wt']:.3f} · TC {v['tc']:.3f} · ET {v['et']:.3f}",
                       overlay(mri, pred[:, :, :, z])))
        print(f"{method:9} volume Dice  WT={v['wt']:.4f} TC={v['tc']:.4f} ET={v['et']:.4f}")

    save_panel(panels, out_dir / "panel.png",
               f"{args.hospital} (shifted) · {case_id} · axial slice {z} · {args.dim.upper()}")

    caption = {"case_id": case_id, "hospital": args.hospital, "slice": z, "dim": args.dim,
               "seed": args.seed, "modality": args.modality, "runs_root": str(runs),
               "selection": pick_note, "methods": [a, b], "dice": dice,
               "note": "Dice under each panel is over the whole volume, not the shown slice; "
                       "the per-slice value is in dice.<method>.slice."}
    (out_dir / "caption.json").write_text(json.dumps(caption, indent=2, sort_keys=True))
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
