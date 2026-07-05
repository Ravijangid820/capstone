# 06 · Running — environment, knobs, performance, scaling

## 6.1 Environment

- **OS:** **WSL2 (Linux)**, not native Windows. NVIDIA FLARE needs the POSIX
  `resource` module and UTF-8 source handling that Windows lacks (two separate
  crashes confirmed). The centralized baseline (pure PyTorch/MONAI) runs on either;
  only the FLARE parts require WSL.
- **Package manager:** **`uv`** (never pip). Install deps: `uv sync`. Run anything:
  `uv run python ...`.
- **Python:** 3.12 (pinned in `pyproject.toml`). **PyTorch:** CUDA 12.4 wheels.
- **Sanity check:**
  ```bash
  uv run python -c "import torch, nvflare; print(torch.cuda.is_available(), nvflare.__version__)"
  # expect: True 2.8.0
  ```
- **Filesystem:** keep the repo + `data/` on native **ext4** (`~/capstone`), not
  `/mnt/c` — the 9p bridge is slow for the many small slice reads.
- **`winshim/`** is Windows-only dead code (a `resource` shim from an abandoned
  native-Windows attempt). Never put it on `PYTHONPATH` in WSL — it would shadow the
  real module.

## 6.2 One command: `run_all.sh`

Runs all 8 steps (see [Architecture §3.3](03-architecture.md#33-the-8-step-pipeline-run_allsh)):

```bash
bash run_all.sh                 # full run: 6 hospitals, all cases, shift on, GPU
SMOKE=1 bash run_all.sh         # tiny CPU dry-run — checks everything wires up
```

Outputs land in `results/`: `per_hospital.csv`, `summary.csv`, `comparison.png`,
`personalization_gain.png`.

## 6.3 Every environment knob

`run_all.sh` is parameterized entirely by env vars, so you never edit the script:

| Var | Default | Meaning |
|-----|---------|---------|
| `DR` | `data/BraTS2021_Training_Data` | data root (BraTS case dirs) |
| `SPLIT` | `--n-clients 6` | hospitals: even split into N, **or** `--fets-csv <path>` for real FeTS |
| `ROUNDS` | `50` | FL rounds per method (steps 4–6) |
| `EPOCHS` | `40` | epochs for centralized (step 1) and local (step 3) |
| `GPU` | `0` | CUDA device for the FL steps; empty = CPU |
| `WORKERS` | `8` | **DataLoader workers** per step — the main speed lever (see §6.5) |
| `BATCH` | `8` | batch size (all steps) |
| `MAX_CASES` | *(unset)* | cap the number of cases for a **scoped** run (hours vs days); unset = all cases |
| `SHIFT` | `--synthetic-shift` | synthetic scanner heterogeneity; set `SHIFT=""` to disable. Auto-disabled when `SPLIT` uses FeTS. |
| `SMOKE` | `0` | `1` → tiny CPU dry-run: 2 clients, 1 round, 1 epoch, 6 cases, no shift-scale |

**Examples:**

```bash
# Scoped validation run (a few hours) — confirm H1-H3 before a big run
MAX_CASES=200 ROUNDS=20 EPOCHS=20 WORKERS=12 bash run_all.sh

# Full headline run
WORKERS=12 bash run_all.sh

# Real FeTS hospitals (shift auto-off)
SPLIT="--fets-csv data/fets_partitioning.csv" ROUNDS=80 bash run_all.sh

# Disable the synthetic shift on an even split (sanity check: FedBN should ≈ FedAvg)
SHIFT="" bash run_all.sh
```

Individual scripts also accept `--batch-size`, `--size`, `--workers`, `--max-cases`,
`--lr`, etc. for running/tuning one step at a time (see each script's `--help`).

## 6.4 Reading the run

The pipeline prints a `### N/8 <step>` header per step. Within training steps you'll
see per-epoch lines:

```
[epoch 14] loss=0.1537 | Dice mean=0.8103 (TC=0.802 WT=0.870 ET=0.759)
[epoch 14] saved new best (0.8103) -> data/centralized_unet.pt
```

A healthy centralized ceiling lands around **~0.79–0.81 mean Dice** (WT ~0.87). If it
craters to near-zero, something is wrong (that was the symptom of an early
under-trained run — 1 FL round + IID split — that produced meaningless numbers).

## 6.5 Performance model — the pipeline is CPU-bound

**The single most important operational fact:** at this model size (a small 2D U-Net,
batch 8, 192²), GPU compute per step is trivial — the GPU finishes a batch in
milliseconds and then **waits for data**. The bottleneck is the **CPU**: `nibabel`
slice reads plus the transforms (especially the synthetic-shift `gaussian_filter` +
percentile per slice per channel).

Measured evidence:

| Config | GPU util | CPU | epoch time |
|--------|----------|-----|------------|
| batch 8, **2 workers** | ~11 % | ~2 cores | ~50 min/epoch (full data) |
| batch 8, **8 workers** | ~40 % | ~8 cores (819 %) | — |
| batch 8, **12 workers** | ~33 % | ~12 cores (1241 %) | **~167 s/epoch** (200 cases) |
| **batch 32**, 8 workers | ~0–33 % | ~8 cores | ~312 s/epoch — **no faster**, worse convergence |

**Consequences:**
- **Workers (`WORKERS`), not batch size, are the speed lever.** More DataLoader
  workers = more parallel CPU transform throughput = the GPU starves less. On a
  16-core box, `WORKERS=12` roughly halved epoch time vs 8.
- **Bigger batches do *not* help** (the CPU produces slices at a fixed rate regardless
  of how the GPU gulps them) and **hurt convergence** at a fixed LR (fewer gradient
  steps/epoch). Batch 8 is the sweet spot here.
- **VRAM is not the constraint** — batch 8 uses ~221 MiB of 4096 MiB. The 4 GB limit
  matters for *3D* or big batches, not this config.

## 6.6 Scaling — scope before you commit

At **full scale** (all ~1251 cases, 6 hospitals, 40 epochs, 50 rounds × 3 FL methods),
a single epoch is ~50 min and the whole pipeline is **multi-day** on the RTX 3050 —
dominated by the federated steps. So:

- **Validate first** with a scoped run: `MAX_CASES=200 ROUNDS=20 EPOCHS=20 WORKERS=12`
  (~a few hours) to confirm the **H1–H3 signal** appears (FedBN > FedAvg on the shifted
  hospitals).
- **Then** scale up case count / rounds / epochs for the headline numbers.
- There is currently **no mid-run resume** — a run must complete in one sitting.
  (Between-step resume is possible by re-running only the remaining `uv run` steps,
  since each writes its own result JSONs.)

## 6.7 4 GB survival tips

- FL clients run **sequentially** (`--threads 1`, default) → one model on the GPU at a
  time.
- On CUDA OOM: drop `--batch-size 4` or `--size 160`.
- AMP is always on for CUDA; it roughly halves activation memory.

## 6.8 What a green run looks like

1. `### 1/8` centralized climbs to ~0.8 mean Dice over its epochs, saves
   `data/centralized_unet.pt`.
2. `### 2/8`–`7/8` each write `results/<method>/site-*.json` for all hospitals.
3. `### 8/8` writes `per_hospital.csv`, `summary.csv`, and the two PNGs, and prints
   the per-hospital table + the FedBN−FedAvg gain.

Then interpret with [Evaluation](07-evaluation.md).
