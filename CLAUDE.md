# CLAUDE.md — Personalized Federated Learning for Brain Tumor Segmentation

Project context for Claude Code. **Read this first.** This is a **capstone project**.

## What this is

Personalized Federated Learning (FL) for **2D brain-tumor segmentation** on **BraTS/FeTS** MRI. Train a segmentation U-Net across simulated hospitals that never share raw data, and show that **personalized FL (primarily FedBN)** beats both a single global **FedAvg** model and **local-only** training — especially for hospitals whose scanners/protocols differ most.

**Research hypotheses (the result we're after):**
- **H1** — FedAvg ≥ local-only on average (collaboration helps).
- **H2** — FedAvg underperforms on *outlier* hospitals (heterogeneity hurts).
- **H3** — FedBN recovers/exceeds those hospitals **without** hurting the average.

Optimizing simultaneously for career value, publishability, and real-world impact.

## Environment (important)

- **Runs in WSL2 (Linux), NOT native Windows.** NVIDIA FLARE needs the POSIX `resource` module + UTF-8 source handling that Windows lacks (two separate crashes confirmed). All execution happens in WSL.
- **Package manager is `uv`, never pip.** Install: `uv sync`. Run anything: `uv run python ...`.
- **Hardware: RTX 3050 (4GB VRAM), 24GB RAM.** The 4GB budget drives the design: 2D slices (not 3D volumes), batch 8, AMP mixed precision, FL clients run **sequentially** (`--threads 1`). On CUDA OOM: drop `--batch-size 4` or `--size 160`.
- **`winshim/` is Windows-only dead code** — a `resource` shim from the abandoned native-Windows attempt. NEVER put it on `PYTHONPATH` in WSL (it shadows the real module). Safe to ignore/delete.
- Sanity check: `uv run python -c "import torch, nvflare; print(torch.cuda.is_available(), nvflare.__version__)"` → expect `True 2.8.0`.

## Stack

MONAI (U-Net + medical transforms) · NVIDIA FLARE 2.8 (federation, run via the **Simulator**) · PyTorch cu124.

## How to run

Full guide: **`WSL.md`**. One command runs all 8 steps: `bash run_all.sh`
(fast dry-run: `SMOKE=1 bash run_all.sh`; real FeTS hospitals: `SPLIT="--fets-csv data/fets_partitioning.csv" bash run_all.sh`).

Methods (flags on `fl/run_fedavg.py`): `fedavg`, `fedprox` (`--prox-mu`), `fedbn` (`--personalization fedbn`), `personal_head`. Baselines: `fl/run_local_baselines.py` (floor), `train_centralized` + `fl/evaluate.py` (ceiling), `fl/finetune.py`. Aggregate → `analyze.py` writes `results/` tables + plots.

## Code map

```
src/braintumor_fl/
  data.py            BraTS 2D lazy slice loading; case-level splits (single source of truth, no leakage)
  model.py           build_unet (MONAI UNet, BatchNorm) + BratsUNet wrapper (USE THIS) + Dice loss/metric
  trainer.py         shared train/eval: AMP, NaN-skip, grad-clip, FedProx term
  personalization.py keep-local param selection (FedBN=norm layers; personal_head=+output conv) + load_global
  partition.py       cases -> hospitals (even split OR FeTS CSV); partitioned_splits (no-leak centralized)
  results.py         uniform per-hospital result JSON schema
  train_centralized.py   centralized ceiling / Phase-1 pipeline check
fl/
  brats_client.py    FLARE client = one hospital
  run_fedavg.py      FLARE job runner (all FL methods)
  run_local_baselines.py / finetune.py / evaluate.py
analyze.py           results/ -> per_hospital.csv, summary.csv, personalization_gain.png
```

## Design invariants — DON'T break these

1. **Instantiate `BratsUNet`, not raw `build_unet`, in every script.** FLARE rebuilds the server model from a JSON config by introspecting constructor args; MONAI's UNet has a required `spatial_dims` that FLARE drops → server crash. `BratsUNet` has all-default args so it reconstructs cleanly. Its state-dict keys are all `net.*` — keep consistent everywhere or checkpoints won't cross-load.
2. **No data leakage.** `data.case_split` is the ONE source of truth for train/val (split by *patient*, never by slice). `partition.partitioned_splits` guarantees the centralized model never trains on any hospital's val cases. Every method evaluates on the *same* per-hospital val set.
3. **eval-before-train protocol.** In `brats_client.py`, the client evaluates the received model BEFORE local training each round → FedAvg reports the true global model, FedBN the genuine personalized one. Do not move eval after training.
4. **Personalization is client-side.** The server always runs plain FedAvg; personalization = which params the client KEEPS LOCAL when the global model arrives (`personalization.keep_local_keys`). FedBN keeps BatchNorm layers; `personal_head` also keeps the output conv.
5. **FedBN local params persist across rounds.** `brats_client.py` saves `{site}_local.pt` after training and restores it before eval, so BatchNorm accumulates even if FLARE re-runs the client each round. Removing this silently collapses FedBN → FedAvg.
6. **Training stability:** cosine LR decay + grad-clip (1.0) + NaN-skip; LR default `5e-4`. (Earlier NaN blow-ups occurred at lr=1e-3.)

## Current status

- **Phase 1 (centralized) validated** — ~0.79 mean Dice (WT 0.88) on 100 cases.
- **Full FL pipeline runs end-to-end in WSL** — FedAvg / FedProx / FedBN + all baselines + analyze all execute.
- ⚠️ Smoke-mode numbers are meaningless (1 round, IID even split). Real results need scale + heterogeneity.

## Next steps (priority order)

1. **Get non-IID hospitals — the contribution depends on heterogeneity.** On an even/IID split, FedBN ≈ FedAvg by design.
   - *Best:* real FeTS partitioning CSV → `data/fets_partitioning.csv` (Synapse/FeTS registration; see `DATA.md`).
   - *Fallback:* synthetic per-hospital scanner-shift intensity transforms — lets FedBN show value without FeTS access.
2. **Scale a real run:** `rounds≈40`, most/all data, `--gpu 0`.
3. **Produce the headline:** per-hospital table + `personalization_gain.png` demonstrating H1–H3.

## Conventions / preferences

- **Git: do NOT add "Co-Authored-By: Claude" or any Anthropic/Claude attribution to commits or PR bodies.** Commits are authored solely by the user (Ravi Jangid). (This overrides the default.)
- Use **`uv`**, never pip.
- BraTS data (~15GB) and model checkpoints (`*.pt`) are gitignored — never commit them.
- Repo: `github.com/Ravijangid820/capstone`, `master` branch, solo project → push to `master` directly (no PR flow).
- For best training I/O in WSL, keep the repo + `data/` on the **native ext4** filesystem, not `/mnt/c` (9p is slow for the many small slice reads).

## Data

- **BraTS 2021**: `data/BraTS2021_Training_Data/` — 1251 case folders, each with `*_flair/_t1/_t1ce/_t2/_seg.nii.gz`. Gitignored.
- **FeTS partitioning CSV**: needed for the real non-IID experiment; not yet obtained.
