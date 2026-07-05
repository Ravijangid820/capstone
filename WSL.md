# Running the project in WSL2

NVIDIA FLARE is Linux-first and does **not** run on native Windows (needs the
POSIX `resource` module + UTF-8 source handling). So we **develop/edit on Windows**
but **execute in WSL2**. The centralized baseline is pure PyTorch/MONAI and runs
on either; only the FLARE (federated) parts require WSL2.

## 0. One-time WSL2 setup

```bash
# In an Ubuntu (WSL2) terminal:
nvidia-smi          # should list your RTX 3050 — proves GPU passthrough works
                    # (comes from the Windows NVIDIA driver; nothing to install in WSL)

# install uv inside WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.bashrc
```

## 1. Get the project + a Linux venv

The project lives on the Windows disk, reachable from WSL at `/mnt/c/...`.

```bash
cd /mnt/c/Users/ravi_jangir/Desktop/capstone

# The Windows .venv can't be reused on Linux — rebuild it:
rm -rf .venv
uv sync                       # installs Linux CUDA wheels for torch/monai/nvflare
uv run python -c "import torch, nvflare; print('cuda', torch.cuda.is_available()); print('nvflare', nvflare.__version__)"
# expect: cuda True  /  nvflare 2.8.0    (and NO 'resource' error — Linux has it)
```

> `/mnt/c` I/O is slower than WSL-native. If training feels I/O-bound, copy the repo
> into WSL home (`cp -r /mnt/c/.../capstone ~/capstone`) and work there. The BraTS
> `data/` folder can stay on `/mnt/c` and be symlinked, or copied for speed.
>
> **Do NOT set PYTHONPATH to `winshim/`** in WSL — that folder is a Windows-only shim
> for the missing `resource` module; on Linux the real module must win.

## 2. Validate FLARE works (tiny CPU smoke — do this first)

```bash
uv run python fl/run_fedavg.py --data-root data/BraTS2021_Training_Data \
    --n-clients 3 --rounds 1 --epochs 1 --max-cases 6 --method smoke
```
Success = the simulator starts 3 clients, runs 1 round, and exits cleanly. This is
the run that failed on Windows — it should now complete. (First run may still need a
small tweak; treat it like the Phase-1 baseline shakeout.)

## 3. The full experiment pipeline

**One command runs all 8 steps:** `bash run_all.sh`
(dry-run first with `SMOKE=1 bash run_all.sh`; switch to real hospitals with
`SPLIT="--fets-csv data/fets_partitioning.csv" bash run_all.sh`).

The individual steps are below, for running/tuning them one at a time. Start with an
**even split of 6 hospitals** (works today); swap `--n-clients 6` for
`--fets-csv data/fets_partitioning.csv` once you have the real FeTS partitions.
Use `--gpu 0` for real runs.

```bash
DR=data/BraTS2021_Training_Data
SPLIT="--n-clients 6"          # later: SPLIT="--fets-csv data/fets_partitioning.csv"

# 1) CEILING — centralized (trains on union of hospital train-cases; no val leakage)
uv run python -m braintumor_fl.train_centralized --data-root $DR $SPLIT \
    --epochs 40 --out data/centralized_unet.pt
uv run python fl/evaluate.py --data-root $DR $SPLIT \
    --model data/centralized_unet.pt --method centralized

# 2) FLOOR — local-only (each hospital alone)
uv run python fl/run_local_baselines.py --data-root $DR $SPLIT --epochs 40

# 3) FedAvg baseline
uv run python fl/run_fedavg.py --data-root $DR $SPLIT --rounds 50 --epochs 1 \
    --method fedavg --gpu 0

# 4) FedProx baseline
uv run python fl/run_fedavg.py --data-root $DR $SPLIT --rounds 50 --epochs 1 \
    --method fedprox --prox-mu 0.01 --gpu 0

# 5) FedBN — our PRIMARY personalization method
uv run python fl/run_fedavg.py --data-root $DR $SPLIT --rounds 50 --epochs 1 \
    --method fedbn --personalization fedbn --gpu 0

# 6) personal-head personalization (optional)
uv run python fl/run_fedavg.py --data-root $DR $SPLIT --rounds 50 --epochs 1 \
    --method personal_head --personalization personal_head --gpu 0

# 7) fine-tune personalization (starts from the FedAvg global)
uv run python fl/finetune.py --data-root $DR $SPLIT --ft-epochs 5

# 8) AGGREGATE — the headline comparison + plots
uv run python analyze.py
```

Outputs land in `results/`:
- `per_hospital.csv`, `summary.csv` — the numbers
- `comparison.png` — per-hospital Dice by method
- `personalization_gain.png` — FedBN − FedAvg per hospital (**the core result**)

## 4. What we're looking for (H1–H3)

- **H1**: FedAvg ≥ Local-only on average (collaboration helps).
- **H2**: FedAvg underperforms on outlier hospitals (heterogeneity hurts).
- **H3**: FedBN recovers/exceeds those hospitals **without** hurting the average →
  positive bars in `personalization_gain.png`, especially for the worst hospitals.

## 5. 4GB GPU notes
- FL clients run **sequentially** (`--threads 1`, the default) so only one model is
  on the GPU at a time — fits 4GB.
- If CUDA OOM: add `--batch-size 4` (all scripts accept it) or `--size 160`.
- Big/long runs: raise `--rounds`/`--epochs` on Colab's T4 or the Azure credit; the
  code is identical.
