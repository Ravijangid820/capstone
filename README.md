# Brain Tumor Segmentation with Personalized Federated Learning

Capstone: 2D brain tumor segmentation on **BraTS/FeTS**, federated across hospitals
with **NVIDIA FLARE + MONAI**, using **personalized FL** to handle scanner/protocol
heterogeneity across institutions.

- **Plan & research question:** [PLAN.md](PLAN.md)
- **Data access (do this first):** [DATA.md](DATA.md)

## Setup

```powershell
uv sync                                        # installs MONAI, NVIDIA FLARE, CUDA torch, ...
uv run python -c "import torch; print('cuda:', torch.cuda.is_available())"   # expect: cuda: True
```

## Phase 1 — centralized baseline (start here)

Get the data (see [DATA.md](DATA.md)), then:

```powershell
uv run python -m braintumor_fl.train_centralized `
    --data-root data/BraTS2021_TrainingData `
    --max-cases 100 --epochs 20 --batch-size 8
```

On the 4GB 3050, start with `--max-cases 50-100` and `--batch-size 8` to iterate fast;
scale up on Colab's T4. If you hit CUDA OOM, drop `--batch-size` to 4 or `--size` to 160.

**Phase 1 exit criterion:** validation Dice climbs and lands in a sane range
(rough ballpark: WT ~0.80+, TC/ET lower). Then we federate (Phase 2).

## Layout

```
capstone/
├─ PLAN.md / DATA.md          ← plan + data access
├─ pyproject.toml             ← uv deps (MONAI, nvflare, CUDA torch)
├─ src/braintumor_fl/
│  ├─ data.py                 ← BraTS 2D slice pipeline (lazy, no pre-slicing)
│  ├─ model.py                ← MONAI U-Net + Dice loss/metric
│  └─ train_centralized.py    ← Phase 1 baseline
└─ data/                      ← BraTS/FeTS (gitignored)
```

## Roadmap
Phase 1 centralized ✅ code ready → Phase 2 FedAvg in FLARE → Phase 3 personalization
(FedBN / fine-tune / shared-encoder) → Phase 4 writeup. See [PLAN.md](PLAN.md).
