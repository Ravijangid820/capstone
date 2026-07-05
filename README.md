# Brain Tumor Segmentation with Personalized Federated Learning

Capstone: 2D brain-tumor segmentation on **BraTS/FeTS**, federated across hospitals
with **NVIDIA FLARE + MONAI**, using **personalized FL** (primarily **FedBN**) to
handle the scanner/protocol heterogeneity between institutions.

- **Plan & research question:** [PLAN.md](PLAN.md)
- **Data access:** [DATA.md](DATA.md)
- **How to run everything (in WSL2):** [WSL.md](WSL.md)

## The idea in one line

Prove that a personalized federated model beats **both** a single global FedAvg
model **and** each hospital training alone — especially for hospitals whose data
differs most from the majority (hypotheses H1–H3 in [PLAN.md](PLAN.md)).

## ⚠️ Where things run

NVIDIA FLARE is Linux-first and does **not** run on native Windows. We **edit on
Windows, execute in WSL2**. The centralized baseline (pure PyTorch/MONAI) runs on
either. Full execution steps: **[WSL.md](WSL.md)**.

## Method matrix

| Method | Type | How it's implemented |
|---|---|---|
| Centralized | ceiling | train on all hospitals' data pooled (no leakage into val) |
| Local-only | floor | each hospital trains alone |
| FedAvg | baseline | standard global averaging |
| FedProx | baseline | FedAvg + proximal term (`--prox-mu`) |
| **FedBN** | **personalized (primary)** | keep BatchNorm layers local per hospital |
| personal-head | personalized | share body, keep the output head local |
| fine-tune | personalized | FedAvg global, then a few local epochs per hospital |

All personalization is enforced client-side by *which params are kept local* when
the global model arrives (`src/braintumor_fl/personalization.py`). The server always
runs plain FedAvg. Each hospital is scored **before** local training, so FedAvg
reports the true global model and FedBN reports the genuine personalized model.

## Code map

```
src/braintumor_fl/
├─ data.py            BraTS 2D slice pipeline (lazy load) + case-level splits (no leakage)
├─ model.py           MONAI 2D U-Net (BatchNorm) + Dice loss/metric
├─ trainer.py         shared train/eval (AMP, NaN-skip, grad-clip, FedProx term)
├─ personalization.py keep-local param selection (FedBN / personal-head) + load_global
├─ partition.py       cases -> hospitals (even split OR real FeTS CSV) + no-leak splits
├─ results.py         uniform per-hospital result JSON schema
└─ train_centralized.py   centralized ceiling / Phase-1 check
fl/
├─ brats_client.py    FLARE client = one hospital
├─ run_fedavg.py      FLARE job runner (all FL methods via flags)
├─ run_local_baselines.py   local-only floor
├─ finetune.py        post-FedAvg per-hospital fine-tuning
└─ evaluate.py        score one model per hospital (centralized ceiling)
analyze.py            aggregate all results -> tables + plots (the headline figure)
winshim/              Windows-only `resource` shim (ignore in WSL)
```

## Status

- ✅ **Phase 1** — centralized pipeline validated on Windows (100 cases, 20 epochs,
  val Dice ~0.79 WT 0.88). Stability fixes (grad-clip + NaN-skip) in place.
- ✅ **Phase 2–3 code** — full FL + personalization + evaluation + analysis written
  and unit-tested (pure logic). Pending execution in WSL2.
- ⏭️ **Next** — run the pipeline in WSL2 ([WSL.md](WSL.md)), then swap in the real
  FeTS partition and scale rounds on Colab/Azure.

## Quick Phase-1 check (native Windows, works today)

```powershell
uv run python -m braintumor_fl.train_centralized --data-root data/BraTS2021_Training_Data --max-cases 100 --epochs 20
```
