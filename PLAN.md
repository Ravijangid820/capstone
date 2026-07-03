# Personalized Federated Learning for Brain Tumor Segmentation — Project Plan

*Scope locked 2026-07-03.*

## 1. Scope (what we're building)

| Dimension | Decision |
|---|---|
| **Task** | 2D-slice brain tumor **segmentation** (outline tumor sub-regions) |
| **Data** | BraTS 2021 volumes + **FeTS 2022 real institutional partitioning** |
| **FL angle** | **Personalized Federated Learning (pFL)** — the core contribution |
| **Framework** | **NVIDIA FLARE** (federation) + **MONAI** (imaging model/pipeline) |
| **Hardware** | RTX 3050 4GB (dev) · Colab T4 (heavy runs) · Azure $100 (final runs) |

## 2. The core research question

> On brain tumor segmentation across hospitals with different MRI scanners/protocols,
> does **personalized FL** beat both (a) a single global **FedAvg** model and
> (b) each hospital training **alone** — *especially* for the hospitals whose data
> differs most from the majority?

### Hypotheses
- **H1** — FedAvg (collaboration) beats isolated per-hospital training on average.
- **H2** — FedAvg *underperforms* on "outlier" hospitals (heterogeneity hurts them).
- **H3** — Personalized FL recovers/exceeds performance on those outlier hospitals **without** hurting the average.

If H2 + H3 hold on **real** FeTS institutional splits, that's a publishable result.

## 3. Metrics

- **Dice score** per tumor region — **WT** (whole tumor), **TC** (tumor core), **ET** (enhancing tumor) — the BraTS-standard regions.
- Reported **per hospital** and averaged.
- **Fairness:** worst-case and variance of Dice across hospitals.
- **Personalization gain:** per-hospital `Dice(pFL) − Dice(FedAvg)`.

## 4. Methods to compare (the experiment matrix)

| Method | Type | Notes |
|---|---|---|
| Local-only | baseline | each hospital trains alone (no sharing) |
| FedAvg | baseline | one global model |
| FedProx | baseline | FedAvg + proximal term for non-IID |
| **FedBN** | **personalized** | keep BatchNorm layers **local** — natural fit: scanner/intensity differences live in BN stats. Cheap, strong. |
| **Fine-tune** | **personalized** | train FedAvg global, then fine-tune per hospital |
| **Shared encoder + personal head** | **personalized** | share U-Net encoder, keep each hospital's decoder (FedRep-style) |

FedBN is our **primary** personalization method (best-motivated for MRI scanner heterogeneity); the other two are comparisons.

## 5. Phased timeline

- **Phase 0 — Setup & data access** *(now)*: env via `uv sync`; register + download BraTS/FeTS. See [DATA.md](DATA.md).
- **Phase 1 — Centralized baseline**: 2D U-Net (MONAI) training on one GPU. Proves the data + model + metrics pipeline end-to-end. Code: `src/braintumor_fl/train_centralized.py`.
- **Phase 2 — Federate (FedAvg)**: wrap the model in an NVIDIA FLARE job; run FedAvg across FeTS institutions in the FLARE **Simulator**. Establish the centralized-vs-federated gap and per-hospital Dice.
- **Phase 3 — Personalize (the contribution)**: implement FedBN + fine-tuning + shared-encoder; run the full experiment matrix; produce the per-hospital comparison + fairness analysis.
- **Phase 4 — Stretch / writeup**: optional differential privacy or robustness study; ablations; paper draft.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| 4GB VRAM too small | **2D slices** (not 3D volumes), batch 8–16, AMP mixed precision, 192×192 crops |
| NVIDIA FLARE learning curve | Use the **FLARE Simulator** + official BraTS/FeTS examples; budget extra days in Phase 2 |
| FLARE quirks on Windows | Fall back to **WSL2** (Ubuntu) — smoother for FLARE |
| Data access delay | Start Phase 1 on a **Kaggle BraTS mirror** while the official FeTS partition arrives |
| Disk blow-up from pre-slicing | Load 2D slices **on the fly** from 3D volumes via lazy `nibabel` (no pre-slicing to disk) |

## 7. What "done" looks like

A table + figure showing: for each FeTS hospital, the Dice of Local-only vs FedAvg vs our personalized method, with the outlier hospitals highlighted — demonstrating H1–H3. Plus a short paper/report and clean, reproducible FLARE job configs.
