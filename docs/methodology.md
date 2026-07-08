# Methodology

## 1. Research question

Federated learning (FL) lets hospitals train a shared model without pooling patient data.
But real hospitals are **non-IID**: different scanners, field strengths, and acquisition
protocols shift the image distribution. A single global model (FedAvg) is pulled toward the
"average" hospital and can serve an unusual site poorly.

> **Question.** When heterogeneity comes from per-hospital scanner differences, does keeping
> the normalization statistics local (FedBN) recover the worst-served hospitals while keeping
> the average collaboration gain?

## 2. Hypotheses

| # | Statement | How we test it |
|---|---|---|
| **H1** | Collaboration beats going alone on average. | Mean Dice over hospitals: FedAvg ≥ local-only. |
| **H2** | The global model underperforms the outlier hospital. | On the most-shifted hospital, FedAvg < its own local-only model. |
| **H3** | Personalization recovers the outlier without hurting the mean. | FedBN ≥ FedAvg on mean Dice **and** FedBN closes the H2 gap. |

## 3. Data & heterogeneity design

- **Dataset:** BraTS 2021 (see [`data.md`](data.md)) — multi-modal 3D brain MRI with expert
  tumor masks. Details, labels, and prep there.
- **Hospitals:** we partition the cases into **K = 4** simulated hospitals (decided). Three are
  "typical" sites; **one is a designated outlier** with the strongest scanner shift, to drive H2/H3.
  Cases are assigned deterministically (fixed seed) — exact split in [`data-pipeline.md`](data-pipeline.md).
- **Non-IID source — synthetic scanner shift.** Each hospital applies a fixed, hospital-specific
  image transform to emulate its scanner: **gamma (contrast) shift + smooth bias field + slight
  blur**. These are *nonlinear / spatial* on purpose, so they **survive per-image z-normalization**
  (a purely linear intensity shift would be normalized away, leaving no real heterogeneity — a
  mistake we hit earlier and corrected). One hospital is deliberately made an **outlier** (strongest
  shift) to drive H2/H3.
- Later extension: swap the synthetic split for a *natural* multi-institutional split (e.g. FeTS)
  to confirm findings hold on real acquisition differences.

## 4. Methods compared

| Method | What is shared / kept local | Role |
|---|---|---|
| **Local-only** | Nothing shared; each hospital trains on its own data. | Lower baseline (no collaboration). |
| **FedAvg** | All weights averaged into one global model each round. | The "single shared model" baseline. |
| **FedBN** | All weights averaged **except BatchNorm** (running stats + affine kept per hospital). | The personalization method under test. |
| *(optional)* **FedProx** | FedAvg + proximal term to stabilize non-IID updates. | Stronger global baseline. |
| *(optional)* **Fine-tune / personal head** | Start from FedAvg, adapt locally. | Alternative personalization for comparison. |

Rationale for FedBN: under a scanner shift the main mismatch between hospitals is in the
*feature statistics* that BatchNorm captures. Keeping BN local lets each hospital normalize to
its own distribution while still sharing the convolutional filters learned across the federation.

## 5. Model

- **Architecture:** U-Net (encoder–decoder with skip connections), the standard for tumor
  segmentation. BatchNorm normalization layers (required for FedBN to have something to keep local).
- **2D and 3D — we evaluate both.** The pipeline is **dimension-parametric** (a `dim = 2d|3d`
  flag): the hospital partition, synthetic shift, FL loop, and evaluation are shared; only the
  **data sampler** (axial slices vs. 3D patches) and the **U-Net** (2D vs 3D conv) differ.
  - **2D first** — fast per run, so the full FL matrix fits Colab's session limits; this is the
    guaranteed deliverable and builds all the shared machinery.
  - **3D second, feasibility-gated** — a single-model spike measures whether a 3D U-Net trains on
    the T4 within Colab's limits (fit in 16 GB? per-epoch time? projected full-study wall-clock?).
    If it passes, we run the 3D FL study by flipping the flag; if not, the spike result itself is
    reported and 2D stands as the deliverable.
  - Doing both lets us ask a stronger question: *do the H1/H2/H3 findings hold under both backbones?*
- **Precision:** fp32. (fp16/AMP was found to corrupt BatchNorm running statistics on the
  strongly-shifted data — a real bug we fixed by disabling AMP.)

## 6. Evaluation

- **Metric:** Dice score on the three standard BraTS regions — **WT** (whole tumor = labels 1+2+4),
  **TC** (tumor core = 1+4), **ET** (enhancing tumor = 4).
- **Reported both ways:**
  - **per-hospital** Dice (this is where H2/H3 live — the outlier), and
  - **mean across hospitals** (this is where H1/H3-average live).
- **Protocol:** each hospital has a held-out local test split; every method is evaluated on every
  hospital's test set. Fixed seeds; identical data splits across methods for a fair comparison.

## 7. Experiment matrix

| Run | Cases | Hospitals | Method | Output |
|---|---|---|---|---|
| Centralized | pooled subset | 1 | standard training | sanity check + rough upper bound |
| Local ×K | per hospital | K | local-only | per-hospital lower baseline |
| FedAvg | all hospitals | K | FedAvg | global-model result (H1, H2) |
| FedBN | all hospitals | K | FedBN | personalized result (H3) |

Each FL run: *R* communication rounds × *E* local epochs; log per-round per-hospital Dice.

## 8. Open decisions (to resolve as we build)

1. ~~2D vs 3D model~~ — **resolved: build both**, dimension-parametric, 2D first, 3D feasibility-gated (§5).
2. ~~Number of hospitals~~ — **resolved: K = 4** (3 typical + 1 outlier); split in [`data-pipeline.md`](data-pipeline.md).
3. ~~FL execution~~ — **resolved: a lightweight custom PyTorch round-loop.** Clients run *sequentially*,
   sharing one GPU (no OOM risk on a 4 GB card), and FedBN is a one-line filter on the aggregation.
   NVIDIA FLARE is dropped as too heavy/risky for a small-GPU simulation. See [`federated-learning.md`](federated-learning.md).
4. **Shift strength** for the outlier hospital — still to calibrate (strong enough for H2, clinically plausible).
