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

## 2.1. The 2D / 3D comparison — what survives

> **This section replaces an earlier claim.** It previously reported that *all three* hypothesis
> verdicts reverse in 3D, and called that a novel contribution. Iterations v2–v5 refuted it: the
> reversal was an artefact of the v1 training recipe, not a property of the backbone. See
> [conventions.md](conventions.md) §6 and [iteration-report.md](iteration-report.md).

Under **v1**, 3D appeared to reverse every verdict — H1 supported, H2 and H3 not. Under **v5**,
with a properly tuned recipe run identically in both backbones, that picture is gone:

| | H1 | H2 | H3 |
|---|---|---|---|
| 2D v5 | ❌ | ✅ | ✅ |
| 3D v5 | ✅ | ✅ | inequality holds, **not significant** |

**H2 is backbone-independent.** A single global model underperforms the outlier site in both 2D and
3D once the recipe is sound. This is the durable claim of the study.

**H3 differs by backbone in strength, not in sign.** In 2D FedBN beats FedAvg on the outlier
decisively (+0.0604, 59 of 62 volumes, p = 4.5e-11). In 3D the inequality still holds (+0.0087) but
the margin is **not statistically significant** — uncorrected p = 0.066, Holm-corrected p = 0.53,
sign split 39/23. The defensible sentence is *"no significant difference in 3D"*, which is itself a
change from v1, where FedAvg beat FedBN in 3D significantly.

**The earlier explanation is also withdrawn.** FedBN's weaker 3D result was attributed to 150 local
cases being too few for stable 3D BatchNorm statistics. That is not supported: a capacity probe
(`base_channels` 48) was rejected on validation, and 3D local-only — flat under both v2 and v3 —
improved as soon as the round budget rose to 40. The binding constraint was **training length**,
not sample size or model size.

What the data does support about heterogeneity is broader than FedAvg: outlier degradation under
better optimization appears in **Centralized too**, which has no aggregation step at all. The cause
is optimizing over a majority-dominated distribution, and keeping BatchNorm local is what protects
the minority site — placing FedBN's mechanism in exactly the explanatory role its thesis claims.

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
3. ~~FL execution~~ — **resolved: custom loop now, FLARE later.** Build a lightweight custom PyTorch
   round-loop first (clients sequential on one GPU; FedBN = a one-line filter on the aggregation) to get
   the H1/H2/H3 results fast and transparently; then **port to NVIDIA FLARE** as a real-framework
   demonstration once the science is validated. See [`federated-learning.md`](federated-learning.md).
4. ~~Shift strength~~ — **provisionally calibrated** (γ/bias/blur per hospital, H4 outlier margin
   +0.149 σ after z-norm; a linear shift washes out to 0.000). The real calibration is whether **H2**
   appears once we train — revisit then. See [`data-pipeline.md`](data-pipeline.md) §2.
5. ~~Compute budget for local-only~~ — **resolved: `R × E` epochs**, matched to what a hospital spends
   across the whole federated run, so H1 tests collaboration rather than training length.
6. ~~Evaluation scope~~ — **resolved: diagonal headline + a 4×4 cross-hospital matrix for local-only**
   at the final round. See [`experiments.md`](experiments.md) §2.
7. ~~Local execution environment~~ — **resolved: WSL2 *and* native Windows both supported** for the
   custom loop; the FLARE port is Linux-only. See [`environments.md`](environments.md).
