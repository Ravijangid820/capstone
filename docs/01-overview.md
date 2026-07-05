# 01 · Overview — the idea, the goal, the hypotheses

## 1.1 The problem in one paragraph

Hospitals hold the MRI scans that would make a great brain-tumor segmentation
model, but they **cannot pool that data** — patient privacy, regulation (HIPAA/
GDPR), and institutional policy forbid moving raw scans off-site. **Federated
Learning (FL)** sidesteps this: each hospital trains locally and shares only model
weight updates, which a server averages into a global model. But there's a catch
that is the whole point of this project: hospitals use **different MRI scanners and
acquisition protocols**, so their images have systematically different intensity
distributions. A single averaged model (**FedAvg**) fits the majority and quietly
**underperforms on the hospitals that differ most**. **Personalized FL** aims to
fix exactly that.

## 1.2 What we are building

A reproducible pipeline that, on **2D brain-tumor segmentation** over **BraTS/FeTS**
MRI:

1. Trains a segmentation **U-Net** across several simulated hospitals that never
   share raw data (federation via the **NVIDIA FLARE** Simulator).
2. Compares a full **method matrix** — from a local-only floor to a centralized
   ceiling, through FedAvg/FedProx baselines, up to personalized methods.
3. Shows that **personalized FL (primarily FedBN)** beats *both* the single global
   FedAvg model *and* local-only training — **especially** for outlier hospitals —
   and quantifies it per hospital.

The headline deliverable is a **per-hospital table + figure** demonstrating the
three hypotheses below.

## 1.3 The core research question

> On brain-tumor segmentation across hospitals with different MRI scanners/protocols,
> does **personalized FL** beat both (a) a single global **FedAvg** model and
> (b) each hospital training **alone** — *especially* for the hospitals whose data
> differs most from the majority?

## 1.4 The hypotheses (the result we're after)

| ID | Hypothesis | Why it matters |
|----|------------|----------------|
| **H1** | **FedAvg ≥ local-only on average** (collaboration helps). | Establishes that federation is worth doing at all. |
| **H2** | **FedAvg underperforms on *outlier* hospitals** (heterogeneity hurts). | This is the *failure* personalized FL exists to fix; without it there's no story. |
| **H3** | **FedBN recovers/exceeds those hospitals *without* hurting the average.** | The contribution: personalization helps the hurt hospitals for free. |

If **H2 + H3** hold on **real** FeTS institutional splits, that is a publishable
result. On an IID/even split they *cannot* hold — FedBN ≈ FedAvg by construction —
which is why manufacturing heterogeneity is a prerequisite (see §1.6 and
[Data pipeline §Synthetic shift](04-data-pipeline.md#the-synthetic-scanner-shift)).

## 1.5 Why FedBN specifically

The intuition is clean and physically grounded. A scanner/protocol difference is,
to first order, an **intensity-distribution shift**. In a network with
**BatchNorm**, that shift is captured largely by the BatchNorm layers' running
statistics (mean/variance) and affine parameters. **FedBN** keeps precisely those
BatchNorm layers **local** to each hospital and federates everything else. So each
hospital gets the shared, data-rich "anatomy" features of the global model **plus**
its own scanner calibration — cheap (BN is a tiny fraction of parameters), strong,
and exactly matched to the failure mode. FedBN is our **primary** personalization
method; fine-tuning and a personal-head variant are comparisons.

## 1.6 Why we manufacture heterogeneity (the honest caveat)

The contribution depends on hospitals being **non-IID**. Two ways to get that:

- **Real FeTS institutional partitioning** (best): a CSV mapping each BraTS subject
  to the actual hospital that produced it. Not yet obtained (Synapse/FeTS
  registration). See [Data pipeline](04-data-pipeline.md) and `DATA.md`.
- **Synthetic per-hospital scanner shift** (the current path): a deterministic,
  physically-motivated intensity transform applied per hospital — a distinct
  "scanner profile" per site. This lets FedBN demonstrate value **today**, without
  waiting on data access, and swaps out cleanly for the real FeTS CSV later.

On an **even / IID split with no shift**, FedBN ≈ FedAvg *by design* — there is no
scanner difference for BN to specialize on. So a heterogeneity source is not
optional; it is the experiment.

## 1.7 What "done" looks like

A table + figure showing, **for each hospital**, the Dice of local-only vs FedAvg
vs FedBN (and the other methods), with the **outlier hospitals highlighted** —
demonstrating H1–H3 — plus:

- `results/per_hospital.csv` — rows = hospitals, cols = methods, values = mean Dice.
- `results/summary.csv` — per method: average, worst-hospital, and spread (fairness).
- `results/personalization_gain.png` — **FedBN − FedAvg per hospital**, the core
  figure: positive bars (especially for the worst hospitals) *are* H3.
- `results/comparison.png` — per-hospital Dice grouped by method.

## 1.8 Constraints that shaped every decision

- **Hardware:** a single **RTX 3050 laptop GPU (4 GB VRAM)**, 16 CPU cores, ~11 GB
  usable RAM. The 4 GB budget forces **2D slices** (not 3D volumes), small batches,
  AMP mixed precision, and **sequential** FL clients (one model on the GPU at a time).
- **Environment:** **WSL2 (Linux)**, because NVIDIA FLARE needs the POSIX `resource`
  module and UTF-8 source handling that native Windows lacks. Package manager is
  **`uv`** (never pip).
- **Performance reality:** the pipeline is **CPU-bound on data loading + transforms**,
  not GPU-bound — so DataLoader **workers**, not batch size, are the speed lever
  (see [Running §Performance](06-running.md#performance-model)).

## 1.9 Goals beyond the grade

The project is explicitly optimized for three things at once: **career value**
(a clean, real ML-systems artifact end-to-end), **publishability** (a genuine
personalized-FL result on medical imaging heterogeneity), and **real-world impact**
(the exact setting hospitals face). Every design choice is logged in
[Design decisions](08-design-decisions.md).
