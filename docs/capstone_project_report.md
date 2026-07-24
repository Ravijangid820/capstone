# Personalized Federated Learning for Brain Tumor Segmentation
## Technical Research Report & System Architecture Specification

---

## Abstract

In multi-institutional medical imaging, privacy regulations (e.g., HIPAA, GDPR) prohibit pooling patient data across hospitals. However, differences in MRI acquisition protocols, magnetic field strengths, and scanner manufacturers create domain heterogeneity (non-IID data). Under these conditions, standard Federated Averaging (**FedAvg**) produces a single global model that performs poorly on outlier institutions. 

This project presents a comparative study of **Local-only** training, **FedAvg**, and **FedBN** (Federated Learning with Local Batch Normalization) for 3D multi-parametric brain tumor segmentation using the **BraTS 2021** dataset ($1,251$ multi-modal cases). Using a **dimension-parametric framework**, we evaluate performance across both **2D** and **3D** U-Net backbones under a controlled, non-linear per-hospital scanner shift. 

Our findings demonstrate a **dimension-dependent hypothesis reversal**:
1. **In 2D:** FedAvg collapses on an outlier hospital ($0.737$ Whole Tumor Dice vs. $0.857$ Local-only). FedBN personalizes normalization, recovering the outlier to **$0.829$ Dice** and matching the Centralized ceiling mean (**$0.852$ Dice**) without data pooling.
2. **In 3D:** 3D spatial convolutions act as a regularizer, making FedAvg robust to scanner shifts ($0.848$ outlier Dice vs $0.819$ Local-only). Conversely, FedBN degrades performance ($0.834$ mean Dice) because estimating 3D Batch Normalization running statistics on small local datasets ($150$ cases) introduces high variance.

---

## 1. Introduction & Theoretical Motivation

### 1.1 The Non-IID Problem in Medical Imaging
Federated Learning (FL) enables collaborative model training across decentralized hospitals. The canonical FL algorithm, **FedAvg**, computes a weighted average of all client model parameters:

$$\theta_{\text{global}}^{(r)} = \sum_{k=1}^K \frac{n_k}{N} \theta_k^{(r)}$$

When client datasets are non-IID due to scanner variation, feature distributions vary across institutions ($P_i(X) \neq P_j(X)$). Standard FedAvg forces a single global normalization, causing performance collapse on institutions whose scanner statistics deviate significantly from the population mean.

### 1.2 FedBN Mechanism
FedBN addresses feature shift by keeping Batch Normalization (BN) parameters local to each client while aggregating all convolutional and linear layers. For a layer with input $x$, BatchNorm computes:

$$\hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

Under FedBN:
* **Global Aggregation:** Convolutional filters $W_{\text{conv}}$ are averaged globally across clients to learn domain-invariant structural feature extractors.
* **Local Statistics:** Running statistics $(\mu_{\text{running}}, \sigma^2_{\text{running}})$ and affine parameters $(\gamma, \beta)$ are kept **strictly local** to each hospital, allowing each site to normalize features relative to its local scanner distribution.

---

## 2. Research Hypotheses & Mathematical Formulations

To evaluate collaboration, outlier failure, and personalization recovery, we test three formal hypotheses against a **Centralized Ceiling ($E_0$)** (trained on pooled data) and a **Local-Only Floor ($E_1$)** (trained independently without communication):

* **H1 — Average Collaboration Benefit:**
  $$\text{Mean Dice}_{\text{FedAvg}} \ge \text{Mean Dice}_{\text{Local-Only}}$$

* **H2 — Outlier Global Failure:**
  $$\text{Dice}_{\text{FedAvg}}(H_{\text{outlier}}) < \text{Dice}_{\text{Local-Only}}(H_{\text{outlier}})$$

* **H3 — Personalization Recovery:**
  $$\text{Mean Dice}_{\text{FedBN}} \ge \text{Mean Dice}_{\text{FedAvg}} \quad \text{and} \quad \text{Dice}_{\text{FedBN}}(H_{\text{outlier}}) \ge \text{Dice}_{\text{FedAvg}}(H_{\text{outlier}})$$

---

## 3. Dataset Specifications & Preprocessing Pipeline

### 3.1 Dataset Overview (BraTS 2021)
* **Dataset:** RSNA-MICCAI Brain Tumor Segmentation Challenge 2021 (Task 1).
* **Total Volume:** $1,251$ multi-parametric 3D MRI cases.
* **Spatial Resolution:** $240 \times 240 \times 155$ voxels at $1.0\text{ mm}^3$ isotropic spacing.
* **Modalities per Case (4 Channels):**
  1. `FLAIR`: Fluid Attenuated Inversion Recovery (suppresses fluid, highlights edema).
  2. `T1`: Native T1-weighted (structural tissue anatomy).
  3. `T1ce`: Contrast-Enhanced T1-weighted (highlights active vascularized tumor edge).
  4. `T2`: T2-weighted (high water sensitivity, illuminates edema/fluid).

### 3.2 Data Types & Precision Across Pipeline Stages

| Pipeline Stage | MRI Modalities | Segmentation Mask | Storage / Computational Rationale |
| :--- | :---: | :---: | :--- |
| **Raw NIfTI Files (`.nii`)** | `int16` | `uint8` | Standard NIfTI scanner range and label integers $\{0, 1, 2, 4\}$. |
| **In-Memory Preprocessing** | `float32` | `uint8` | High-precision continuous intensity transformation and z-scoring. |
| **Cached Tensors (`.npy`)** | `float16` | `uint8` | Memory-mapped arrays; reduces disk footprint to **~35 MB/case** (~44 GB total). |
| **PyTorch Model Tensors** | `float32` | `float32` | Prevents numerical instability / corruption of BatchNorm stats in CUDA. |

### 3.3 Target Region Formulations & Medical Anatomical Definitions

Raw ground truth labels contain integer values $\{0, 1, 2, 4\}$ (label 3 is historically unused). The model predicts **3 binary target region channels** using independent Sigmoid units:

```text
                     ┌──────────────────────────────────────────┐
                     │            Whole Tumor (WT)              │
                     │          Labels: 1 + 2 + 4               │
                     │  ┌────────────────────────────────────┐  │
                     │  │          Tumor Core (TC)           │  │
                     │  │           Labels: 1 + 4            │  │
                     │  │  ┌──────────────────────────────┐  │  │
                     │  │  │    Enhancing Tumor (ET)      │  │  │
                     │  │  │          Label 4             │  │  │
                     │  │  └──────────────────────────────┘  │  │
                     │  └────────────────────────────────────┘  │
                     └──────────────────────────────────────────┘
```

1. **WT (Whole Tumor):** Labels $\{1, 2, 4\}$. Complete tumor mass including edema.
2. **TC (Tumor Core):** Labels $\{1, 4\}$. Solid tumor mass excluding peritumoral edema ($\text{TC} = \text{NCR} + \text{ET}$). Primary target for surgical resection.
3. **ET (Enhancing Tumor):** Label $\{4\}$. Actively proliferating vascularized tumor rim.

---

## 4. Domain Heterogeneity & Calibrated Scanner Shift

### 4.1 Synthetic Scanner Shift Design
Linear intensity shifts ($a \cdot x + b$) are completely removed by per-volume z-score normalization:

$$\hat{x} = \frac{(a \cdot x + b) - \mu_{a \cdot x + b}}{\sigma_{a \cdot x + b}} = \frac{x - \mu_x}{\sigma_x}$$

To ensure heterogeneity survives z-normalization, we implemented a **nonlinear, spatial scanner shift** (`src/fedbrats/shift.py`) composed of three operators:
1. **Nonlinear Gamma Contrast:** $x_{\text{gamma}} = \text{norm}(x)^\gamma \cdot (v_{\max} - v_{\min}) + v_{\min}$
2. **Spatial Multiplicative Bias Field:** $x_{\text{bias}} = x \cdot (1 + A \cdot B(x, y, z))$, where $B$ is a smooth cubic-spline 3D noise field generated from $4 \times 4 \times 4$ control points.
3. **Gaussian Blur:** Spatial Point Spread Function (PSF) variation via Gaussian filtering with kernel parameter $\sigma_{\text{blur}}$.

### 4.2 Hospital Shift Parameters & Partition Manifest

Cases are partitioned deterministically into $K=4$ hospitals using seed $42$. The assignment manifest is saved at `artifacts/splits/partition.json`.

| Hospital ID | Assigned Cases | Train Cases | Test Cases | Gamma ($\gamma$) | Bias Field Amp ($A$) | Blur Sigma ($\sigma$) | Domain Shift Type |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **$H_1$** | $313$ | $251$ | $62$ | $1.06$ | $0.06$ | $0.3$ | Mild baseline |
| **$H_2$** | $313$ | $251$ | $62$ | $1.13$ | $0.09$ | $0.5$ | Mild contrast shift |
| **$H_3$** | $313$ | $251$ | $62$ | $1.20$ | $0.12$ | $0.7$ | Moderate shift |
| **$H_4$** | **$312$** | **$250$** | **$62$** | **$1.85$** | **$0.34$** | **$1.7$** | **Outlier (Severe Shift)** |

*After z-normalization, Hospital 4 maintains a clear outlier margin of **$+0.149\sigma$** relative to typical sites.*

### 4.3 Prevention of Geometry Leakage (The Blur-Halo Safeguard)
Applying Gaussian blur causes brain voxel intensities to spill over into BraTS's zero-valued background. Computing brain masks *after* applying the shift causes $H_4$'s mask to inflate by **$+57\%$ (+833,379 voxels)**, altering volume geometry and leaking hospital identity as crop shape.

**Solution:** Brain masks and bounding boxes are computed from **unshifted volumes**, applied to shifted volumes, and re-masked:

$$\text{Volume}_{\text{final}} = \text{ApplyShift}(\text{Volume}_{\text{raw}}, H_k) \odot \text{Mask}_{\text{unshifted}}$$

---

## 5. System Architecture & Dimension-Parametric Model

The pipeline relies on a unified, dimension-parametric **MONAI U-Net** (`src/fedbrats/model.py`).

```text
  ┌────────────────────────────────────────────────────────┐
  │                 Config (dim = 2d | 3d)                 │
  └───────────────────────────┬────────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
            [ 2D ]                        [ 3D ]
               │                             │
               ▼                             ▼
  ┌──────────────────────────┐  ┌──────────────────────────┐
  │ Axial Slice Sampler      │  │ Cubic Patch Sampler      │
  │ (192 × 192)              │  │ (96 × 96 × 96)           │
  └────────────┬─────────────┘  └────────────┬─────────────┘
               │                             │
               ▼                             ▼
  ┌──────────────────────────┐  ┌──────────────────────────┐
  │ MONAI U-Net              │  │ MONAI U-Net              │
  │ (Conv2d, Base 32)        │  │ (Conv3d, Base 16)        │
  └────────────┬─────────────┘  └────────────┬─────────────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │                   SHARED PIPELINE                      │
  │  • FL Engine (Sequenced on 1 GPU)                      │
  │  • Aggregators (FedAvg / FedBN)                        │
  │  • Per-Volume Evaluation Metrics (Dice WT, TC, ET)     │
  └────────────────────────────────────────────────────────┘
```

### 5.1 Architecture & Loss Function Specifications

$$\mathcal{L}_{\text{total}}(y, \hat{y}) = \mathcal{L}_{\text{Dice}}(\sigma(\hat{y}), y) + \mathcal{L}_{\text{BCE}}(\hat{y}, y)$$

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum p_i g_i + \epsilon}{\sum p_i^2 + \sum g_i^2 + \epsilon}, \quad \mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum [g_i \log(\sigma(p_i)) + (1-g_i)\log(1-\sigma(p_i))]$$

| Parameter | 2D Model Backbone | 3D Model Backbone |
| :--- | :--- | :--- |
| **Convolution Type** | `Conv2d` | `Conv3d` |
| **Channel Progression** | $32 \rightarrow 64 \rightarrow 128 \rightarrow 256$ | $16 \rightarrow 32 \rightarrow 64 \rightarrow 128$ |
| **Normalization Layers** | `BatchNorm2d` | `BatchNorm3d` |
| **Residual Units** | $2$ per resolution level | $2$ per resolution level |
| **Input Unit / Shape** | Axial slice ($192 \times 192$ cropped) | Cubic patch ($96 \times 96 \times 96$) |
| **Batch Size** | $8$ | $1$ |
| **Sampling Units per Case** | $8$ slices / case / epoch | $2$ patches / case / epoch |
| **Foreground Bias (`tumor_frac`)** | $0.7$ ($70\%$ tumor slice chance) | $0.7$ ($70\%$ tumor patch chance) |

---

## 6. Federated Learning Engine & Algorithmic Design

### 6.1 Unified Round Loop (`src/fedbrats/federated.py`)
All four methods share a unified execution loop parameterized by two switches:

| Method | Aggregate Weights? | Keep BN Local? | Pooled Data? | Role |
| :--- | :---: | :---: | :---: | :--- |
| **Centralized ($E_0$)** | No | — | Yes | Upper Ceiling |
| **Local-Only ($E_1$)** | No | — | No | Lower Floor |
| **FedAvg ($E_2$)** | Yes | No | No | Global Baseline |
| **FedBN ($E_3$)** | Yes | Yes | No | Personalized FL |

### 6.2 BatchNorm Key Identification & Buffer Preservation
In PyTorch, BatchNorm parameters include trainable weights $(\gamma, \beta)$ and non-trainable tracking buffers (`running_mean`, `running_var`, `num_batches_tracked`).

```python
def bn_keys(model: nn.Module) -> set[str]:
    """Identify BatchNorm keys by module type rather than string matching."""
    keys: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            prefix = f"{name}." if name else ""
            keys.update(prefix + k for k in module.state_dict())
    return keys
```

> **Critical Handling of `num_batches_tracked`:** `num_batches_tracked` is stored as an `int64` tensor. Computing a weighted float average corrupts PyTorch state loading. In `weighted_average()`, non-floating-point tensors are **copied directly from the heaviest client** rather than averaged.

### 6.3 Evaluation Protocol Timing
Evaluation occurs **after aggregation and before local training**:

$$\text{State}_{\text{eval}}(H_k) = \theta_{\text{global}}^{(r)} \cup \text{BN}_k^{(r-1)}$$

*Scoring after local training would grant FedAvg a round of local adaptation, masking global model failure and invalidating H2.*

---

## 7. Experimental Setup & Hardware Profiling

### 7.1 Execution Sequencing & Matched Compute
To enforce rigorous experimental control, all methods are evaluated under **matched compute**:
* **Communication Rounds ($R$):** $25$ rounds.
* **Local Epochs ($E$):** $1$ epoch per round.
* **Total Training Budget:** $R \times E = 25$ total epochs for all methods (including Centralized and Local-only).

```text
  Step 0: PREPROCESSING (Ran Once)
  ┌────────────────────────────────────────────────────────┐
  │ Build Cache & Partition Manifest                       │
  │  • deterministic split -> partition.json (Seed 42)     │
  │  • preprocessed tensors -> cache/ (float16 / uint8)     │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  Step 1: CENTRALIZED MODEL (E0) — Ceiling Baseline
  ┌────────────────────────────────────────────────────────┐
  │ Train 1 Model on Pooled Data (25 Epochs)               │
  │  • Sanity Gate: Must achieve WT Dice > 0.80             │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  Step 2: LOCAL-ONLY MODELS (E1) — Floor Baseline
  ┌────────────────────────────────────────────────────────┐
  │ Train 4 Separate Local Models (25 Epochs Each)         │
  │  • Hospital H1, H2, H3, H4 train independently         │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  Step 3: FEDAVG MODEL (E2) — Global FL Baseline
  ┌────────────────────────────────────────────────────────┐
  │ Run 25 Communication Rounds (Average ALL Weights)      │
  │  • Evaluates H1 (Avg Benefit) & H2 (Outlier Failure)   │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  Step 4: FEDBN MODEL (E3) — Personalized FL
  ┌────────────────────────────────────────────────────────┐
  │ Run 25 Communication Rounds (Keep BatchNorm Local)     │
  │  • Evaluates H3 (Personalization Outlier Recovery)     │
  └───────────────────────────┘
```

### 7.2 Hardware Benchmark & Throughput Profiling

| Workload Stage | RTX 3050 Laptop (4 GB VRAM) | Google Colab T4 (16 GB VRAM) |
| :--- | :---: | :---: |
| **Training Step (2D, Batch 8)** | $48\text{ ms / step}$ | ~$32\text{ ms / step}$ |
| **Full-Volume Evaluation** | $0.41\text{ s / volume}$ | ~$0.27\text{ s / volume}$ |
| **Preprocess & Cache Time** | $2.3\text{ s / case}$ (4 workers) | ~$1.4\text{ s / case}$ (8 workers) |
| **One Round Time (Train + Eval)** | $2.2\text{ min / round}$ | ~$1.5\text{ min / round}$ |
| **Full 4-Run Matrix (2D)** | ~$3.7\text{ hours}$ | ~$2.5\text{ hours}$ |

> **Evaluation Bottleneck Insight:** Full-volume inference ($248$ test volumes/round) accounts for **~$75\%$ of total round runtime** ($1.7\text{ min}$ evaluation vs $0.5\text{ min}$ training per round).

---

## 8. Quantitative Results & Empirical Evaluation

### 8.1 2D Backbone Results (Final-Round WT Dice Scores)
*Evaluated on local test sets at Round 25 ($R=25, E=1$, Seed 42, 150 train cases/hospital):*

| Method | Mean WT Dice | $H_1$ (Typical) | $H_2$ (Typical) | $H_3$ (Typical) | $H_4$ (**Outlier**) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized** *(Ceiling)* | **0.852** | 0.866 | 0.868 | 0.844 | **0.828** |
| **Local-Only** *(Floor)* | **0.853** | 0.848 | 0.863 | 0.842 | **0.857** |
| **FedAvg** *(Global FL)* | **0.835** | **0.883** | **0.884** | 0.838 | **0.737** ❌ *(Outlier Collapse)* |
| **FedBN** *(Personalized)* | **0.852** | 0.866 | 0.866 | 0.849 | **0.829** ✅ *(Outlier Recovered)* |

#### 2D Sub-Region Breakdown (Mean Dice Across Hospitals)
* **Centralized:** WT: **0.852** | TC: **0.835** | ET: **0.794**
* **Local-Only:** WT: **0.853** | TC: **0.831** | ET: **0.787**
* **FedAvg:** WT: **0.835** | TC: **0.817** | ET: **0.764**
* **FedBN:** WT: **0.852** | TC: **0.828** | ET: **0.779**

---

### 8.2 3D Backbone Results (Final-Round WT Dice Scores)
*Evaluated on local test sets at Round 25 ($R=25, E=1$, Seed 42, 150 train cases/hospital):*

| Method | Mean WT Dice | $H_1$ (Typical) | $H_2$ (Typical) | $H_3$ (Typical) | $H_4$ (**Outlier**) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Centralized** *(Ceiling)* | **0.880** | 0.899 | 0.896 | 0.878 | **0.848** |
| **Local-Only** *(Floor)* | **0.851** | 0.871 | 0.872 | 0.844 | **0.819** |
| **FedAvg** *(Global FL)* | **0.859** | 0.862 | 0.866 | 0.859 | **0.848** ✅ *(Robust)* |
| **FedBN** *(Personalized)* | **0.834** | 0.844 | 0.862 | 0.797 | **0.833** ❌ *(Degraded)* |

---

### 8.3 Local-Only 4x4 Cross-Hospital Generalization Matrix (2D)
*To verify that the synthetic scanner shift induces a genuine domain gap, each Local-only model was evaluated cross-institutionally at Round 25:*

| Trained On $\downarrow$ \ Tested On $\rightarrow$ | $H_1$ Test | $H_2$ Test | $H_3$ Test | $H_4$ Test (**Outlier**) |
| :--- | :---: | :---: | :---: | :---: |
| **Model $H_1$** | **0.848** | 0.831 | 0.798 | **0.671** ❌ |
| **Model $H_2$** | 0.822 | **0.863** | 0.811 | 0.694 |
| **Model $H_3$** | 0.795 | 0.814 | **0.842** | 0.712 |
| **Model $H_4$ (Outlier)** | 0.689 | 0.705 | 0.718 | **0.857** |

*The severe degradation on off-diagonal cells (Model $H_1$ scoring only $0.671$ on $H_4$) confirms a substantial domain gap.*

---

### 8.4 Hypothesis Decision Matrix

| Hypothesis | Verification Inequality | 2D Result | 3D Result | Empirical Conclusion |
| :--- | :--- | :---: | :---: | :--- |
| **H1: Average Collaboration Benefit** | $\text{Mean(FedAvg)} \ge \text{Mean(Local)}$ | ❌ **Failed** ($0.835$ vs $0.853$) | ✅ **Supported** ($0.859$ vs $0.851$) | 2D FedAvg mean dragged down by $H_4$ collapse. 3D FedAvg benefits from regularization. |
| **H2: Outlier Global Failure** | $\text{Dice(FedAvg, H4)} < \text{Dice(Local, H4)}$ | ✅ **Supported** ($0.737$ vs $0.857$) | ❌ **Failed** ($0.848$ vs $0.819$) | 2D FedAvg collapses under shift. 3D spatial convolutions act as domain regularizer. |
| **H3: Personalization Recovery** | $\text{FedBN(H4)} \ge \text{FedAvg(H4)} \land \text{Mean(FedBN)} \ge \text{Mean(FedAvg)}$ | ✅ **Supported** ($0.829 \ge 0.737$<br/>$0.852 \ge 0.835$) | ❌ **Failed** ($0.833 < 0.848$<br/>$0.834 < 0.859$) | FedBN recovers 2D outlier. In 3D, estimating local BN statistics on small sets ($150$ cases) degrades model. |

---

## 9. Theoretical Discussion: The 3D Hypothesis Reversal

```text
                              THE HYPOTHESIS REVERSAL
                              
         2D BACKBONE                                    3D BACKBONE
┌───────────────────────────┐                  ┌───────────────────────────┐
│ • FedAvg Fails H4 (0.737) │                  │ • FedAvg Beats Local H4   │
│ • FedBN Recovers H4(0.829)│                  │   (0.848 vs 0.819)        │
│ • FedBN Wins (Mean 0.852) │                  │ • FedBN Fails (Mean 0.834)│
└─────────────┬─────────────┘                  └─────────────┬─────────────┘
              │                                              │
              ▼                                              ▼
   Feature Statistics Driven                      Spatial Regularization Driven
   Domain Gap Dominates                           Scarcity & Variance Dominate
```

### Explanatory Factors for 3D Behavior
1. **Spatial Convolutional Regularization:** A 3D U-Net possesses significantly higher parameter complexity. Training locally on $150$ 3D volumes causes severe local overfitting. In 3D FedAvg, federated parameter averaging enforces strong collaborative regularization across clients, improving generalization on $H_4$ from **$0.819$ (Local) to $0.848$ (FedAvg)**.
2. **Local BatchNorm Running Statistics Variance:** In 3D architectures, feature maps maintain extensive spatial dimensions. Estimating running statistics ($\mu_{\text{running}}, \sigma^2_{\text{running}}$) locally over only $150$ volumes introduces high sampling variance. Keeping BN local in 3D destabilizes normalization layers, causing **FedBN ($0.834$ Mean WT)** to underperform **FedAvg ($0.859$ Mean WT)**.

---

## 10. Web Demonstration Architecture (`scripts/demo_server.py`)

An interactive HTTP server and WebGL front-end (`http://localhost:8000`) provides interactive model validation:

```text
  ┌────────────────────────────────────────────────────────┐
  │              Browser Front-End (app.js)                │
  └──────┬────────────────────┬────────────────────┬───────┘
         │                    │                    │
         │ GET /api/view      │ POST /api/predict  │ POST /api/mesh
         ▼                    ▼                    ▼
  ┌────────────────────────────────────────────────────────┐
  │        Python HTTP Server (scripts/demo_server.py)     │
  └──────┬────────────────────┬────────────────────┬───────┘
         │                    │                    │
         ▼                    ▼                    ▼
  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
  │ Raw BraTS   │      │ On-the-Fly  │      │ Checkpoint  │
  │ NIfTI Data  │      │ Shift (H1-4)│      │ Model (pt)  │
  └─────────────┘      └─────────────┘      └──────┬──────┘
                                                   │
                                                   ▼
                                            ┌─────────────┐
                                            │ Inference   │
                                            └──────┬──────┘
                                                   │
               ┌───────────────────────────────────┴───────────────────────────────────┐
               ▼                                                                       ▼
  ┌─────────────────────────┐                                             ┌─────────────────────────┐
  │ Base64 Slice Overlays   │                                             │ Binary Volume Bytes     │
  └────────────┬────────────┘                                             └────────────┬────────────┘
               │                                                                       │
               ▼                                                                       ▼
  ┌─────────────────────────┐                                             ┌─────────────────────────┐
  │ 2D Image Display        │                                             │ WASM / WebGL            │
  │ (FLAIR/T1/T1ce/T2)      │                                             │ Marching Cubes          │
  └─────────────────────────┘                                             └────────────┬────────────┘
                                                                                       │
                                                                                       ▼
                                                                          ┌─────────────────────────┐
                                                                          │ 3D Rotatable Mesh       │
                                                                          │ (Three.js WebGL)        │
                                                                          └─────────────────────────┘
```

### API Endpoints
* `GET /api/cases`: Returns `index.json` metadata for cached cases.
* `GET /api/view?case_id=&slice_idx=&modality=&hospital=`: Generates Base64 PNG slices for MRI modalities and ground truth overlays under real-time scanner shifts.
* `POST /api/predict`: Runs multi-model inference ($E_0, E_1, E_2, E_3$) and computes slice/volume Dice scores.
* `POST /api/mesh`: Exports thresholded binary volume buffers to WebAssembly-accelerated **Marching Cubes** for real-time 3D rendering.

---

## 11. Methodological & Implementation Reference

### 11.1 Key Technical Answers for System Defense

#### Q1: "How is Centralized ($E_0$) training conducted when federated learning forbids data sharing?"
> **Answer:** Centralized training is executed as a theoretical control experiment ($E_0$) by pooling training splits on a single machine. It establishes the empirical performance ceiling ($0.852$ WT Dice in 2D, $0.880$ in 3D). In all federated runs ($E_2, E_3$), **no image data is transmitted**; clients exchange only model parameter tensors.

#### Q2: "How are BatchNorm layers identified during FedBN aggregation?"
> **Answer:** Rather than performing string matching on parameter names (which fails when libraries emit dynamic layer keys like `net.model.0.conv.unit0.adn.N.weight`), the aggregator inspects PyTorch module types via `isinstance(module, nn.modules.batchnorm._BatchNorm)`. This extracts all $65$ BN state keys, ensuring buffers and affine parameters are preserved locally.

#### Q3: "Why is evaluation performed at the volume level rather than slice level?"
> **Answer:** Slices without tumor tissue yield a $1.0$ Dice score automatically when predictions are empty, inflating mean scores. We perform slice-by-slice inference, reconstruct the 3D volume, and evaluate **per-volume Dice**, maintaining strict adherence to official BraTS challenge evaluation protocols.

#### Q4: "Why is float32 precision maintained over Automatic Mixed Precision (AMP/float16)?"
> **Answer:** Empirical testing revealed that float16 gradients under severe scanner shifts introduce numerical instability into BatchNorm running statistics, leading to NaN loss propagation. Maintaining float32 precision ensures stable normalization tracking.

---

## 12. Repository Structure & Artifact Layout

```
src/fedbrats/
├── config.py          Dataclasses for paths, hospital split specs, hyperparameters
├── data.py            Load, preprocess, memory-mapped cache, 2D/3D samplers
├── shift.py           Nonlinear scanner shift operators (Gamma, Bias field, Blur)
├── model.py           MONAI BratsUNet wrapper (2D/3D), bn_keys() extractor
├── metrics.py         Per-volume Dice computation (WT, TC, ET)
├── train.py           Local training loop, sliding-window & 2D volume inference
├── federated.py       Unified FL engine (Centralized, Local, FedAvg, FedBN)
└── static/            Web UI assets (HTML, CSS, JS, WASM Marching Cubes)

scripts/
├── build_partition.py Deterministic hospital split generator
├── build_cache.py     Parallel memory-mapped dataset cache builder
├── run_experiment.py  Main execution entrypoint (--method, --dim, --rounds)
├── analyze.py         Metrics JSONL analyzer & hypothesis decision engine
├── plot_results.py    Generates learning curves and regional bar charts
└── demo_server.py     Interactive HTTP/WebGL visual dashboard server

artifacts/
├── splits/partition.json        Committed split manifest (Seed 42)
├── cache/<hash>/                Cached float16 modality & uint8 mask tensors
├── figures/                     Generated PNG plots (2D and 3D)
└── runs/<run_id>/               Run logs, checkpoints, and metrics.jsonl
```
