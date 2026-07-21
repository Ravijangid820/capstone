# Project Status Report: Personalized Federated Learning for Brain Tumor Segmentation

This document provides a comprehensive overview of the current status of our capstone project, detailing what has been completed, key findings, repository structure, and how to run the system. Share this with teammates, advisors, and collaborators to bring everyone up to speed.

---

## 1. Executive Summary

Our project investigates **Personalized Federated Learning (PFL)**, specifically **FedBN**, for 3D Brain Tumor Segmentation using the **BraTS 2021** MRI dataset. We simulate a 4-hospital network (3 typical, 1 outlier with a synthetic scanner shift) to evaluate:
1. **Local-only** training (individual models, the baseline/floor).
2. **FedAvg** (a single collaborative global model).
3. **FedBN** (collaborative model with localized batch-normalization layers).
4. **Centralized** training (combined data, the ceiling).

### Key Takeaway
We successfully built the data pipeline, the custom federated learning engine, 2D and 3D UNet models, and an interactive Web Demo. 
- **In 2D:** Personalization (FedBN) is crucial. A single global model (FedAvg) collapses on the outlier hospital, while FedBN successfully recovers performance.
- **In 3D:** Fully collaborative training (FedAvg) is superior. The high complexity of 3D networks means pooling data acts as a powerful regularizer, whereas FedBN suffers because 150 local cases are statistically insufficient to estimate stable 3D Batch Normalization statistics.

---

## 2. Project Status & Deliverables

| Phase / Deliverable | Status | Description |
|---|---|---|
| **0. Data Acquisition & Prep** | ✅ Complete | Stream-unzipped BraTS 2021 (~114 GB) in 100-case batches to manage storage constraints. |
| **1. Hospital Partition & Shift** | ✅ Complete | Deterministic split of 1251 cases into 4 hospitals (H1–H3 typical, H4 outlier). Simulated scanner shifts using Gamma contrast, RF bias fields, and Gaussian blur. |
| **2. Preprocessing & Caching** | ✅ Complete | Preprocessing pipeline with FP16 memory-mapped caching to speed up local training. |
| **3. Model & FL Engine** | ✅ Complete | Custom dimension-parametric PyTorch wrapper based on MONAI UNet. Built custom FL loop for Centralized, Local-only, FedAvg, and FedBN. |
| **4. 2D Experiments Matrix** | ✅ Complete | Ran all four methods ($R=25$ rounds). FedBN recovered the outlier scanner collapse. |
| **5. 3D Experiments Matrix** | ✅ Complete | Ran full 3D matrix. Discovered hypothesis reversal (FedAvg wins in 3D due to regularization). |
| **6. Interactive Web UI Demo** | ✅ Complete | Built a responsive dashboard featuring a 2D slice viewer, 3D rotatable WebGL tumor viewer (Three.js), simulated scanner shift controls, and live model inference. |
| **7. Academic Reporting** | ✅ Complete | Fully detailed project reports, methodology specs, and generated plots under `docs/` and `artifacts/`. |

---

## 3. Key Findings & Scientific Insights

We evaluated the final-round Whole Tumor (WT) Dice score across methods:

### 2D vs. 3D WT Dice Score Comparison

| Backbone | Centralized | Local-only | FedAvg | FedBN (PFL) | Main Insight |
|---|---|---|---|---|---|
| **2D Backbone** | 0.852 | 0.853 | 0.835 | **0.852** | FedAvg collapsed on the outlier H4 (0.737), but **FedBN recovered it (0.829)**. |
| **3D Backbone** | 0.880 | 0.851 | **0.859** | 0.834 | FedAvg did not collapse on H4 (0.848). **FedBN degraded (0.833)**. |

### The 2D-vs-3D Personalization Divergence
- **2D Slice Density:** Training in 2D yields ~12,000 slices per hospital. This high sample density allows FedBN to calculate stable local Batch Normalization running statistics, successfully adapting to the scanner shift.
- **3D Collaborative Regularization:** In 3D, we only have 150 volumes per hospital. The 3D U-Net is highly complex; local BN statistics degenerate due to overfitting on small sample sizes. By pooling updates globally, FedAvg acts as a regularizer, outperforming FedBN and Local-only on the outlier hospital.

---

## 4. Repository Structure & Guided Tour

Teammates can find components in the following locations:

```
├── .agents/                 # Workspace agent rules
├── artifacts/
│   ├── figures/             # Learning curves & comparison plots
│   ├── splits/              # Deterministic partition manifest (partition.json)
│   └── runs/                # Raw metrics from runs (metrics.jsonl)
├── docs/                    # Deep-dive documentation markdown files
│   ├── methodology.md       # Hypotheses (H1/H2/H3) & study design
│   ├── workflow.md          # Step-by-step pipeline execution guide
│   ├── data.md              # BraTS data spec, labels, and regions
│   ├── architecture.md      # System components & data flow diagrams
│   └── progress-log.md      # Historical lab notebook of project milestones
├── scripts/
│   ├── build_partition.py   # Script to generate hospital partition JSON
│   ├── build_cache.py       # Preprocesses and caches volumes to local disk
│   ├── run_experiment.py    # Main training entry point (supports all 4 FL methods)
│   ├── analyze.py           # Evaluates runs, calculates metrics, tests hypotheses
│   ├── plot_results.py      # Generates results plots from run outputs
│   └── demo_server.py       # Starts the interactive Web UI Dashboard
├── src/fedbrats/            # Core library packages
│   ├── data.py              # Data loaders, dataset classes, and caching
│   ├── federated.py         # Federated aggregation logic (FedAvg / FedBN)
│   ├── model.py             # Custom MONAI UNet wrapper (dimension-parametric)
│   ├── shift.py             # Scanner shift simulation logic
│   └── static/              # Frontend files for the Web UI (HTML, CSS, JS, WASM)
├── pyproject.toml           # Package and dependency configuration (managed via uv)
└── README.md                # General quickstart and entry index
```

---

## 5. How to Run the Project (Quickstart)

Ensure you have `uv` installed. If not, install it via standalone installer or pip.

### 5.1 Environment Setup
```bash
# Sync dependencies and create virtual environment
uv sync
```

### 5.2 Fast Pipeline Smoke Test (Runs in ~2 minutes)
To verify everything is working locally on any OS (CPU or GPU):
```bash
# 1. Build deterministic splits
uv run python scripts/build_partition.py

# 2. Build a tiny cache (3 cases)
uv run python scripts/build_cache.py --max-cases 3 --workers 4

# 3. Run a quick FedBN experiment (2 rounds, small subsets)
uv run python scripts/run_experiment.py --method fedbn --rounds 2 --max-train-cases 3 --max-test-cases 2
```

### 5.3 Launch the Web UI Dashboard
Teammates can run and interact with the segmentations and scanner shifts directly:
```bash
uv run python scripts/demo_server.py
```
Then, open [http://localhost:8000](http://localhost:8000) in your browser.
**Dashboard features:**
- Interactive 2D slice selector with overlay options for T1, T1ce, T2, FLAIR.
- Live 3D rotatable mesh rendering (powered by Three.js/WebGL).
- Live model inference comparison side-by-side.
- Scanner shift simulator toggles.

---

## 6. What Teammates Can Focus On Next

If you want to collaborate or expand this work:
1. **Optimization of 3D Patch Sizes:** Experiment with larger patch sizes (e.g., $128 \times 128 \times 128$) on higher-VRAM GPUs.
2. **Alternative Normalization Layers:** Since Batch Normalization degrades under client data scarcity, test Group Normalization (GroupNorm) or Layer Normalization (LayerNorm) which do not rely on batch statistics and might improve Federated Learning robustness.
3. **NVIDIA FLARE Integration:** Extend the local simulation to a distributed server-client deployment using NVIDIA FLARE.
