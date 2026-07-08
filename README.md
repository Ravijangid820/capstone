# Personalized Federated Learning for Brain Tumor Segmentation

**Capstone project.** Can *personalized* federated learning (FedBN) give hospitals with
unusual scanners a better tumor-segmentation model than either a single shared global
model (FedAvg) or training alone (local-only) — without hurting everyone else?

We simulate several "hospitals" from the **BraTS 2021** MRI dataset, make them
heterogeneous with a controlled per-hospital scanner shift, and compare local-only,
FedAvg, and FedBN on brain-tumor segmentation.

## Research question & hypotheses

> When client data is non-IID because each hospital's scanner differs, does keeping the
> normalization layers **local** (FedBN) recover the hospitals that a single global model
> serves worst, while preserving the collaboration gain on average?

- **H1 — collaboration helps on average.** FedAvg ≥ local-only in mean Dice across hospitals.
- **H2 — the global model fails outliers.** FedAvg underperforms local-only on the hospital(s)
  whose scanner deviates most from the federation.
- **H3 — personalization recovers outliers.** FedBN matches/beats FedAvg on average *and*
  closes the outlier gap from H2.

## Status

| Phase | State |
|---|---|
| 0. Data acquisition & prep | ✅ done — compressed + unzipped in Google Drive; unzipped locally on D: |
| 1. Docs & design | ✅ done — full structured doc set (see below) |
| 2. Model choice | ✅ decided — build **both** 2D & 3D (dimension-parametric, 2D first, 3D feasibility-gated) |
| 3. Hospital partition (4 hospitals, 1 outlier) | 🔨 in progress — deterministic split |
| 4. Data pipeline (Colab) — preprocess + cache | ⬜ |
| 5. Model + centralized baseline | ⬜ |
| 6. FL methods (local / FedAvg / FedBN) | ⬜ |
| 7. Experiments | ⬜ |
| 8. Analysis (H1/H2/H3) → report | ⬜ |

## Documentation

Start at the [documentation index](docs/README.md). The set is split by concern so each doc stays focused:

| Doc | Scope |
|---|---|
| [methodology.md](docs/methodology.md) | Research design — question, hypotheses, methods, evaluation (the *why*) |
| [data.md](docs/data.md) | Dataset spec, labels, and the reproducible data-prep pipeline |
| [architecture.md](docs/architecture.md) | System architecture, end-to-end flow, module layout, logging strategy |
| [data-pipeline.md](docs/data-pipeline.md) | Case → hospital partition, synthetic shift, preprocessing, caching, sampling |
| [federated-learning.md](docs/federated-learning.md) | The FL round loop; FedAvg / FedBN / local-only aggregation |
| [experiments.md](docs/experiments.md) | Experiment matrix, evaluation protocol, how H1/H2/H3 are measured |
| [specs.md](docs/specs.md) | Reference sheet — hyperparameters, model dims, hardware, seeds, artifact layout |
| [progress-log.md](docs/progress-log.md) | Dated lab notebook of decisions and milestones |

## Repository layout

```
unzip_data.py        Local one-off: decompress .nii.gz -> .nii onto the D: drive
colab_setup.ipynb    Colab: download from Kaggle -> stream-unzip in batches -> Google Drive
docs/                Project documentation (see above)
pyproject.toml       Python environment (managed with uv)
```

## Compute

- **Local:** WSL2, RTX 3050 Laptop (4 GB VRAM) — used for data prep and quick checks only.
- **Training:** Google Colab **T4 (16 GB VRAM)**, with the dataset staged in Google Drive.

## Reproducing the data prep

See [`docs/data.md`](docs/data.md). In short: run `colab_setup.ipynb` in Colab to pull the
dataset from Kaggle and build the unzipped copy in `Drive/MyDrive/capstone/`.
