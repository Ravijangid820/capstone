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
| 1. Docs & scaffolding | 🔨 in progress |
| 2. Model choice | ✅ decided — build **both** 2D & 3D (dimension-parametric, 2D first, 3D feasibility-gated) |
| 2b. Data pipeline (Colab) | ⬜ next |
| 3. Hospital partition + synthetic non-IID | ⬜ |
| 4. Model + centralized baseline | ⬜ |
| 5. FL methods (local / FedAvg / FedBN) | ⬜ |
| 6. Experiments | ⬜ |
| 7. Analysis (H1/H2/H3) | ⬜ |
| 8. Report | ⬜ |

## Documentation

- [`docs/methodology.md`](docs/methodology.md) — research design: heterogeneity, methods, model, evaluation, experiment matrix.
- [`docs/data.md`](docs/data.md) — dataset, labels, and the reproducible data-prep pipeline.
- [`docs/progress-log.md`](docs/progress-log.md) — dated lab notebook of decisions and milestones.

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
