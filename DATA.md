# Data access guide — BraTS 2021 + FeTS 2022

> ⚠️ Start this **now** — data access is the critical path. The code is ready before the data is.

## What we need

1. **BraTS 2021 volumes** — the MRI scans + segmentation masks (the actual training data).
2. **FeTS 2022 institutional partitioning** — a CSV mapping each subject to a real hospital/institution. **This is what makes the project federated for real** (not a synthetic split). Prioritize getting this.

## Dataset structure (what you'll download)

Each subject is a folder with 4 MRI sequences + 1 label volume, all `.nii.gz` (3D, 240×240×155):

```
BraTS2021_00001/
├── BraTS2021_00001_flair.nii.gz   # FLAIR
├── BraTS2021_00001_t1.nii.gz      # T1
├── BraTS2021_00001_t1ce.nii.gz    # T1 contrast-enhanced
├── BraTS2021_00001_t2.nii.gz      # T2
└── BraTS2021_00001_seg.nii.gz     # labels: 1=necrotic, 2=edema, 4=enhancing tumor
```

Our code stacks the 4 sequences as a 4-channel input and converts the labels to the 3 BraTS regions (WT/TC/ET).

## Route A — fast start (recommended for Phase 1)

A **Kaggle mirror** of BraTS 2021 downloads immediately (no long application):
- Search Kaggle for **"BraTS 2021 Task 1"** (~15 GB). Common mirror: `dschettler8845/brats-2021-task-1`.
- Download with the Kaggle API (your GitHub Student pack works with Kaggle):
  ```powershell
  # after `pip`/`uv`-installing kaggle and placing kaggle.json in %USERPROFILE%\.kaggle\
  kaggle datasets download -d dschettler8845/brats-2021-task-1 -p data/ --unzip
  ```
- ⚠️ Exact Kaggle slugs change over time — if that one 404s, just search "BraTS 2021" and pick a mirror with the folder structure above.

Point our code at the extracted folder: `--data-root data/BraTS2021_TrainingData`.

## Route B — official (needed for the real FeTS partition)

For the **institutional partitioning** (the federated part), register with the official challenge:
- **Synapse** account → BraTS 2021 project (agree to the data-use terms).
- **FeTS 2022** challenge (`fets-ai`) → download the **partitioning CSVs** (`partitioning_1.csv` ≈ split by institution; `partitioning_2.csv` = a finer split). These map `Subject_ID → Partition_ID`.
- No CITI training required (unlike MIMIC) — just account + accepting terms. Lighter, but can take a day or two to approve. **Start it today.**

## Licensing / citation

BraTS & FeTS data are for **research use**; cite the BraTS and FeTS challenge papers in the report. Do **not** commit the data (it's in `.gitignore` under `data/`).

## Immediate action items

- [ ] Kaggle API set up + BraTS 2021 mirror downloading (Route A) → unblocks Phase 1 today.
- [ ] Synapse + FeTS registration started (Route B) → unblocks Phase 2's real federation.
- [ ] Confirm the extracted folder matches the structure above, then run the Phase-1 baseline.
