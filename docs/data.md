# Data

## 1. Dataset — BraTS 2021

Brain Tumor Segmentation Challenge 2021, training set (Task 1).

| Property | Value |
|---|---|
| Cases (patients) | **1251** |
| Modality | 3D multi-parametric MRI |
| Per case | 4 MRI modalities + 1 segmentation mask = **5 volumes** |
| Modalities | FLAIR, T1, T1ce (contrast-enhanced), T2 |
| Volume shape | **240 × 240 × 155** voxels |
| Voxel spacing | 1.0 mm isotropic |
| Voxel dtype | int16 (modalities), uint8 (mask) |
| Preprocessing (by challenge) | co-registered to a common template, resampled to 1 mm, skull-stripped |

The data is **3D volumetric**, not 2D images — this is fundamental to the model choice.

### Segmentation labels

| Label | Meaning |
|---|---|
| 0 | background / healthy tissue |
| 1 | NCR — necrotic tumor core |
| 2 | ED — peritumoral edema |
| 4 | ET — enhancing tumor |

(Label 3 is unused, a historical BraTS convention.) Tumor voxels are only **~0.6%** of a volume —
a strong class imbalance the loss/metric must account for.

### Evaluation regions (derived from labels)

| Region | Composition |
|---|---|
| **WT** — whole tumor | labels 1 + 2 + 4 |
| **TC** — tumor core | labels 1 + 4 |
| **ET** — enhancing tumor | label 4 |

## 2. On-disk format & the two "unzip" layers

The data ships in **two layers of packaging**:

1. **Outer archive** — Kaggle serves everything as one `.zip` (containing a `.tar`). *Extracting*
   it just unpacks the folder structure; the files inside stay compressed.
2. **Inner compression** — each volume is a `.nii.gz` (gzip-compressed NIfTI). *Unzipping*
   decompresses each one to a raw `.nii` for fast random-access reads during training.

| Step | Input | Output | Size |
|---|---|---|---|
| Extract | 1 `.zip` | 6255 `.nii.gz` (still compressed) | ~13 GB → ~13 GB |
| Unzip | 6255 `.nii.gz` | 6255 raw `.nii` | ~13 GB → **~114 GB** |

The ~9× expansion is because ~99% of each volume is background zeros + smooth tissue that gzip
compresses heavily.

Per volume uncompressed: 240·240·155·2 bytes ≈ **17.9 MB**. Per case (5 volumes) ≈ **90 MB**.

## 3. Where the data lives

| Location | Form | Size | Purpose |
|---|---|---|---|
| `data/BraTS2021_Training_Data/` (local) | compressed `.nii.gz` | 13 GB | original source |
| `/mnt/d/data/unzipped/` (local D: drive) | raw `.nii` | 114 GB | local access / quick checks |
| `Drive/MyDrive/capstone/brats-2021-task1.zip` | compressed archive | 13 GB | durable source for Colab |
| `Drive/MyDrive/capstone/unzipped/` | raw `.nii` | 114 GB | **training data for Colab** |

> Local C: has little free space, so the local unzipped copy lives on the **D:** drive.
> Colab's local disk (~100 GB) can't hold all 114 GB unzipped at once, so on Colab the data lives
> in **Google Drive**, and each training session pulls only the working subset it needs.

## 4. Reproducing the prep

### Local (optional — for offline checks)
```bash
.venv/bin/python unzip_data.py   # decompress every .nii.gz -> .nii onto D:
```
Idempotent (skips done files) and atomic (temp-file + rename). Targets `/mnt/d/data/unzipped/`.

### Colab (the path we actually use for training)
Run [`../colab_setup.ipynb`](../colab_setup.ipynb) top-to-bottom:
1. Mount Google Drive.
2. Add a Kaggle API token (only needed the first time).
3. Get the archive — reused from Colab/Drive if present, else downloaded from Kaggle.
4. Extract the compressed cases on Colab's local disk.
5. **Stream-unzip in batches of 100:** unzip a batch → move it to Drive → delete local → repeat,
   so Colab's disk never fills. Resumable — re-run after any disconnect; finished cases are skipped.

Result: the full unzipped dataset in `Drive/MyDrive/capstone/unzipped/`, plus the compressed
archive kept alongside it.

## 5. Source

Kaggle: `dschettler8845/brats-2021-task1`
(<https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1>)
