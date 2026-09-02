# Data

Everything about the dataset and how a raw scan becomes a training batch: the source data, the
four-hospital partition, the synthetic scanner shift, preprocessing, sampling and the cache.

> Hospitals are keyed `H1`–`H4` in code and logs. In prose they are **Site A–D**, with
> **Site D = `H4` = the outlier**. `H1`–`H3` unqualified always mean **hypotheses**.
> See [conventions.md](conventions.md).

**Related:** [training.md](training.md) for what is done with this data · [code.md](code.md) for
the modules that implement it.

---

## 1. The dataset — BraTS 2021

Brain Tumor Segmentation Challenge 2021, training set (Task 1).

| Property | Value |
|---|---|
| Cases (patients) | **1,251** |
| Modality | 3D multi-parametric MRI |
| Per case | 4 MRI modalities + 1 segmentation mask = **5 volumes** |
| Modalities | FLAIR, T1, T1ce (contrast-enhanced), T2 |
| Volume shape | **240 × 240 × 155** voxels |
| Voxel spacing | 1.0 mm isotropic |
| Voxel dtype | int16 (modalities), uint8 (mask) |
| Prepared by the challenge | co-registered to a common template, resampled to 1 mm, skull-stripped |

The data is **3D volumetric**, not 2D images — which is why the study runs both backbones.

### Why four modalities

No single sequence shows the whole tumour, so all four are stacked as input channels.

| Modality | What it makes visible |
|---|---|
| **FLAIR** | Oedema — the full extent of the abnormality |
| **T1** | Anatomy; tumour appears dark |
| **T1ce** | Post-contrast — the enhancing rim glows |
| **T2** | Fluid — oedema and cysts |

### Labels and evaluation regions

| Label | Meaning |
|---|---|
| 0 | background / healthy tissue |
| 1 | NCR — necrotic tumour core |
| 2 | ED — peritumoral oedema |
| 4 | ET — enhancing tumour |

Label 3 is unused, a historical BraTS convention. Tumour voxels are only **~0.6%** of a volume — a
strong class imbalance the loss and metric must account for.

The three reported regions are derived from those labels and **nest**:

| Region | Composition |
|---|---|
| **WT** — whole tumour | labels 1 ∪ 2 ∪ 4 |
| **TC** — tumour core | labels 1 ∪ 4 |
| **ET** — enhancing tumour | label 4 |

Because WT ⊃ TC ⊃ ET, the model predicts three **independent sigmoids**, not a softmax. A softmax
would force the regions to compete for the same voxel, which is the wrong relationship.

Unless stated otherwise, a bare Dice figure anywhere in this project is **WT**.

---

## 2. On-disk format, and the two layers of packaging

The data ships double-packaged:

1. **Outer archive** — Kaggle serves one `.zip` containing a `.tar`. Extracting it unpacks the
   folder structure; the files inside stay compressed.
2. **Inner compression** — each volume is a `.nii.gz`. Decompressing to raw `.nii` roughly doubles
   read speed during cache building.

| Step | Input | Output | Size |
|---|---|---|---|
| Extract | 1 `.zip` | 6,255 `.nii.gz` (still compressed) | ~13 GB → ~13 GB |
| Unzip | 6,255 `.nii.gz` | 6,255 raw `.nii` | ~13 GB → **~114 GB** |

The ~9× expansion is because ~99% of each volume is background zeros plus smooth tissue, which
gzip compresses heavily. Per volume uncompressed: 240·240·155·2 bytes ≈ **17.9 MB**; per case
(5 volumes) ≈ **90 MB**.

`load_case` reads either form, so both layouts work.

### Where it lives

| Location | Form | Size | Purpose |
|---|---|---|---|
| `data/BraTS2021_Training_Data/` | `.nii.gz` | 13 GB | original source, in-repo |
| `D:/data/unzipped/` | raw `.nii` | 114 GB | fast local access |
| `Drive/MyDrive/capstone/unzipped/` | raw `.nii` | 114 GB | training data for Colab |

Override the location with `FEDBRATS_DATA_ROOT`. The raw data is **not** in git.

**Source:** Kaggle `dschettler8845/brats-2021-task1`.

**Acquisition on Colab:** run [`../colab_setup.ipynb`](../colab_setup.ipynb) top to bottom. It
mounts Drive, fetches the archive, extracts on local disk, then **stream-unzips in batches of 100**
— unzip a batch, move it to Drive, delete the local copy, repeat — so Colab's ~100 GB disk never
fills. Resumable: finished cases are skipped on re-run.

---

## 3. Partition — 1,251 cases into four hospitals

We **partition first, then split each hospital into train/test**, so every hospital owns both a
training set and a test set drawn from its own shifted distribution. That is required for the
per-site H2 and H3 claims: without it there would be no way to ask how a site's own model performs
on its own patients.

| | Site A | Site B | Site C | **Site D** (outlier) | Total |
|---|---|---|---|---|---|
| assigned | 313 | 313 | 313 | 312 | **1,251** |
| → train pool | 251 | 251 | 251 | 250 | **1,003** |
| → test | 62 | 62 | 62 | 62 | **248** |

- **Deterministic.** Sort case IDs → seeded shuffle → assign → split.
- **Committed manifest.** The assignment is written once to `artifacts/splits/partition.json`
  (`case_id → {hospital, split, is_outlier}`) and **checked into git**. Every run in every
  iteration reads the identical split, so no result can be an accident of who got which patient.
- **`train_per_hospital`** caps how many of the ~250 training cases a run actually uses — 150 in
  v1/v2, **230** from v3 onward. Test sets are fixed regardless.
- **`val_per_hospital`** takes 20 cases per site for validation from v2 onward, drawn from the
  cases *after* the training cap, so enabling validation does not change the training set.
- **Centralized** trains on the union of the four training sets.

---

## 4. The scanner shift — the non-IID source

Real hospitals differ because their machines differ. Each simulated site applies a fixed transform
emulating its scanner, implemented in [`src/fedbrats/shift.py`](../src/fedbrats/shift.py).

| Ingredient | Effect |
|---|---|
| **Gamma** | nonlinear contrast, `x^γ` over the brain intensity range |
| **Bias field** | smooth low-frequency multiplicative gradient across the volume |
| **Gaussian blur** | mild resolution / point-spread difference |

| Site | key | gamma | bias amp | blur σ |
|---|---|---|---|---|
| Site A | `H1` | 1.06 | 0.06 | 0.3 |
| Site B | `H2` | 1.13 | 0.09 | 0.5 |
| Site C | `H3` | 1.20 | 0.12 | 0.7 |
| **Site D** | `H4` | **1.85** | **0.34** | **1.7** |

Sites A–C are mild and same-direction, so they cluster; Site D is deliberately far from them to
drive H2 and H3. The shift is applied to a site's **train and test cases alike**, and the bias
field is fixed per `(hospital, seed)` — a hospital's scanner does not change between patients.

### Why it must be nonlinear and spatial

Preprocessing z-normalizes every volume per case. A purely **linear** intensity change would be
normalized straight back out, leaving no heterogeneity to study at all — a bug we hit and fixed.
Gamma, a spatial bias field and blur all survive normalization.

Verified on a real case: after z-normalization the sites still differ (pairwise mean absolute
difference 0.095–0.388 σ), Site D is the clear outlier (**margin +0.149 σ** over the most distant
typical site), and a purely linear shift washes out to **0.000**.

### ⚠ The blur-halo trap

The shift's Gaussian blur smears brain intensity into BraTS's exactly-zero skull-stripped
background. If the brain mask is derived *after* the shift, it grows with the site's blur σ:

| | Site A (σ=0.3) | Site B (σ=0.5) | Site C (σ=0.7) | **Site D (σ=1.7)** |
|---|---|---|---|---|
| leaked background voxels | +112,726 | +226,623 | +342,450 | **+833,379 (+57%)** |

Three consequences, the last of which invalidates the whole experiment:

1. The crop bounding box becomes site-dependent, so volumes get **different shapes**.
2. Z-normalization statistics are computed over the halo.
3. **Hospital identity leaks as geometry** — a model could read the site off the tensor
   dimensions, and a FedBN "recovery" of Site D would be an artefact rather than adaptation.

**The fix:** take the mask and bounding box from the **unshifted** volume, then apply the shift and
re-mask with it. All four sites then share one shape. This is why steps 2 and 4 of the
preprocessing chain exist and why their order is load-bearing.

---

## 5. Preprocessing — seven steps, in a load-bearing order

```
load → brain mask/bbox (UNSHIFTED) → shift → re-mask → crop → z-normalise → clip → regions
```

| # | Step | Detail |
|---|---|---|
| 1 | **Load** | 4 modalities + segmentation, as float32 |
| 2 | **Mask & bbox** | brain mask from the **unshifted** volume — see the trap above |
| 3 | **Scanner shift** | the site's gamma, bias field and blur |
| 4 | **Re-mask** | kill the blur halo the shift created |
| 5 | **Crop** | to the brain bounding box; ~99% background is wasted compute |
| 6 | **Z-normalise** | per case, per modality, over brain voxels |
| 7 | **Clip & label** | ±5σ, then labels `{0,1,2,4}` → WT / TC / ET masks |

Clipping at ±5σ tames outliers: a gamma-shift outlier once produced NaNs at ~14σ.

---

## 6. Sampling — the one place `dim` changes the data

| | **2D** (`dim=2d`) | **3D** (`dim=3d`) |
|---|---|---|
| Unit | axial slice, cropped to 192×192 | cubic patch, 96³ |
| Per case per epoch | **8 slices** | **2 patches** |
| Batch size | 8 | 1 |
| Evaluation | whole slice stack | sliding window, 0.25 overlap |

**Tumour bias is a mix, not a filter.** With probability `tumor_frac = 0.7` a draw is forced to
contain tumour. Uniform sampling would spend most of the budget on empty brain, where the Dice
gradient is nearly flat — but training on tumour-bearing slices *only* makes the model hallucinate
tumour on the empty slices it meets during full-volume evaluation.

The 3D backbone gets far fewer samples per epoch, so each 3D epoch is a much noisier estimate.
That single fact explains most of where 2D and 3D results diverge.

---

## 7. The cache

Preprocessing a case takes seconds; a training run touches every case dozens of times. So the
pipeline output is written once to disk as fp16 tensors.

- **Volumes are cached, not pre-sampled units** — so one cache serves *both* backbones, and
  sampling happens at load time from the memory-mapped volume.
- **Keyed by content.** The cache directory is named by a hash of the shift parameters, the clip
  threshold and the seed. Change the scanner model and you get a **new cache**, not a silently
  stale one.
- **Resumable.** A case with a `meta.json` is skipped, so a killed build resumes.
- **Also cached: the tumour bounding box**, so tumour-biased sampling does not rescan the volume
  on every draw.

```
<cache>/<key>/<case_id>/{x.npy, y.npy, meta.json}   +   index.json
   x = (4, X, Y, Z) float16      y = (3, X, Y, Z) uint8
```

**Measured ~35 MB/case** → ~44 GB for one full cache variant. This machine holds ~105 GB across
three variants (different shift settings and backbones). The cache is git-ignored and fully
regenerable:

```bash
uv run python scripts/build_cache.py --workers 8
```

It must not land on a WSL VHDX backed by a nearly-full C: — point `FEDBRATS_CACHE_DIR` at a
drive with room.

---

## 8. What this stage produces

| Output | Path | Tracked |
|---|---|---|
| Split manifest | `artifacts/splits/partition.json` | **committed** |
| Preprocessed cache | `artifacts/cache/<key>/` | git-ignored, regenerable |

Both are consumed by the FL engine — see [training.md](training.md).
