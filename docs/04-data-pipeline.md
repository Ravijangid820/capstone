# 04 · Data pipeline — loading, splits, partitioning, the synthetic shift

Everything data-related lives in `src/braintumor_fl/data.py` and
`src/braintumor_fl/partition.py`. This chapter walks the full path from raw
`.nii.gz` volumes to the batched tensors each method trains on, then details the two
pieces most specific to this project: **no-leakage splitting** and the **synthetic
scanner shift**.

## 4.1 Raw data format

BraTS 2021 ships one folder per subject:

```
BraTS2021_00001/
├── BraTS2021_00001_flair.nii.gz   # FLAIR
├── BraTS2021_00001_t1.nii.gz      # T1
├── BraTS2021_00001_t1ce.nii.gz    # T1 contrast-enhanced
├── BraTS2021_00001_t2.nii.gz      # T2
└── BraTS2021_00001_seg.nii.gz     # labels {0,1,2,4}
```

All volumes are 3D, 240×240×155, co-registered and skull-stripped (background is
exactly 0). The four sequences become the model's **4 input channels** (fixed order
`MODALITIES`); the seg volume becomes the **3 output regions** (WT/TC/ET). Data is
gitignored and ~15 GB; see `DATA.md` for how to obtain it.

## 4.2 On-the-fly 2D slicing (no pre-slicing to disk)

We never write 2D slices to disk (that would be tens of GB and slow on WSL's `/mnt/c`
filesystem). Instead:

- **`build_slice_index(cases, min_tumor_pixels=100, cache_csv=...)`** reads only the
  small `_seg.nii.gz` volumes and records every axial slice `(case_dir, z)` whose
  tumor area ≥ `min_tumor_pixels`. Empty/near-empty slices are skipped — they'd waste
  compute and bias the Dice. The result is cached to a CSV keyed by case dir and is
  **incremental**: growing the case set (5 → 100 → 1251) never rescans work already
  done. (Caveat: the cache assumes a fixed `min_tumor_pixels`; change it → delete the
  cache.)
- **`BratsSliceDataset[i]`** does a **lazy** per-slice read: `nibabel`'s memory-mapped
  `dataobj[..., z]` pulls exactly slice `z` off disk for each of the 4 modalities plus
  the seg. It stacks the modalities into a `(4, H, W)` float32 image and returns
  `{"image", "label"}`. This keeps RAM flat and disk I/O proportional to slices
  actually used.

At full scale there are ~60 tumor slices per case → ~75k slices across 1251 cases.
Per-slice reads + transforms on the CPU are the pipeline's bottleneck — see
[Running §Performance](06-running.md#performance-model).

## 4.3 Transforms (MONAI dictionary transforms)

Applied inside the dataset after loading (and after the optional site shift, §4.6).

**`train_transforms(size=192)`** — in order:
1. `ConvertToMultiChannelBasedOnBratsClassesd` — turn raw labels `{0,1,2,4}` into the
   3 overlapping region channels (TC, WT, ET). The label is kept *without* a channel
   dim on input because this transform builds the channels itself.
2. `NormalizeIntensityd(nonzero=True, channel_wise=True)` — **per-channel z-normalization
   over the nonzero (brain) voxels**. Background stays 0; each modality is scaled to
   ~zero-mean/unit-variance over the brain. *This is the transform that erases any
   plain affine intensity difference — the crux of the synthetic-shift design (§4.6).*
3. `Resized(192, 192)` — bilinear for image, nearest for label.
4. `RandFlipd` (both axes), `RandRotate90d`, `RandScaleIntensityd`, `RandShiftIntensityd`
   — light augmentation (train only).
5. `ToTensord`.

**`eval_transforms(size=192)`** — same but **without** the random augmentation
(steps 4), so validation is deterministic.

Output per item: `image (4,192,192)`, `label (3,192,192)`.

## 4.4 Splitting — the single source of truth, no leakage

**`case_split(cases, val_frac=0.2, seed=42)`** deterministically splits a case list
into `(train_cases, val_cases)` **at the case (patient) level**, never by slice. This
is *the* splitting primitive; every method routes through it, so:

- Slices from one patient never straddle the train/val boundary (no slice-level
  leakage — a huge, easy-to-miss bug in medical imaging).
- Given the same `seed`, every method (local, FL, centralized, fine-tune, eval) sees
  the **identical** per-hospital validation set → the comparison is fair.

`split_by_case` applies the same logic to an already-built slice index. Loaders:
`loaders_for_cases` (one hospital → train/val loaders) and `loaders_from_case_lists`
(explicit lists, used by the centralized union).

## 4.5 Partitioning — cases → hospitals (`partition.py`)

Two modes, both **deterministic** so every script independently reconstructs the same
hospitals:

- **`even_partition(cases, n_clients)`** — round-robin `cases[i::n_clients]`. Roughly
  IID. Good for validating the plumbing; **not** a real heterogeneity source.
- **`fets_partition(data_root, csv_path)`** — group cases by real FeTS institution
  from a CSV (`Subject_ID → Partition_ID`), dropping institutions with `< min_cases`
  (default 5) so every hospital has enough to train + validate. This is the genuine
  non-IID setting the method targets (not yet obtained — see [Roadmap](09-roadmap.md)).

`get_partitions(...)` picks FeTS if a CSV is given, else the even split.
`partitioned_splits(partitions)` returns:
- `all_train` — union of every hospital's train cases (the centralized training set),
- `all_val` — union of every hospital's val cases (held out from centralized training),
- `per_site` — `[(site_name, train_cases, val_cases), ...]`.

This is what guarantees the **centralized ceiling never trains on any hospital's val
cases** (invariant #2).

`case_site_map(partitions)` flattens to `{case_dir → 0-based site index}` — the driver
for the synthetic shift (§4.6). Because partitions are deterministic, a given case maps
to the same site in every script.

## 4.6 The synthetic scanner shift

> **Purpose:** manufacture the per-hospital heterogeneity that FedBN needs, when the
> real FeTS institutional split isn't available. Without heterogeneity, FedBN ≈ FedAvg
> by construction and there is no result (see [Overview §1.6](01-overview.md#17-why-we-manufacture-heterogeneity-the-honest-caveat)).

### The one non-obvious constraint: it must survive z-normalization

The pipeline z-normalizes every channel over nonzero voxels (§4.3, step 2). A **plain
affine** intensity change `a·x + b` is almost perfectly **erased** by that
normalization — subtract the (shifted) mean, divide by the (shifted) std, and you're
back where you started. So an affine "scanner shift" would leave the model, and its
BatchNorm statistics, essentially unchanged — nothing for FedBN to specialize on.

Therefore the shift must be **nonlinear** (changes the *shape* of the intensity
histogram) or **spatial** (changes local statistics), so that a real distributional
difference survives normalization and lands in the BatchNorm running stats. This was
verified empirically:

| Shift applied to a real slice | mean\|Δ\| after z-norm |
|-------------------------------|------------------------|
| affine `2x + 3` (control) | **0.0000** — fully erased |
| gamma-only sites | 0.023–0.026 |
| gamma + bias-field + blur sites | up to 0.066 |

Nonlinear/spatial shifts survive; the affine control does not. This is *the* reason
the design looks the way it does.

### The scanner profiles

Each hospital gets a deterministic **scanner profile** — a fixed combination of three
physically-motivated operations:

- **gamma** — a nonlinear intensity remap `x^γ` on the [0,1]-rescaled brain. Changes
  the histogram *shape* (contrast curve). Survives z-norm.
- **bias field** — a smooth low-frequency multiplicative field (a tilted plane + a
  radial "bowl"), modeling MRI coil/scanner intensity inhomogeneity. Spatially
  varying → survives z-norm.
- **blur** — a Gaussian blur (`scipy.ndimage.gaussian_filter`), modeling a lower-
  resolution scanner / different point-spread function. Changes local texture.

`SITE_PROFILES` (in `data.py`) is a fixed table; **site 0 is the reference scanner
(identity — no shift)** so there's always a canonical hospital, and later sites drift
progressively:

| Site (0-based) | γ | bias | blur | Character |
|---|---|---|---|---|
| 0 | 1.0 | 0.0 | 0.0 | reference scanner (no-op) |
| 1 | 1.5 | 0.0 | 0.0 | darker mid-tones |
| 2 | 0.65 | 0.0 | 0.0 | brighter mid-tones |
| 3 | 1.0 | 0.0 | 1.1 | low-resolution scanner |
| 4 | 1.35 | 0.45 | 0.6 | field inhomogeneity + contrast |
| 5 | 0.75 | 0.35 | 0.4 | opposite contrast + field |
| 6 | 1.7 | 0.5 | 0.9 | strong outlier |
| 7 | 0.55 | 0.4 | 0.7 | strong outlier (other direction) |

`site_shift_params(site)` indexes this table (cycling if there are more sites than
profiles). With the default `--n-clients 6`, sites 0–5 are used: **site-1 matches the
pooled distribution** (FedAvg does fine there) while **site-4/5/6 are progressively
bigger outliers** — exactly the gradient H2/H3 need.

### How it's applied

- **`apply_site_shift(image, site)`** operates per channel on the **brain (nonzero)
  voxels only** (background stays 0 so the nonzero-normalization mask is preserved):
  robust-percentile rescale to [0,1] → gamma → optional bias-field multiply → map back
  → optional Gaussian blur → re-zero background. Deterministic; identity for site 0.
- **`SiteShift`** wraps a `case_site` map and, given a `(image, case_dir)`, looks up
  the case's site and applies its shift. It's passed into `BratsSliceDataset`, which
  applies it **before** the MONAI transforms (so the shift is on raw intensities and
  the z-normalization then acts on the shifted distribution).

### Why it's correct across methods

The shift is a pure function of `site`, and `case → site` is deterministic (from the
same partitions everywhere). So a given case gets the **identical** shift in the
centralized run, the local baseline, every FL client, the fine-tune, and the eval.
The `--synthetic-shift` flag is threaded through all of them (and forwarded to FLARE
clients by `run_fedavg.py`), so no method is accidentally comparing shifted vs
unshifted data. `run_all.sh` turns it on by default for even splits and **off for
FeTS** (real FeTS is already non-IID — stacking a synthetic shift on top would
confound it).

### Verified properties

1. **Site 0 = exact identity** (`max|Δ| = 0`).
2. **Deterministic** (same input+site → identical output).
3. **Survives z-normalization** (0.023–0.066 while the affine control → 0.0000).
4. `case_site_map` matches the partition site indices.
