# 03 · Architecture — structure, modules, data flow

## 3.1 Repository layout

```
capstone/
├── src/braintumor_fl/          # the library (installable package, src-layout)
│   ├── __init__.py             # version + constants: MODALITIES, REGIONS
│   ├── data.py                 # BraTS 2D slice pipeline + splits + synthetic shift
│   ├── model.py                # MONAI 2D U-Net (BratsUNet) + Dice loss/metric
│   ├── trainer.py              # shared train/eval loop (AMP, NaN-skip, grad-clip, FedProx)
│   ├── personalization.py      # which params a client keeps local (FedBN / personal-head)
│   ├── partition.py            # cases → hospitals (even OR FeTS) + no-leak splits + site map
│   ├── results.py              # uniform per-hospital result JSON schema (read/write)
│   └── train_centralized.py    # the centralized ceiling run (also Phase-1 pipeline check)
├── fl/                         # runnable scripts (entry points)
│   ├── brats_client.py         # a FLARE client = one hospital (the per-round loop)
│   ├── run_fedavg.py           # defines + launches the FLARE job (all FL methods)
│   ├── run_local_baselines.py  # local-only floor (one model per hospital)
│   ├── finetune.py             # fine-tune the FedAvg global per hospital
│   └── evaluate.py             # score one checkpoint on every hospital's val set
├── analyze.py                  # aggregate results/ → CSV tables + PNG figures
├── run_all.sh                  # the 8-step pipeline; one command runs everything
├── data/                       # BraTS cases + checkpoints (gitignored)
├── results/                    # per-method result JSONs + aggregated CSV/PNG
├── docs/                       # this documentation
├── pyproject.toml              # deps; uv project; Python 3.12; torch cu124
└── {README,PLAN,DATA,WSL,CLAUDE}.md   # older top-level notes (docs/ supersedes)
```

**src-layout.** The library lives under `src/braintumor_fl/` and is imported as
`braintumor_fl`. The `fl/` scripts and `train_centralized` are thin CLIs around it,
so there is exactly one implementation of every piece of logic.

## 3.2 Module reference (what each file owns)

### `__init__.py` — constants
- `MODALITIES = ("flair", "t1", "t1ce", "t2")` — the fixed 4-channel input order.
  Every image tensor is these four sequences stacked, always in this order.
- `REGIONS = ("TC", "WT", "ET")` — the BraTS eval regions and the output-channel
  order. This ordering is the contract between the model, the metric, and the
  result schema; changing it silently would misalign every score.

### `data.py` — the 2D data pipeline *(see [Data pipeline](04-data-pipeline.md) for depth)*
- **Case discovery / slice index:** `find_cases`, `build_slice_index` (indexes only
  the axial slices that actually contain tumor, cached to CSV).
- **Dataset:** `BratsSliceDataset` — lazily reads one 2D slice per item from the 3D
  `.nii.gz` via `nibabel`'s memory-mapped `dataobj`; optionally applies a per-hospital
  scanner shift.
- **Transforms:** `train_transforms` / `eval_transforms` (MONAI dictionary
  transforms: BraTS-label→region conversion, nonzero channel-wise z-normalization,
  resize to 192², and — train only — flips/rotations/intensity jitter).
- **Splits:** `case_split` (the single source of truth: split by *patient*, never by
  slice), `loaders_for_cases`, `loaders_from_case_lists`, `split_by_case`.
- **Synthetic shift:** `SITE_PROFILES`, `site_shift_params`, `apply_site_shift`,
  `SiteShift` — the deterministic per-hospital scanner heterogeneity.

### `model.py` — the network, loss, metric
- `build_unet(...)` — a **MONAI 2D U-Net**: `spatial_dims=2`, in=4, out=3,
  `channels=(16,32,64,128,256)`, `strides=(2,2,2,2)`, `num_res_units=2`,
  `norm="batch"`. Sized to fit 4 GB at 192².
- `BratsUNet` — a thin `nn.Module` **wrapper** around `build_unet` with **all-default
  constructor args**. This is the class you instantiate everywhere (see invariant #1).
- `build_loss()` = `DiceLoss(sigmoid=True)` (multi-label). `build_metric()` =
  `DiceMetric(include_background=True, reduction="mean_batch")`. `logits_to_preds` =
  `sigmoid ≥ 0.5`.

### `trainer.py` — the one training/eval code path
- `train_one_epoch` — a single pass with **AMP autocast**, **NaN-skip** (a batch
  whose loss isn't finite is skipped, never corrupting weights), **grad clipping**
  (max-norm 1.0), and the optional **FedProx** proximal term.
- `local_train` — run several epochs; this is what a FLARE client does each round.
- `evaluate` — per-region + mean Dice over a loader (no-grad, eval mode).
- Used by **both** the centralized run and the federated client, so training is
  provably identical across methods.

### `personalization.py` — the research contribution, mechanically
- `norm_param_names(model)` — state-dict keys of all normalization layers (params
  **and** running buffers), detected by module *type* (robust to naming).
- `head_param_names(model)` — keys of the final conv that emits `out_channels`.
- `keep_local_keys(model, strategy)` — the dispatch: `fedavg` → `{}`, `fedbn` →
  norm keys, `personal_head` → norm ∪ head keys.
- `load_global(model, global_params, keep_local)` — load the incoming global weights
  **except** the `keep_local` keys, which retain their local values. Robust to a
  global payload that omits some keys.

### `partition.py` — cases → hospitals
- `even_partition` — round-robin split into N clients (roughly IID).
- `fets_partition` — group by real FeTS institution from a CSV (`Subject_ID →
  Partition_ID`), dropping institutions with `< min_cases`.
- `get_partitions` / `client_cases` — the unified entry points (FeTS if a CSV is
  given, else even split).
- `partitioned_splits` — per-hospital case-level train/val splits, aggregated for
  the centralized run so it trains on the **union of hospital train-cases** and
  never touches any hospital's val cases.
- `case_site_map` — flatten partitions into `{case_dir → site index}`, the driver
  for the synthetic scanner shift (deterministic, so every method maps a case to the
  same site).

### `results.py` — the uniform result schema
- `write_scores(...)` writes one JSON per `(method, site)` at
  `results/<method>/<site>.json` with keys: `method, site, n_train, n_val,
  dice_mean, dice_TC, dice_WT, dice_ET`.
- `read_all(results_dir)` loads them all for `analyze.py`.

### `train_centralized.py` — the ceiling / Phase-1 check
- Builds partition-consistent train/val loaders (union of hospital train-cases),
  trains `BratsUNet` with cosine-decayed Adam, saves the best-val checkpoint.

### `fl/` scripts
- `brats_client.py` — the per-hospital FLARE client loop (see §3.4).
- `run_fedavg.py` — defines the FLARE `FedJob` (server = FedAvg, one ScriptRunner
  per client) and runs it in the Simulator. One script covers FedAvg/FedProx/FedBN/
  personal-head via flags.
- `run_local_baselines.py` — trains an independent model per hospital (the floor).
- `finetune.py` — loads the converged FedAvg global and fine-tunes per hospital.
- `evaluate.py` — scores one checkpoint on every hospital's held-out val set (used
  for the centralized ceiling's per-hospital numbers).

### `analyze.py` — the headline
- Reads all `results/*/*.json`, pivots to a per-hospital × method table, computes
  fairness summaries, and renders `comparison.png` and `personalization_gain.png`.

## 3.3 The 8-step pipeline (`run_all.sh`)

`run_all.sh` runs the whole experiment matrix in order. Each step writes result
JSONs that step 8 aggregates.

| # | Step | Script | Produces |
|---|------|--------|----------|
| 1 | **Centralized ceiling** — train on pooled hospital train-cases | `train_centralized` | `data/centralized_unet.pt` |
| 2 | **Centralized per-hospital eval** | `fl/evaluate.py` | `results/centralized/site-*.json` |
| 3 | **Local-only floor** — one model per hospital | `fl/run_local_baselines.py` | `results/local/site-*.json` |
| 4 | **FedAvg** | `fl/run_fedavg.py --method fedavg` | `results/fedavg/site-*.json` |
| 5 | **FedProx** | `... --method fedprox --prox-mu 0.01` | `results/fedprox/site-*.json` |
| 6 | **FedBN** *(primary)* | `... --method fedbn --personalization fedbn` | `results/fedbn/site-*.json` |
| 7 | **Fine-tune** — from the FedAvg global | `fl/finetune.py` | `results/finetune/site-*.json` |
| 8 | **Analyze** | `analyze.py` | `per_hospital.csv`, `summary.csv`, `*.png` |

Every step is parameterized by the **same** split, case cap, and synthetic-shift
flag, so all methods see identical hospitals, identical val cases, and identical
per-hospital scanner shifts — the comparison is apples-to-apples by construction.
See [Running](06-running.md) for the env knobs.

## 3.4 End-to-end data flow

**Shared front-end (every method):**

```
BraTS case dirs (.nii.gz)
   └─ find_cases → get_partitions → per-hospital case lists   (partition.py)
        └─ case_split (by patient, seed 42) → train/val cases  (data.py)
             └─ build_slice_index (tumor slices only, cached)  (data.py)
                  └─ BratsSliceDataset[i]:
                       read 4 modality slices + seg slice (lazy, nibabel)
                       → [optional] apply_site_shift(image, site)   (data.py)
                       → MONAI transforms: label→regions, z-norm, resize, (aug)
                       → {"image": (4,192,192), "label": (3,192,192)}
```

**Federated round (FedAvg/FedProx/FedBN/personal-head), per client:**

```
server (plain FedAvg)  ── global weights ──▶  brats_client.py
                                               1. load_global(keep_local)         ← personalization
                                               2. restore {site}_local.pt         ← FedBN persistence
                                               3. evaluate BEFORE training  ──▶ results/<method>/<site>.json
                                               4. local_train (this hospital)
                                               5. save {site}_local.pt
server  ◀── updated weights ──                 6. send weights back
   └─ average → next round's global weights
```

**Aggregation:** every method's `results/<method>/site-*.json` → `analyze.py` →
tables + the personalization-gain figure.

## 3.5 Design invariants — do not break these

These are load-bearing. Breaking one silently corrupts the experiment (wrong
numbers, data leakage, or a method quietly degenerating into another).

1. **Instantiate `BratsUNet`, never raw `build_unet`, in every script.** FLARE
   rebuilds the server model from a JSON config by introspecting constructor args;
   MONAI's `UNet` has a required `spatial_dims` that FLARE drops → server crash.
   `BratsUNet` has all-default args, so it reconstructs cleanly. Its state-dict keys
   are all `net.*` — keep them consistent everywhere or checkpoints won't cross-load.
2. **No data leakage.** `data.case_split` is the *one* source of truth for train/val,
   split by **patient** (never by slice). `partition.partitioned_splits` guarantees
   the centralized model never trains on any hospital's val cases. Every method
   evaluates on the *same* per-hospital val set.
3. **Eval-before-train protocol.** In `brats_client.py`, the client evaluates the
   received model **before** local training each round → FedAvg reports the true
   global model and FedBN the genuine personalized one. Do not move eval after
   training.
4. **Personalization is client-side.** The server always runs plain FedAvg;
   personalization = which params the client **keeps local** when the global model
   arrives (`personalization.keep_local_keys`).
5. **FedBN local params persist across rounds.** `brats_client.py` saves
   `{site}_local.pt` after training and restores it before eval, so BatchNorm
   accumulates even if FLARE re-runs the client script each round. Remove this and
   FedBN silently collapses to FedAvg.
6. **Training stability.** Cosine LR decay + grad-clip (1.0) + NaN-skip; default LR
   `5e-4`. (Earlier NaN blow-ups occurred at `lr=1e-3`.)
7. **The synthetic shift must be nonlinear/spatial, and applied identically to every
   method.** A plain affine intensity shift is erased by the pipeline's channel-wise
   z-normalization; only gamma/bias-field/blur survive. And it must reach centralized,
   local, all FL methods, and fine-tune via the same `case→site` map, or the
   comparison breaks. See [Data pipeline](04-data-pipeline.md).

Each invariant's *why* and the failure it prevents is in
[Design decisions](08-design-decisions.md).
