# Code

A working engineer's guide to this repository: how to run it, what every module does, how one
training run flows through the code, and what changed in the code across v1→v5.

**Related:** [data.md](data.md) for the dataset · [training.md](training.md) for the science and
the results · [conventions.md](conventions.md) for naming and reporting rules.

---

## 1. Running it

### Setup

```bash
uv sync                                  # dependencies
uv run pytest                            # smoke tests (metrics, partition, model, data)
```

`nvflare` is an optional extra with a platform marker, so `uv sync` never installs it on Windows.
Use `uv sync --extra flare` on Linux only.

Two environment variables override where things live, so no path is hardcoded:

| Variable | Meaning |
|---|---|
| `FEDBRATS_DATA_ROOT` | where the raw BraTS cases are |
| `FEDBRATS_CACHE_DIR` | where the preprocessed cache goes (needs ~44 GB per variant) |

### The pipeline, in order

```bash
# 1. the deterministic 4-hospital split -> artifacts/splits/partition.json (committed)
uv run python scripts/build_partition.py

# 2. preprocess once (~2.3 s/case with 4 workers, ~33 min, ~44 GB)
uv run python scripts/build_cache.py --workers 8

# 3. train — one method
uv run python scripts/run_experiment.py --method fedbn --dim 2d --preset v5 --tag v5

# 3b. or all four, resumable across sessions
uv run python scripts/run_matrix.py --dim 2d --seed 42 --preset v5 --tag v5

# 4. freeze the results with a hash manifest
uv run python scripts/freeze_baseline.py --runs-dir artifacts/runs/v5 --dest artifacts/snapshots/v5
```

### The demo and the walkthrough

```bash
uv run python scripts/demo_server.py                # / = live inference, /showcase = walkthrough
uv run python scripts/demo_server.py --port 8010    # if 8000 is taken (VS Code often holds it)
uv run python scripts/demo_server.py --runs artifacts/runs   # serve the v1 checkpoints instead
```

The demo serves the **v5** checkpoints by default when they exist, so the models it runs are the
ones every reported table describes. `/api/health` names which set is loaded.

### Analysis

```bash
uv run python scripts/check_docs.py                 # re-derive 16 figures from the logs
uv run python scripts/compare_runs.py --baseline artifacts/snapshots/v1 \
    --new artifacts/snapshots/v5 --dim 2d --seed 42
uv run python scripts/compute_significance.py --runs artifacts/snapshots/v5 \
    --dim 2d --a fedavg --b fedbn
uv run python scripts/plot_results.py --dim 2d --tag v5
```

### Environments

| Env | Role | Notes |
|---|---|---|
| **Windows native — RTX 3050 (4 GB)** | everything in this study ran here | fast `D:`; no FLARE (POSIX-only `resource`) |
| **WSL2** | dev, smoke runs | matches Colab (`fork`); `/mnt/d` is slow |
| **Colab — T4 (16 GB)** | heavier sweeps | data in Drive, cache on local disk per session |

Windows uses `spawn`, so every `scripts/` entry point needs an `if __name__ == "__main__":` guard,
and `num_workers` defaults to **0** on Windows (spawn re-imports per worker, which is slower than
single-threaded loading here).

---

## 2. Repository map

```
src/fedbrats/          the library — everything the science depends on
  config.py            every knob, the presets, path resolution          408 lines
  partition.py         cases → hospitals → train/test, writes manifest   119
  shift.py             the per-hospital synthetic scanner shift           76
  data.py              preprocess · cache · augment · samplers           454
  model.py             the U-Net, and bn_keys()                           65
  metrics.py           per-volume Dice, BraTS empty-GT convention         43
  train.py             one local training loop + full-volume inference   218
  federated.py         THE round loop — all four methods                 288
  logging_utils.py     logger, JSONL writer, run-dir guard                82
  static/              the demo front end (vanilla JS + Three.js)

scripts/               entry points and tooling
tests/                 pytest smoke suite
artifacts/             runs, snapshots, cache, figures, showcase (mostly git-ignored)
```

**Read them in this order:** `config.py` → `federated.py` → `train.py` → `data.py`. The other four
are small and self-explanatory once those make sense.

---

## 3. The modules

### `config.py` — one dataclass drives everything

A single `Config` is built once and passed everywhere. Every knob is written to the run's
`config.json`, so a run is described by its own output rather than by whatever command someone
remembers typing.

Key members beyond the plain fields:

| Member | What it does |
|---|---|
| `__post_init__` | validates, then fills dimension-dependent defaults (`base_channels` 32/16, `batch_size` 8/1) and **forces `aug_rot90_p=0` in 3D** |
| `run_dir(method)` | `artifacts/runs/[<tag>/]<method>_<dim>_<seed>` — `--tag` is what stops a rerun colliding with an archived result |
| `round_lr(rnd)` | the cosine schedule **across rounds**, decaying `lr` → `lr * lr_min_factor` over `lr_anneal_rounds` (or the whole run) |
| `scores_test(rnd)` | whether this round gets a full test evaluation — always true for the final `report_last_k` rounds, so thinning never changes the headline |
| `to_dict()` | typed snapshot for `config.json`; typed, not stringified, so `compare_runs.py` can diff `25` from `"25"` |
| `PRESETS` / `apply_preset` | v1–v5 as named recipes |

**The single most important design decision in this file:** every default reproduces the **v1
baseline**, so an unflagged run today still yields the frozen numbers bit for bit. Improvements are
opt-in through `--preset`. That is what keeps "before" reproducible after "after" exists.

### `partition.py` — the split

`build_partition` sorts case IDs, shuffles with the global seed, assigns to four hospitals, then
splits each into train/test. `save_manifest` writes `artifacts/splits/partition.json`, which is
**committed**. Everything downstream reads it, so no run can drift onto a different split.

### `shift.py` — the non-IID source

`HOSPITAL_SHIFTS` is the parameter table; `apply_shift(volume, hospital, seed)` applies gamma →
bias field → blur to the 4 modalities only, never the segmentation. `_bias_field` builds a smooth
field from 4³ control points upsampled cubically, seeded per `(hospital, seed)` so a hospital's
scanner is fixed. See [data.md §4](data.md#4-the-scanner-shift--the-non-iid-source).

### `data.py` — the biggest module, and the subtlest

| Function | Role |
|---|---|
| `load_case` | reads `.nii` or `.nii.gz` → `(4,X,Y,Z)` float32 + seg |
| `brain_bbox`, `labels_to_regions` | mask geometry; labels `{0,1,2,4}` → WT/TC/ET |
| `preprocess` | **the seven-step chain** — mask from the *unshifted* volume, shift, re-mask, crop, z-norm, clip |
| `cache_key` | md5 of shift params + clip + seed → a new cache when the recipe changes, never a stale one |
| `build_case_cache`, `assemble_index` | resumable parallel cache build; skips cases with `meta.json` |
| `_augment_spatial` | flips (+ rot90 in 2D) — applied to image **and** mask together |
| `_augment_intensity` | scale/shift/noise in z-score units; **restores exact-zero background** afterwards |
| `SliceDataset` / `PatchDataset` | tumour-biased 2D slices / foreground-biased 3D patches |
| `train_val_cases` | validation drawn from cases *after* the training cap, so training is unchanged |

Two things here are load-bearing and easy to break:

1. **`preprocess` takes the brain mask from the unshifted volume.** Reversing those two lines
   re-introduces the blur-halo leak that makes volume *shape* correlate with hospital identity.
2. **`_augment_intensity` restores the exact-zero background.** A shift or noise draw that lifts
   the background off zero trains the model on inputs it never meets at evaluation.

### `model.py` — small, and one function matters

`BratsUNet` wraps MONAI's `UNet` with `dim`-dependent `spatial_dims`, `num_res_units=2`, BatchNorm.
2D: 1,607,562 parameters; 3D: 1,191,516.

`bn_keys(model)` is the function the whole study rests on. It walks `named_modules()` looking for
`nn.modules.batchnorm._BatchNorm` and returns their `state_dict` keys — **65 of the U-Net's 114**.
It cannot be done by name matching: MONAI emits keys like `net.model.0.conv.unit0.adn.N.bias` with
no `"bn"` substring anywhere.

### `metrics.py` — 43 lines, one convention

`dice_binary` implements the BraTS empty-ground-truth convention: if both prediction and truth are
empty the score is 1.0, if only one is, it is 0.0. `dice_regions` returns WT/TC/ET for one volume.
Dice is always computed **per volume**, never over a pooled voxel set — that is what makes every
downstream comparison paired.

### `train.py` — one local loop, and the whole inference path

| Function | Role |
|---|---|
| `DiceBCELoss` | Dice + BCE, three independent sigmoid channels |
| `train_epochs` | the local training loop; builds a **fresh Adam each round** (standard FL — optimizer state is not transmitted) |
| `_probs_2d` / `_probs_3d` | full-volume inference; 3D uses a sliding window at `sw_overlap` |
| `_tta_flips` | flip test-time augmentation (4× cost, final round only) |
| `remove_small_components`, `postprocess` | drop connected components below `postproc_min_voxels` |
| `evaluate_cases` | scores a list of cases, returning the mean **and every per-case value with its `case_id`** |

`evaluate_cases` returning per-case values is what makes paired statistics possible. It originally
returned only the mean, which is why `scripts/rescore.py` exists — to backfill the baseline's
per-case numbers from its checkpoints without retraining.

### `federated.py` — the round loop

The core of the project, and deliberately short. `Method` is a table of three booleans
(`aggregate`, `keep_bn_local`, `pooled`); `run()` is one loop that expresses all four methods.

| Piece | Role |
|---|---|
| `weighted_average` | averages states by sample count; **`int64` buffers are copied, never averaged** |
| `start_state(client)` | what a site starts the round with — global, or global + own BN under FedBN |
| `eval_state(hospital)` | what gets scored — the true federated model for that site |
| `snapshot` / `restore` | best-validation checkpoint selection |
| `score(...)` | writes both `metrics.jsonl` and `per_case.jsonl`, tagged with a `stage` field |

The `stage` field (`round` / `final` / `cross`) is what keeps ordinary per-round rows from being
mixed with TTA'd final rows and the cross-hospital matrix in later analysis.

### `logging_utils.py` — small but protective

`guard_run_dir` refuses to write into a populated run directory unless `--overwrite` is passed.
The writers append, so a second run into the same directory would stack duplicate rows behind the
first — invisible in the file, fatal to a paired test that then matches each case against itself.

---

## 4. Following one run through the code

```
scripts/run_experiment.py
  └─ apply_preset("v5")                       config.py    → Config
  └─ federated.run(cfg, "fedbn")              federated.py
       ├─ guard_run_dir()                     logging_utils.py
       ├─ load_index()                        data.py       → the cache index
       ├─ train_val_cases() per hospital      data.py
       ├─ build_model()                       model.py      → BratsUNet
       ├─ bn_keys(model)                      model.py      → the 65 keys FedBN protects
       └─ for rnd in 1..40:
            ├─ cfg.round_lr(rnd)              config.py     → cosine, floors at round 25
            ├─ for h in H1..H4:
            │    ├─ start_state(h)            federated.py  → global + own BN
            │    ├─ build_dataset()           data.py       → SliceDataset (tumour-biased)
            │    └─ train_epochs()            train.py      → fresh Adam, DiceBCELoss
            ├─ weighted_average(skip=bn_keys) federated.py  → aggregate
            ├─ score() on validation          train.py      → evaluate_cases
            └─ if cfg.scores_test(rnd):
                 score() on test              → metrics.jsonl + per_case.jsonl
       └─ best-val restore, final TTA scoring, summary.json
```

---

## 5. What changed in the code, v1 → v5

Every improvement is a **config field**, not a code path — that is why v1 is still reproducible.

| Iteration | Fields added or changed | Code that had to change |
|---|---|---|
| **v2** | `lr_schedule`, `lr_min_factor`, `augment`, `aug_*`, `tta`, `postproc_min_voxels`, `val_per_hospital`, `select_by`, `report_last_k` | `config.round_lr()`; `data._augment_spatial/_augment_intensity`; `train._tta_flips`, `remove_small_components`; `federated.snapshot/restore`; `data.train_val_cases` |
| **v3** | `aug_intensity_p` → 0, `aug_noise_std` → 0, `train_per_hospital` 150 → 230 | none — configuration only |
| **v4** | `rounds` → 40, `eval_test_every` | `config.scores_test()` |
| **v5** | `lr_anneal_rounds`, `eval_batch_size`, `sw_batch_size` | `config.round_lr()` gained the `span` argument; `train._probs_2d/_probs_3d` gained batch parameters |

Supporting infrastructure added along the way:

| Field / feature | Why |
|---|---|
| `tag` + `run_dir()` | namespacing, so a rerun cannot append into an archived run |
| `guard_run_dir` | refuse to write into a populated run dir |
| `per_case.jsonl` + `case_id` | paired statistics need per-volume values, not means |
| `stage` field | keep round rows, TTA final rows and cross-matrix rows distinguishable |
| SHA-256 manifests | prove an archived log has not been edited |

**The v4 lesson is visible in `round_lr`.** v4 stretched a 25-round cosine over 40 rounds and lost
0.0099. The fix was `lr_anneal_rounds`, which decouples *how long we train* from *how fast we
anneal* — the schedule bottoms out on v3's timetable and the extra rounds are spent at the floor.

---

## 6. The tooling scripts

| Script | Use it when |
|---|---|
| `build_partition.py` | first-time setup — writes the committed split |
| `build_cache.py` | first-time setup, or after changing shift parameters |
| `run_experiment.py` | run one method |
| `run_matrix.py` | run all four; resumable, skips finished runs, centralized first as a sanity gate |
| `freeze_baseline.py` | archive runs with a SHA-256 manifest; `--verify` re-checks |
| `rescore.py` | re-evaluate a saved checkpoint (per-case Dice, TTA) **without retraining** |
| `compare_runs.py` | two runs, one estimator: config diff, Δ Dice, paired CI + Wilcoxon |
| `compute_significance.py` | FedBN vs FedAvg at one recipe, **Holm-corrected** |
| `analyze.py` | hypothesis verdicts from a run directory |
| `plot_results.py` | learning curves, per-hospital bars, outlier plots |
| `export_predictions.py` | qualitative prediction PNGs for one case |
| `check_docs.py` | **run before committing docs** — re-derives 16 figures, checks naming, links, snapshots |
| `demo_server.py` | the live UI and the walkthrough |
| `build_showcase.py` / `showcase_assets.py` | rebuild the walkthrough from the logs |
| `build_review_pack.py` | assemble `review/` for a presentation |

`run_matrix.py` runs **centralized first on purpose** — it is the sanity gate. If the pooled ceiling
does not climb well past 0.7 WT, the fault is in the data pipeline or the loss, not in federation,
and the remaining three runs would be hours spent measuring the same bug three more times.

---

## 7. Tests

```bash
uv run pytest
```

| File | Covers |
|---|---|
| `test_metrics.py` | Dice, including the empty-GT convention |
| `test_partition.py` | determinism, no case in two hospitals, split sizes |
| `test_model.py` | shapes in 2D and 3D, and that `bn_keys` finds the BN layers |
| `test_data.py` | preprocessing chain, cache round-trip, sampler shapes |
| `test_improvements.py` | the v2+ additions: LR schedule, augmentation, TTA, postprocessing |

Two identities from the aggregation math make cheap tests that need no training:

- **FedBN with K = 1 ≡ local-only** — aggregating one client and restoring its own BN is identity.
- **Averaging identical client states ≡ that state**, with dtypes (including `int64`) surviving.

---

## 8. Things that bit us — worth knowing before you change anything

| Trap | Symptom | Guard |
|---|---|---|
| Averaging `parameters()` not `state_dict()` | FedBN silently becomes FedAvg | `bn_keys()` walks modules by type |
| Averaging `num_batches_tracked` (`int64`) | `load_state_dict` rejects or corrupts | integer buffers copied, not averaged |
| Brain mask taken after the shift | hospital identity leaks as volume shape | mask comes from the unshifted volume |
| Linear-only scanner shift | z-norm erases it; no heterogeneity | shift is gamma + bias field + blur |
| AMP / fp16 | corrupts BN running statistics | fp32 everywhere |
| Augmenting intensity | competes with the experimental variable (v2) | `aug_intensity_p = 0` from v3 |
| Reading a single round | conclusions flip sign | estimator = mean of last 5 rounds |
| Re-running into a populated run dir | duplicate rows, self-paired tests | `guard_run_dir` |
| Editing a doc without rebuilding `review/` | stale copies presented as current | `check_docs.py` fails |
