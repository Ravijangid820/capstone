# Specifications

Reference sheet — concrete numbers in one place. Values marked *(default)* are starting points, tunable.

## 1. Dataset

BraTS 2021 — 1251 cases, 3D, 240×240×155, 4 modalities + `{0,1,2,4}` mask. Full spec in [data.md](../data.md).

## 2. Model — U-Net

| Property | Value |
|---|---|
| Family | U-Net (encoder–decoder + skips) |
| Levels | 3 down / 3 up + bottleneck |
| Channels | base **16 or 32**, doubling per level (b, 2b, 4b, 8b) |
| Norm | **BatchNorm** (2d or 3d) — required for FedBN |
| Input channels | 4 (FLAIR, T1, T1ce, T2) |
| Output channels | 3 (WT, TC, ET) |
| Conv | `Conv2d` or `Conv3d` by the `dim` flag |
| Precision | **fp32** — AMP/fp16 corrupts BN running stats on shifted data |

## 3. Hyperparameters *(default)*

| Knob | 2D | 3D |
|---|---|---|
| Optimizer | Adam (fresh per round — no optimizer state is transmitted) | same |
| Learning rate | 1e-3 | 1e-3 |
| Loss | Dice + BCE on **independent sigmoids** (regions overlap; softmax would be wrong) | same |
| Input unit | axial slice, random-cropped to 192² | patch 96³ *(or 128³)* |
| Batch size | 8 | 1 |
| Base channels | 32 | 16 |
| Units drawn per case per epoch | 8 slices | 2 patches |
| `tumor_frac` (foreground bias) | 0.7 | 0.7 |
| Local epochs per round `E` | 1–2 | 1–2 |
| FL rounds `R` | 20 (default) / 25 (actual experiments) | 20 (default) / 25 (actual experiments) |
| `train_per_hospital` | 120–150 (of ~250) | 120–150 |
| Seed | 42 | 42 |

**Matched compute.** `local` and `centralized` train for `R × E` epochs — the same total local epochs
a hospital spends across a whole federated run. See [experiments](experiments.md) §3.

### 3.1 Presets — `--preset v5` is the current best

Every knob in §3 keeps its baseline value unless a preset or an explicit flag changes it, so an
un-flagged run still reproduces the frozen v1 baseline bit for bit. Full rationale for each
iteration in [iteration-report.md](iteration-report.md); naming and retracted claims in
[conventions.md](../conventions.md).

| Knob | v1 (default) | v2 | v3 | v4 | **v5** |
|---|---|---|---|---|---|
| `rounds` | 25 | 25 | 25 | 40 | **40** |
| `lr_anneal_rounds` | — | — | — | (40) | **25** |
| `lr_schedule` / `lr_min_factor` | `constant` | `cosine` → 5 % of `lr` | same | same | same |
| `train_per_hospital` | 150 | 150 | **230** | 230 | **230** |
| `val_per_hospital` | 0 | 20 | 20 | 20 | **20** |
| `augment` | `False` | `True` | `True` | `True` | **`True`** |
| `aug_flip_p` / `aug_rot90_p` | — | 0.5 / 0.5 | 0.5 / 0.5 | 0.5 / 0.5 | **0.5 / 0.5** |
| `aug_intensity_p` | — | 0.3 | **0.0** | 0.0 | **0.0** |
| `aug_noise_std` | — | 0.02 | **0.0** | 0.0 | **0.0** |
| `tta` | `False` | `True` | `True` | `True` | **`True`** |
| `postproc_min_voxels` | 0 | 50 | 50 | 50 | **50** |
| `select_by` | `last` | `best_val` | `best_val` | `best_val` | **`best_val`** |
| `report_last_k` | 1 | 5 | 5 | 5 | **5** |
| `eval_test_every` | 1 | 1 | 1 | 3 | **3** |
| `eval_batch_size` / `sw_batch_size` | 8 / 1 | 8 / 1 | 8 / 1 | 8 / 1 | **64 / 4** |

**v2 and v4 are kept only as evidence and should not be run.** v2's intensity and noise
augmentation perturbs the same channels as the synthetic scanner shift and regressed FedAvg in 2D
and three of four methods in 3D. v4 stretched the cosine across all 40 rounds and lost 0.0099 to
v3 on 2D centralized.

Cache requirement: v3–v5 need **250 cases/hospital** cached (230 train + 20 val); v2 needs 170;
v1 needs 150. `rot90` applies in 2D only — the 3D axes are anatomically distinct, and the config
records `aug_rot90_p=0.0` for 3D so the recorded configuration matches what ran.

Augmentation and TTA never overlap: no augmentation is applied on any evaluation path, and TTA
touches no weights. `eval_batch_size` / `sw_batch_size` change speed only — the model is in
`eval()` mode, so per-case Dice moves by at most 1.5e-4.

## 4. Hospitals / split

4 hospitals (3 typical + 1 outlier). ~1000 train / ~251 test, partition-then-split. Details and the
scanner-shift spec in [data-pipeline.md](data-pipeline.md).

## 5. Hardware — measured

Numbers from probes on the actual hardware (see [progress-log](progress-log.md)).

**Local — RTX 3050 Laptop, 4 GB · 16 cores · 11 GB RAM**

| Config | VRAM | Time / step |
|---|---|---|
| 2D 240², base 32, batch 8 | — | 175 ms (8 slices) |
| 3D 96³, base 16, batch 1 | 0.88 GB | 201 ms |
| 3D 96³, base 32, batch 1 | 1.78 GB | 464 ms |
| 3D 128³, base 16, batch 1 | 2.06 GB | 480 ms |
| 3D 128³, base 32, batch 1 | 4.07 GB *(over 4 GB — spills)* | — |

→ 3D **fits in memory** locally at 96³/128³; speed is the limiter for full sweeps.

**Measured throughput** (RTX 3050, fp32, 2D, batch 8, 192²):

| Operation | Cost | Full-run implication |
|---|---|---|
| Training step | 48 ms | 600 steps/round → 0.5 min |
| Full-volume evaluation | 0.41 s / volume | 248 volumes/round → **1.7 min** |
| Preprocess + cache one case | 2.3 s (4 workers) | 848 cases → ≈ 33 min |
| Cache size | 35 MB / case | 848 cases → ≈ 30 GB |

→ **Evaluation costs 3.5× training** per round. The run is eval-bound, not train-bound. Full 2D
matrix (4 methods × 25 rounds): ≈ 3.7 h on the 3050, ≈ 2.5 h on a T4. See [workflow.md](workflow.md) §4.

**Training — Colab T4, 16 GB** (heavy runs; ~100 GB ephemeral local disk).

## 6. Reproducibility

- One global **seed = 42** drives partition shuffle, sampler, and weight init.
- Deterministic where practical (`torch` deterministic flags; fixed seed per run).
- The **committed manifest** (`artifacts/splits/partition.json`) pins the exact split for every run.

## 7. Directory & artifact layout

```
tests/                           pytest suite (44 tests)
src/fedbrats/static/             web demo frontend
src/fedbrats/static/vendor/      bundled Three.js + OrbitControls (offline)
scripts/demo_server.py           web demo HTTP server (port 8000)
artifacts/                       (git-ignored, except splits/ and baseline/)
  splits/partition.json          committed — the source-of-truth split
  cache/<key>/                   preprocessed tensors; <key> = md5(shift params + clip + seed)
    <case_id>/x.npy              (4,X,Y,Z) float16
    <case_id>/y.npy              (3,X,Y,Z) uint8
    <case_id>/meta.json          shape, tumor_z, tumor_bbox  (presence = "already built")
    index.json                   assembled from the meta files
  runs/[<tag>/]<run_id>/
    config.json                  exact config used (typed JSON, every knob)
    run.log                      human log (timestamped INFO)
    metrics.jsonl                machine metrics (one row per measurement)
    per_case.jsonl               per-volume Dice — the input to CIs and paired tests
    summary.json                 reported round, selection rule, best val score
    checkpoints/final.pt         model weights (ignored)
  baseline/                      committed — frozen "before" snapshot
    MANIFEST.json                SHA-256 per file + git commit + headline numbers
    <run_id>/                    copied metrics/config/log, plus rescore_* backfills
```

`<run_id>` = `<method>_<dim>_<seed>` (e.g. `fedbn_2d_42`). `<tag>` namespaces a rerun
(`--tag v2` → `artifacts/runs/v2/fedbn_2d_42/`); without one, writing into an existing run
directory is refused rather than appended to. See [improvements.md](improvements.md) §3.

The full cache is **~44 GB** (~35 MB/case, measured) and must not land on the WSL VHDX — override
with `FEDBRATS_CACHE_DIR`. See [environments.md](environments.md).

## 8. Logging format

`metrics.jsonl` — one JSON object per line, one per (round × model × test-set) measurement:

```json
{"run_id":"local_2d_42","method":"local","dim":"2d","round":30,
 "model_hospital":"H4","test_hospital":"H1","split":"test",
 "dice_wt":0.41,"dice_tc":0.33,"dice_et":0.28}
```

| Field | Meaning |
|---|---|
| `run_id` | which run — `<method>_<dim>_<seed>` |
| `method` | centralized · local · fedavg · fedbn |
| `dim` | 2d · 3d |
| `round` | FL round |
| `model_hospital` | **whose model** — `"global"` for centralized/FedAvg; `H1`–`H4` for FedBN/local |
| `test_hospital` | **whose test set** — `H1`–`H4` |
| `split` | train · **val** · test |
| `stage` | `round` · `final` · `cross` · `rescore` — see below |
| `tta` | whether flip test-time augmentation produced this row |
| `dice_wt/tc/et` | per-volume Dice per region, averaged over that test set's cases |

The two `*_hospital` fields are separate because the local-only run reports a full 4×4
cross-hospital matrix. **Diagonal** = `model_hospital == test_hospital`; that is where H1/H2/H3 live.
Off-diagonal cells exist only for `method == "local"`.

`stage` distinguishes measurements that are **not interchangeable**, and reading one as the other
silently mixes inference settings:

| `stage` | What produced it |
|---|---|
| `round` | the per-round evaluation — the learning curve. No TTA. |
| `final` | the **selected** model re-scored at the end, with TTA if enabled. The headline number. |
| `cross` | local-only's off-diagonal 4×4 matrix, final round only |
| `rescore` | `scripts/rescore.py` re-evaluating a saved checkpoint; `round` is `-1` |

Rows written before this field existed have no `stage` and are treated as `round`.

`per_case.jsonl` carries the same fields plus `case_id`, one row per volume rather than a mean —
that is what supports confidence intervals and paired significance tests. Rescored rows land in
`rescore_metrics.jsonl`, never in `metrics.jsonl`, so a frozen baseline's hashes stay valid.

Plots and the H1/H2/H3 tables in [experiments](experiments.md) are generated directly from these rows.

## 9. Software environment

`uv`-managed; key deps: `torch` (CUDA), `monai` (U-Net + sliding-window inference), `nibabel`
(NIfTI I/O), `numpy`, `scipy` (the shift's bias field + blur). Run with `.venv/bin/python`
or `uv run python`.

`nvflare` is an **optional extra with a platform marker** (`uv sync --extra flare`, Linux only) —
it imports the POSIX-only `resource` module and cannot run on native Windows. Full matrix in
[environments.md](environments.md).
