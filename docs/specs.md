# Specifications

Reference sheet — concrete numbers in one place. Values marked *(default)* are starting points, tunable.

## 1. Dataset

BraTS 2021 — 1251 cases, 3D, 240×240×155, 4 modalities + `{0,1,2,4}` mask. Full spec in [data.md](data.md).

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
| FL rounds `R` | 20–30 | 20–30 |
| `train_per_hospital` | 120–150 (of ~250) | 120–150 |
| Seed | 42 | 42 |

**Matched compute.** `local` and `centralized` train for `R × E` epochs — the same total local epochs
a hospital spends across a whole federated run. See [experiments](experiments.md) §3.

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

**Training — Colab T4, 16 GB** (heavy runs; ~100 GB ephemeral local disk).

## 6. Reproducibility

- One global **seed = 42** drives partition shuffle, sampler, and weight init.
- Deterministic where practical (`torch` deterministic flags; fixed seed per run).
- The **committed manifest** (`artifacts/splits/partition.json`) pins the exact split for every run.

## 7. Directory & artifact layout

```
artifacts/                       (git-ignored, except splits/)
  splits/partition.json          committed — the source-of-truth split
  cache/<key>/                   preprocessed tensors; <key> = md5(shift params + clip + seed)
    <case_id>/x.npy              (4,X,Y,Z) float16
    <case_id>/y.npy              (3,X,Y,Z) uint8
    <case_id>/meta.json          shape, tumor_z, tumor_bbox  (presence = "already built")
    index.json                   assembled from the meta files
  runs/<run_id>/
    config.json                  exact config used
    run.log                      human log (timestamped INFO)
    metrics.jsonl                machine metrics (one row per measurement)
    checkpoints/final.pt         model weights (ignored)
```

`<run_id>` = `<method>_<dim>_<seed>` (e.g. `fedbn_2d_42`).

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
| `split` | train · test |
| `dice_wt/tc/et` | per-volume Dice per region, averaged over that test set's cases |

The two `*_hospital` fields are separate because the local-only run reports a full 4×4
cross-hospital matrix. **Diagonal** = `model_hospital == test_hospital`; that is where H1/H2/H3 live.
Off-diagonal cells exist only for `method == "local"`.

Plots and the H1/H2/H3 tables in [experiments](experiments.md) are generated directly from these rows.

## 9. Software environment

`uv`-managed; key deps: `torch` (CUDA), `monai` (U-Net + sliding-window inference), `nibabel`
(NIfTI I/O), `numpy`, `scipy` (the shift's bias field + blur). Run with `.venv/bin/python`
or `uv run python`.

`nvflare` is an **optional extra with a platform marker** (`uv sync --extra flare`, Linux only) —
it imports the POSIX-only `resource` module and cannot run on native Windows. Full matrix in
[environments.md](environments.md).
