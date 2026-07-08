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
| Optimizer | Adam | Adam |
| Learning rate | 1e-3 | 1e-3 |
| Loss | Dice + BCE (soft Dice for imbalance) | same |
| Input unit | axial slice 240×240 (cropped) | patch 96³ *(or 128³)* |
| Batch size | 8–16 | 1–2 |
| Local epochs per round `E` | 1–2 | 1–2 |
| FL rounds `R` | 20–30 | 20–30 |
| `train_per_hospital` | 120–150 (of ~250) | 120–150 |
| Seed | 42 | 42 |

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
  runs/<run_id>/
    config.json                  exact config used
    run.log                      human log (timestamped INFO)
    metrics.jsonl                machine metrics (one row per measurement)
    checkpoints/                 model weights (ignored)
```

`<run_id>` = `<method>_<dim>_<seed>` (e.g. `fedbn_2d_42`).

## 8. Logging format

`metrics.jsonl` — one JSON object per line, one per (round × hospital × split) measurement:

```json
{"run_id":"fedbn_2d_42","method":"fedbn","dim":"2d","round":12,
 "hospital":"H4","split":"test","dice_wt":0.78,"dice_tc":0.71,"dice_et":0.66,"loss":0.21}
```

| Field | Meaning |
|---|---|
| `run_id` | which run |
| `method` | centralized · local · fedavg · fedbn |
| `dim` | 2d · 3d |
| `round` | FL round (0 for centralized/local epochs) |
| `hospital` | H1–H4 |
| `split` | train · test |
| `dice_wt/tc/et` | Dice per region |
| `loss` | training/eval loss |

Plots and the H1/H2/H3 tables in [experiments](experiments.md) are generated directly from these rows.

## 9. Software environment

`uv`-managed; key deps: `torch` (CUDA), `nibabel` (NIfTI I/O), `numpy`. Run with `.venv/bin/python`.
