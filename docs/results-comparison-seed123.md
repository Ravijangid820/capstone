# Baseline vs rerun — 2D

- **before:** `D:\capstone\artifacts\baseline`
- **after:** `D:\capstone\artifacts\runs\v2`
- **estimator (both sides):** mean of last 5 rounds
- **runs compared:** local_2d_123, fedavg_2d_123, fedbn_2d_123

## 1. What changed

| Field               | before | after    |
|---------------------|--------|----------|
| aug_flip_p          | —      | 0.5      |
| aug_intensity_p     | —      | 0.3      |
| aug_intensity_scale | —      | 0.1      |
| aug_intensity_shift | —      | 0.1      |
| aug_noise_std       | —      | 0.02     |
| aug_rot90_p         | —      | 0.5      |
| augment             | —      | True     |
| lr_min_factor       | —      | 0.05     |
| lr_schedule         | —      | cosine   |
| postproc_min_voxels | —      | 50       |
| report_last_k       | —      | 5        |
| select_by           | —      | best_val |
| sw_overlap          | —      | 0.25     |
| tta                 | —      | True     |
| val_per_hospital    | —      | 20       |

## 2. Dice, before → after

| Run           | WT before | WT after | WT Δ    | TC before | TC after | TC Δ    | ET before | ET after | ET Δ    |
|---------------|-----------|----------|---------|-----------|----------|---------|-----------|----------|---------|
| local_2d_123  | 0.8400    | 0.8546   | +0.0146 | 0.7960    | 0.8182   | +0.0222 | 0.7604    | 0.7814   | +0.0210 |
| fedavg_2d_123 | 0.8488    | 0.8420   | -0.0068 | 0.8156    | 0.8135   | -0.0021 | 0.7693    | 0.7702   | +0.0009 |
| fedbn_2d_123  | 0.8454    | 0.8560   | +0.0106 | 0.8149    | 0.8258   | +0.0109 | 0.7713    | 0.7828   | +0.0115 |

### Per-hospital WT Dice

| Run           | H1                        | H2                        | H3                        | H4                        |
|---------------|---------------------------|---------------------------|---------------------------|---------------------------|
| local_2d_123  | 0.8592 → 0.8636 (+0.0044) | 0.8605 → 0.8654 (+0.0049) | 0.8253 → 0.8482 (+0.0229) | 0.8149 → 0.8412 (+0.0263) |
| fedavg_2d_123 | 0.8866 → 0.8815 (-0.0051) | 0.8878 → 0.8812 (-0.0066) | 0.8648 → 0.8567 (-0.0080) | 0.7561 → 0.7487 (-0.0074) |
| fedbn_2d_123  | 0.8708 → 0.8761 (+0.0053) | 0.8573 → 0.8731 (+0.0158) | 0.8599 → 0.8577 (-0.0022) | 0.7937 → 0.8173 (+0.0236) |

## 3. Is the difference real? (paired, per case)

| Run           | Hospital | n   | Δ mean WT | 95% CI (bootstrap) | better/worse | Wilcoxon p  |
|---------------|----------|-----|-----------|--------------------|--------------|-------------|
| local_2d_123  | H1       | 62  | +0.0103   | [+0.0007, +0.0205] | 45/17        | 0.0017 **   |
| local_2d_123  | H2       | 62  | -0.0019   | [-0.0104, +0.0059] | 35/27        | 0.58 ns     |
| local_2d_123  | H3       | 62  | +0.0128   | [-0.0030, +0.0266] | 47/15        | 7.3e-05 *** |
| local_2d_123  | H4       | 62  | +0.0363   | [+0.0213, +0.0518] | 51/11        | 3.2e-06 *** |
| local_2d_123  | ALL      | 248 | +0.0144   | [+0.0078, +0.0209] | 178/70       | 3.3e-11 *** |
| fedavg_2d_123 | H1       | 62  | -0.0008   | [-0.0070, +0.0046] | 30/32        | 0.39 ns     |
| fedavg_2d_123 | H2       | 62  | -0.0056   | [-0.0124, +0.0003] | 23/39        | 0.026 *     |
| fedavg_2d_123 | H3       | 62  | -0.0066   | [-0.0151, +0.0009] | 23/39        | 0.026 *     |
| fedavg_2d_123 | H4       | 62  | -0.0190   | [-0.0284, -0.0103] | 20/42        | 9.3e-05 *** |
| fedavg_2d_123 | ALL      | 248 | -0.0080   | [-0.0119, -0.0043] | 96/152       | 3.3e-06 *** |
| fedbn_2d_123  | H1       | 62  | +0.0083   | [-0.0025, +0.0168] | 47/15        | 2.4e-05 *** |
| fedbn_2d_123  | H2       | 62  | +0.0044   | [-0.0036, +0.0132] | 35/27        | 0.36 ns     |
| fedbn_2d_123  | H3       | 62  | -0.0163   | [-0.0319, -0.0016] | 21/41        | 0.036 *     |
| fedbn_2d_123  | H4       | 62  | +0.1584   | [+0.1310, +0.1862] | 59/3         | 2.7e-11 *** |
| fedbn_2d_123  | ALL      | 248 | +0.0387   | [+0.0267, +0.0511] | 162/86       | 4.1e-09 *** |

`better/worse` counts cases that moved in each direction; the remainder tied. A CI excluding zero and a small p mean the improvement survives case-level noise.

Per-case rows read from stage: final, rescore. These can differ from section 2 — section 2 averages rounds of ordinary inference, while a `final` stage is the selected model re-scored with TTA. The two answer different questions and are expected to disagree in size.

> **⚠ Different stages on each side:** `fedavg_2d_123`: rescore → final; `fedbn_2d_123`: rescore → final; `local_2d_123`: rescore → final

