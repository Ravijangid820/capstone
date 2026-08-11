# Baseline vs rerun — 2D

- **before:** `D:\capstone\artifacts\baseline`
- **after:** `D:\capstone\artifacts\runs\v2`
- **estimator (both sides):** mean of last 5 rounds
- **runs compared:** local_2d_7, fedavg_2d_7, fedbn_2d_7

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

| Run         | WT before | WT after | WT Δ    | TC before | TC after | TC Δ    | ET before | ET after | ET Δ    |
|-------------|-----------|----------|---------|-----------|----------|---------|-----------|----------|---------|
| local_2d_7  | 0.8462    | 0.8495   | +0.0033 | 0.8166    | 0.8244   | +0.0078 | 0.7734    | 0.7839   | +0.0106 |
| fedavg_2d_7 | 0.8504    | 0.8482   | -0.0022 | 0.8232    | 0.8280   | +0.0048 | 0.7767    | 0.7813   | +0.0045 |
| fedbn_2d_7  | 0.8456    | 0.8599   | +0.0143 | 0.8165    | 0.8326   | +0.0161 | 0.7709    | 0.7906   | +0.0197 |

### Per-hospital WT Dice

| Run         | H1                        | H2                        | H3                        | H4                        |
|-------------|---------------------------|---------------------------|---------------------------|---------------------------|
| local_2d_7  | 0.8537 → 0.8630 (+0.0093) | 0.8596 → 0.8622 (+0.0026) | 0.8338 → 0.8391 (+0.0053) | 0.8377 → 0.8336 (-0.0041) |
| fedavg_2d_7 | 0.8854 → 0.8826 (-0.0028) | 0.8849 → 0.8851 (+0.0002) | 0.8542 → 0.8539 (-0.0003) | 0.7769 → 0.7711 (-0.0059) |
| fedbn_2d_7  | 0.8748 → 0.8782 (+0.0034) | 0.8729 → 0.8699 (-0.0030) | 0.8163 → 0.8584 (+0.0421) | 0.8185 → 0.8330 (+0.0146) |

## 3. Is the difference real? (paired, per case)

| Run         | Hospital | n   | Δ mean WT | 95% CI (bootstrap) | better/worse | Wilcoxon p  |
|-------------|----------|-----|-----------|--------------------|--------------|-------------|
| local_2d_7  | H1       | 62  | +0.0227   | [+0.0128, +0.0329] | 51/11        | 4.3e-08 *** |
| local_2d_7  | H2       | 62  | +0.0096   | [-0.0023, +0.0212] | 44/18        | 0.019 *     |
| local_2d_7  | H3       | 62  | -0.0077   | [-0.0231, +0.0054] | 33/29        | 0.97 ns     |
| local_2d_7  | H4       | 62  | -0.0061   | [-0.0203, +0.0068] | 36/26        | 0.64 ns     |
| local_2d_7  | ALL      | 248 | +0.0046   | [-0.0020, +0.0110] | 164/84       | 3.3e-05 *** |
| fedavg_2d_7 | H1       | 62  | -0.0012   | [-0.0099, +0.0073] | 33/29        | 0.99 ns     |
| fedavg_2d_7 | H2       | 62  | +0.0049   | [-0.0025, +0.0117] | 39/23        | 0.016 *     |
| fedavg_2d_7 | H3       | 62  | +0.0198   | [+0.0090, +0.0328] | 45/17        | 0.00022 *** |
| fedavg_2d_7 | H4       | 62  | +0.0215   | [+0.0116, +0.0316] | 46/16        | 5e-05 ***   |
| fedavg_2d_7 | ALL      | 248 | +0.0113   | [+0.0064, +0.0163] | 163/85       | 7.2e-08 *** |
| fedbn_2d_7  | H1       | 62  | +0.0183   | [+0.0103, +0.0276] | 50/12        | 5.3e-07 *** |
| fedbn_2d_7  | H2       | 62  | -0.0009   | [-0.0118, +0.0101] | 36/26        | 0.66 ns     |
| fedbn_2d_7  | H3       | 62  | +0.0305   | [+0.0158, +0.0474] | 40/22        | 0.00055 *** |
| fedbn_2d_7  | H4       | 62  | +0.0111   | [-0.0009, +0.0242] | 45/17        | 0.0043 **   |
| fedbn_2d_7  | ALL      | 248 | +0.0147   | [+0.0086, +0.0214] | 171/77       | 8.6e-10 *** |

`better/worse` counts cases that moved in each direction; the remainder tied. A CI excluding zero and a small p mean the improvement survives case-level noise.

Per-case rows read from stage: final, rescore. These can differ from section 2 — section 2 averages rounds of ordinary inference, while a `final` stage is the selected model re-scored with TTA. The two answer different questions and are expected to disagree in size.

> **⚠ Different stages on each side:** `fedavg_2d_7`: rescore → final; `fedbn_2d_7`: rescore → final; `local_2d_7`: rescore → final

