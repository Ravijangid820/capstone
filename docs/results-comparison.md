# Baseline vs rerun — 2D

- **before:** `D:\capstone\artifacts\baseline`
- **after:** `D:\capstone\artifacts\runs\v2`
- **estimator (both sides):** mean of last 5 rounds
- **runs compared:** centralized_2d_42, local_2d_42, fedavg_2d_42, fedbn_2d_42

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

| Run               | WT before | WT after | WT Δ    | TC before | TC after | TC Δ    | ET before | ET after | ET Δ    |
|-------------------|-----------|----------|---------|-----------|----------|---------|-----------|----------|---------|
| centralized_2d_42 | 0.8582    | 0.8835   | +0.0253 | 0.8341    | 0.8522   | +0.0180 | 0.7959    | 0.8067   | +0.0108 |
| local_2d_42       | 0.8436    | 0.8583   | +0.0148 | 0.8228    | 0.8348   | +0.0120 | 0.7809    | 0.7947   | +0.0138 |
| fedavg_2d_42      | 0.8502    | 0.8444   | -0.0057 | 0.8274    | 0.8291   | +0.0017 | 0.7762    | 0.7816   | +0.0054 |
| fedbn_2d_42       | 0.8514    | 0.8610   | +0.0096 | 0.8315    | 0.8381   | +0.0067 | 0.7838    | 0.7940   | +0.0102 |

### Per-hospital WT Dice

| Run               | H1                        | H2                        | H3                        | H4                        |
|-------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| centralized_2d_42 | 0.8735 → 0.8962 (+0.0228) | 0.8780 → 0.9026 (+0.0246) | 0.8516 → 0.8776 (+0.0260) | 0.8299 → 0.8576 (+0.0277) |
| local_2d_42       | 0.8508 → 0.8707 (+0.0199) | 0.8638 → 0.8629 (-0.0008) | 0.8226 → 0.8547 (+0.0320) | 0.8371 → 0.8451 (+0.0080) |
| fedavg_2d_42      | 0.8800 → 0.8768 (-0.0032) | 0.8852 → 0.8865 (+0.0014) | 0.8601 → 0.8618 (+0.0017) | 0.7754 → 0.7526 (-0.0228) |
| fedbn_2d_42       | 0.8662 → 0.8788 (+0.0126) | 0.8654 → 0.8740 (+0.0087) | 0.8405 → 0.8669 (+0.0263) | 0.8334 → 0.8242 (-0.0093) |

## 3. Is the difference real? (paired, per case)

| Run               | Hospital | n   | Δ mean WT | 95% CI (bootstrap) | better/worse | Wilcoxon p  |
|-------------------|----------|-----|-----------|--------------------|--------------|-------------|
| centralized_2d_42 | H1       | 62  | +0.0355   | [+0.0206, +0.0507] | 54/8         | 3.6e-08 *** |
| centralized_2d_42 | H2       | 62  | +0.0388   | [+0.0243, +0.0565] | 57/5         | 4.5e-09 *** |
| centralized_2d_42 | H3       | 62  | +0.0454   | [+0.0289, +0.0652] | 53/9         | 1e-09 ***   |
| centralized_2d_42 | H4       | 62  | +0.0360   | [+0.0208, +0.0527] | 45/17        | 6.4e-06 *** |
| centralized_2d_42 | ALL      | 248 | +0.0389   | [+0.0311, +0.0475] | 209/39       | 3.1e-28 *** |
| local_2d_42       | H1       | 62  | +0.0268   | [+0.0144, +0.0393] | 49/13        | 2.6e-07 *** |
| local_2d_42       | H2       | 62  | +0.0059   | [-0.0062, +0.0173] | 40/22        | 0.05 ns     |
| local_2d_42       | H3       | 62  | +0.0111   | [-0.0014, +0.0235] | 42/20        | 0.0039 **   |
| local_2d_42       | H4       | 62  | -0.0016   | [-0.0135, +0.0082] | 40/22        | 0.051 ns    |
| local_2d_42       | ALL      | 248 | +0.0106   | [+0.0045, +0.0167] | 171/77       | 3.2e-09 *** |
| fedavg_2d_42      | H1       | 62  | -0.0063   | [-0.0145, +0.0011] | 24/38        | 0.11 ns     |
| fedavg_2d_42      | H2       | 62  | +0.0024   | [-0.0045, +0.0098] | 35/27        | 0.27 ns     |
| fedavg_2d_42      | H3       | 62  | +0.0244   | [+0.0133, +0.0373] | 48/14        | 1.1e-05 *** |
| fedavg_2d_42      | H4       | 62  | +0.0274   | [+0.0186, +0.0365] | 51/11        | 1.3e-07 *** |
| fedavg_2d_42      | ALL      | 248 | +0.0120   | [+0.0071, +0.0169] | 158/90       | 3.3e-07 *** |
| fedbn_2d_42       | H1       | 62  | +0.0181   | [+0.0113, +0.0259] | 53/9         | 1.6e-08 *** |
| fedbn_2d_42       | H2       | 62  | +0.0118   | [+0.0038, +0.0218] | 48/14        | 0.00035 *** |
| fedbn_2d_42       | H3       | 62  | +0.0234   | [+0.0127, +0.0358] | 49/13        | 6.4e-06 *** |
| fedbn_2d_42       | H4       | 62  | +0.0022   | [-0.0087, +0.0124] | 46/16        | 0.0063 **   |
| fedbn_2d_42       | ALL      | 248 | +0.0139   | [+0.0089, +0.0190] | 196/52       | 4.7e-16 *** |

`better/worse` counts cases that moved in each direction; the remainder tied. A CI excluding zero and a small p mean the improvement survives case-level noise.

Per-case rows read from stage: final, rescore. These can differ from section 2 — section 2 averages rounds of ordinary inference, while a `final` stage is the selected model re-scored with TTA. The two answer different questions and are expected to disagree in size.

> **⚠ Different stages on each side:** `centralized_2d_42`: rescore → final; `fedavg_2d_42`: rescore → final; `fedbn_2d_42`: rescore → final; `local_2d_42`: rescore → final

