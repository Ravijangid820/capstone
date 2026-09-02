# Baseline vs rerun — 3D

- **before:** `D:\capstone\artifacts\baseline`
- **after:** `D:\capstone\artifacts\runs\v2`
- **estimator (both sides):** mean of last 5 rounds
- **runs compared:** centralized_3d_42, local_3d_42, fedavg_3d_42, fedbn_3d_42

## 1. What changed

| Field               | before | after    |
|---------------------|--------|----------|
| aug_flip_p          | —      | 0.5      |
| aug_intensity_p     | —      | 0.3      |
| aug_intensity_scale | —      | 0.1      |
| aug_intensity_shift | —      | 0.1      |
| aug_noise_std       | —      | 0.02     |
| aug_rot90_p         | —      | 0.0      |
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
| centralized_3d_42 | 0.8768    | 0.8770   | +0.0002 | 0.8326    | 0.8281   | -0.0045 | 0.7911    | 0.7893   | -0.0017 |
| local_3d_42       | 0.8490    | 0.8475   | -0.0016 | 0.7718    | 0.7554   | -0.0164 | 0.7502    | 0.7442   | -0.0060 |
| fedavg_3d_42      | 0.8525    | 0.8399   | -0.0125 | 0.7922    | 0.7631   | -0.0291 | 0.7627    | 0.7307   | -0.0320 |
| fedbn_3d_42       | 0.8422    | 0.8333   | -0.0089 | 0.7896    | 0.7583   | -0.0314 | 0.7570    | 0.7263   | -0.0307 |

### Per-hospital WT Dice

| Run               | H1                        | H2                        | H3                        | H4                        |
|-------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| centralized_3d_42 | 0.8879 → 0.8923 (+0.0044) | 0.8871 → 0.8887 (+0.0016) | 0.8761 → 0.8810 (+0.0049) | 0.8561 → 0.8461 (-0.0100) |
| local_3d_42       | 0.8692 → 0.8664 (-0.0027) | 0.8497 → 0.8625 (+0.0127) | 0.8457 → 0.8325 (-0.0132) | 0.8315 → 0.8285 (-0.0030) |
| fedavg_3d_42      | 0.8568 → 0.8537 (-0.0032) | 0.8617 → 0.8503 (-0.0113) | 0.8521 → 0.8386 (-0.0135) | 0.8392 → 0.8171 (-0.0221) |
| fedbn_3d_42       | 0.8449 → 0.8428 (-0.0021) | 0.8456 → 0.8410 (-0.0046) | 0.8452 → 0.8374 (-0.0078) | 0.8332 → 0.8120 (-0.0211) |

## 3. Is the difference real? (paired, per case)

| Run               | Hospital | n   | Δ mean WT | 95% CI (bootstrap) | better/worse | Wilcoxon p  |
|-------------------|----------|-----|-----------|--------------------|--------------|-------------|
| centralized_3d_42 | H1       | 62  | -0.0048   | [-0.0133, +0.0034] | 30/32        | 0.42 ns     |
| centralized_3d_42 | H2       | 62  | -0.0040   | [-0.0206, +0.0172] | 37/25        | 0.92 ns     |
| centralized_3d_42 | H3       | 62  | +0.0112   | [+0.0018, +0.0226] | 41/21        | 0.0065 **   |
| centralized_3d_42 | H4       | 62  | -0.0110   | [-0.0265, +0.0026] | 30/32        | 0.6 ns      |
| centralized_3d_42 | ALL      | 248 | -0.0022   | [-0.0091, +0.0050] | 138/110      | 0.66 ns     |
| local_3d_42       | H1       | 62  | -0.0029   | [-0.0149, +0.0087] | 26/35        | 0.39 ns     |
| local_3d_42       | H2       | 62  | -0.0130   | [-0.0375, +0.0067] | 31/31        | 0.89 ns     |
| local_3d_42       | H3       | 62  | -0.0034   | [-0.0120, +0.0049] | 29/33        | 0.69 ns     |
| local_3d_42       | H4       | 62  | +0.0127   | [-0.0047, +0.0291] | 43/19        | 0.007 **    |
| local_3d_42       | ALL      | 248 | -0.0016   | [-0.0097, +0.0061] | 129/118      | 0.24 ns     |
| fedavg_3d_42      | H1       | 62  | -0.0001   | [-0.0098, +0.0097] | 28/33        | 0.68 ns     |
| fedavg_3d_42      | H2       | 62  | -0.0105   | [-0.0232, +0.0020] | 20/42        | 0.013 *     |
| fedavg_3d_42      | H3       | 62  | -0.0172   | [-0.0334, -0.0044] | 19/43        | 0.0021 **   |
| fedavg_3d_42      | H4       | 62  | -0.0342   | [-0.0497, -0.0202] | 14/48        | 1.2e-06 *** |
| fedavg_3d_42      | ALL      | 248 | -0.0155   | [-0.0224, -0.0089] | 81/166       | 1.9e-08 *** |
| fedbn_3d_42       | H1       | 62  | -0.0082   | [-0.0218, +0.0033] | 32/29        | 0.6 ns      |
| fedbn_3d_42       | H2       | 62  | -0.0178   | [-0.0487, +0.0079] | 36/26        | 0.84 ns     |
| fedbn_3d_42       | H3       | 62  | +0.0393   | [+0.0076, +0.0720] | 38/23        | 0.0068 **   |
| fedbn_3d_42       | H4       | 62  | -0.0189   | [-0.0383, -0.0009] | 31/31        | 0.19 ns     |
| fedbn_3d_42       | ALL      | 248 | -0.0014   | [-0.0141, +0.0109] | 137/109      | 0.5 ns      |

`better/worse` counts cases that moved in each direction; the remainder tied. A CI excluding zero and a small p mean the improvement survives case-level noise.

Per-case rows read from stage: final, rescore. These can differ from section 2 — section 2 averages rounds of ordinary inference, while a `final` stage is the selected model re-scored with TTA. The two answer different questions and are expected to disagree in size.

> **⚠ Different stages on each side:** `centralized_3d_42`: rescore → final; `fedavg_3d_42`: rescore → final; `fedbn_3d_42`: rescore → final; `local_3d_42`: rescore → final

