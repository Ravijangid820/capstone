# Baseline vs rerun — 3D

- **before:** `artifacts\baseline`
- **after:** `artifacts\runs\v3`
- **estimator (both sides):** mean of last 5 rounds
- **runs compared:** centralized_3d_42, local_3d_42, fedavg_3d_42, fedbn_3d_42

## 1. What changed

| Field               | before | after    |
|---------------------|--------|----------|
| aug_flip_p          | —      | 0.5      |
| aug_intensity_p     | —      | 0.0      |
| aug_intensity_scale | —      | 0.1      |
| aug_intensity_shift | —      | 0.1      |
| aug_noise_std       | —      | 0.0      |
| aug_rot90_p         | —      | 0.0      |
| augment             | —      | True     |
| lr_min_factor       | —      | 0.05     |
| lr_schedule         | —      | cosine   |
| postproc_min_voxels | —      | 50       |
| report_last_k       | —      | 5        |
| select_by           | —      | best_val |
| sw_overlap          | —      | 0.25     |
| train_per_hospital  | 150    | 230      |
| tta                 | —      | True     |
| val_per_hospital    | —      | 20       |

## 2. Dice, before → after

| Run               | WT before | WT after | WT Δ    | TC before | TC after | TC Δ    | ET before | ET after | ET Δ    |
|-------------------|-----------|----------|---------|-----------|----------|---------|-----------|----------|---------|
| centralized_3d_42 | 0.8768    | 0.8836   | +0.0068 | 0.8326    | 0.8381   | +0.0054 | 0.7911    | 0.7978   | +0.0068 |
| local_3d_42       | 0.8490    | 0.8489   | -0.0002 | 0.7718    | 0.7664   | -0.0054 | 0.7502    | 0.7490   | -0.0012 |
| fedavg_3d_42      | 0.8525    | 0.8555   | +0.0030 | 0.7922    | 0.7974   | +0.0052 | 0.7627    | 0.7615   | -0.0012 |
| fedbn_3d_42       | 0.8422    | 0.8550   | +0.0128 | 0.7896    | 0.7967   | +0.0070 | 0.7570    | 0.7635   | +0.0065 |

### Per-hospital WT Dice

| Run               | H1                        | H2                        | H3                        | H4                        |
|-------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| centralized_3d_42 | 0.8879 → 0.8943 (+0.0064) | 0.8871 → 0.8947 (+0.0077) | 0.8761 → 0.8904 (+0.0143) | 0.8561 → 0.8551 (-0.0010) |
| local_3d_42       | 0.8692 → 0.8604 (-0.0088) | 0.8497 → 0.8591 (+0.0093) | 0.8457 → 0.8438 (-0.0020) | 0.8315 → 0.8324 (+0.0008) |
| fedavg_3d_42      | 0.8568 → 0.8653 (+0.0085) | 0.8617 → 0.8686 (+0.0069) | 0.8521 → 0.8598 (+0.0077) | 0.8392 → 0.8282 (-0.0110) |
| fedbn_3d_42       | 0.8449 → 0.8623 (+0.0175) | 0.8456 → 0.8679 (+0.0223) | 0.8452 → 0.8602 (+0.0150) | 0.8332 → 0.8297 (-0.0035) |

## 3. Is the difference real? (paired, per case)

| Run               | Hospital | n   | Δ mean WT | 95% CI (bootstrap) | better/worse | Wilcoxon p  |
|-------------------|----------|-----|-----------|--------------------|--------------|-------------|
| centralized_3d_42 | H1       | 62  | -0.0047   | [-0.0126, +0.0039] | 34/28        | 0.29 ns     |
| centralized_3d_42 | H2       | 62  | +0.0003   | [-0.0137, +0.0173] | 40/22        | 0.49 ns     |
| centralized_3d_42 | H3       | 62  | +0.0156   | [+0.0044, +0.0294] | 47/15        | 0.00062 *** |
| centralized_3d_42 | H4       | 62  | +0.0151   | [+0.0065, +0.0241] | 50/12        | 6e-05 ***   |
| centralized_3d_42 | ALL      | 248 | +0.0066   | [+0.0010, +0.0130] | 171/77       | 0.0003 ***  |
| local_3d_42       | H1       | 62  | -0.0022   | [-0.0148, +0.0102] | 32/29        | 0.96 ns     |
| local_3d_42       | H2       | 62  | -0.0109   | [-0.0310, +0.0047] | 32/30        | 0.95 ns     |
| local_3d_42       | H3       | 62  | +0.0022   | [-0.0104, +0.0142] | 37/25        | 0.17 ns     |
| local_3d_42       | H4       | 62  | +0.0222   | [+0.0104, +0.0343] | 47/15        | 0.00034 *** |
| local_3d_42       | ALL      | 248 | +0.0028   | [-0.0046, +0.0099] | 148/99       | 0.0054 **   |
| fedavg_3d_42      | H1       | 62  | +0.0104   | [-0.0064, +0.0231] | 43/18        | 8.7e-05 *** |
| fedavg_3d_42      | H2       | 62  | +0.0108   | [-0.0058, +0.0312] | 35/27        | 0.36 ns     |
| fedavg_3d_42      | H3       | 62  | +0.0060   | [-0.0076, +0.0208] | 38/24        | 0.066 ns    |
| fedavg_3d_42      | H4       | 62  | -0.0177   | [-0.0350, -0.0035] | 28/34        | 0.36 ns     |
| fedavg_3d_42      | ALL      | 248 | +0.0024   | [-0.0058, +0.0103] | 144/103      | 0.0046 **   |
| fedbn_3d_42       | H1       | 62  | +0.0249   | [+0.0004, +0.0445] | 45/16        | 4.2e-06 *** |
| fedbn_3d_42       | H2       | 62  | +0.0030   | [-0.0231, +0.0255] | 40/22        | 0.054 ns    |
| fedbn_3d_42       | H3       | 62  | +0.0721   | [+0.0362, +0.1110] | 47/15        | 4.8e-06 *** |
| fedbn_3d_42       | H4       | 62  | +0.0021   | [-0.0153, +0.0181] | 42/20        | 0.071 ns    |
| fedbn_3d_42       | ALL      | 248 | +0.0255   | [+0.0116, +0.0392] | 174/73       | 1.6e-10 *** |

`better/worse` counts cases that moved in each direction; the remainder tied. A CI excluding zero and a small p mean the improvement survives case-level noise.

Per-case rows read from stage: final, rescore. These can differ from section 2 — section 2 averages rounds of ordinary inference, while a `final` stage is the selected model re-scored with TTA. The two answer different questions and are expected to disagree in size.

> **⚠ Different stages on each side:** `centralized_3d_42`: rescore → final; `fedavg_3d_42`: rescore → final; `fedbn_3d_42`: rescore → final; `local_3d_42`: rescore → final

