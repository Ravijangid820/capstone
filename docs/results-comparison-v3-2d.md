# Baseline vs rerun — 2D

- **before:** `artifacts\baseline`
- **after:** `artifacts\runs\v3`
- **estimator (both sides):** mean of last 5 rounds
- **runs compared:** centralized_2d_42, local_2d_42, fedavg_2d_42, fedbn_2d_42

## 1. What changed

| Field               | before | after    |
|---------------------|--------|----------|
| aug_flip_p          | —      | 0.5      |
| aug_intensity_p     | —      | 0.0      |
| aug_intensity_scale | —      | 0.1      |
| aug_intensity_shift | —      | 0.1      |
| aug_noise_std       | —      | 0.0      |
| aug_rot90_p         | —      | 0.5      |
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
| centralized_2d_42 | 0.8582    | 0.8931   | +0.0349 | 0.8341    | 0.8718   | +0.0376 | 0.7959    | 0.8292   | +0.0333 |
| local_2d_42       | 0.8436    | 0.8661   | +0.0225 | 0.8228    | 0.8340   | +0.0112 | 0.7809    | 0.7973   | +0.0165 |
| fedavg_2d_42      | 0.8502    | 0.8533   | +0.0031 | 0.8274    | 0.8328   | +0.0054 | 0.7762    | 0.7870   | +0.0108 |
| fedbn_2d_42       | 0.8514    | 0.8653   | +0.0139 | 0.8315    | 0.8400   | +0.0086 | 0.7838    | 0.7940   | +0.0102 |

### Per-hospital WT Dice

| Run               | H1                        | H2                        | H3                        | H4                        |
|-------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| centralized_2d_42 | 0.8735 → 0.9046 (+0.0312) | 0.8780 → 0.9076 (+0.0296) | 0.8516 → 0.8943 (+0.0428) | 0.8299 → 0.8660 (+0.0361) |
| local_2d_42       | 0.8508 → 0.8832 (+0.0324) | 0.8638 → 0.8751 (+0.0114) | 0.8226 → 0.8604 (+0.0377) | 0.8371 → 0.8455 (+0.0084) |
| fedavg_2d_42      | 0.8800 → 0.8880 (+0.0080) | 0.8852 → 0.8925 (+0.0073) | 0.8601 → 0.8642 (+0.0040) | 0.7754 → 0.7683 (-0.0070) |
| fedbn_2d_42       | 0.8662 → 0.8852 (+0.0190) | 0.8654 → 0.8842 (+0.0188) | 0.8405 → 0.8502 (+0.0097) | 0.8334 → 0.8417 (+0.0083) |

## 3. Is the difference real? (paired, per case)

| Run               | Hospital | n   | Δ mean WT | 95% CI (bootstrap) | better/worse | Wilcoxon p  |
|-------------------|----------|-----|-----------|--------------------|--------------|-------------|
| centralized_2d_42 | H1       | 62  | +0.0392   | [+0.0245, +0.0558] | 55/7         | 2e-08 ***   |
| centralized_2d_42 | H2       | 62  | +0.0418   | [+0.0272, +0.0587] | 54/8         | 6.3e-09 *** |
| centralized_2d_42 | H3       | 62  | +0.0495   | [+0.0333, +0.0690] | 58/4         | 2.9e-11 *** |
| centralized_2d_42 | H4       | 62  | +0.0447   | [+0.0289, +0.0626] | 52/10        | 6.9e-08 *** |
| centralized_2d_42 | ALL      | 248 | +0.0438   | [+0.0358, +0.0525] | 219/29       | 1.8e-31 *** |
| local_2d_42       | H1       | 62  | +0.0433   | [+0.0291, +0.0591] | 53/9         | 2.6e-09 *** |
| local_2d_42       | H2       | 62  | +0.0146   | [+0.0040, +0.0248] | 45/17        | 0.00053 *** |
| local_2d_42       | H3       | 62  | +0.0206   | [+0.0088, +0.0339] | 47/15        | 9.6e-05 *** |
| local_2d_42       | H4       | 62  | +0.0031   | [-0.0069, +0.0123] | 42/20        | 0.03 *      |
| local_2d_42       | ALL      | 248 | +0.0204   | [+0.0142, +0.0268] | 187/61       | 3.5e-15 *** |
| fedavg_2d_42      | H1       | 62  | +0.0076   | [+0.0013, +0.0152] | 38/24        | 0.03 *      |
| fedavg_2d_42      | H2       | 62  | +0.0142   | [+0.0075, +0.0228] | 49/13        | 6.6e-06 *** |
| fedavg_2d_42      | H3       | 62  | +0.0309   | [+0.0185, +0.0456] | 54/8         | 2.3e-08 *** |
| fedavg_2d_42      | H4       | 62  | +0.0393   | [+0.0299, +0.0491] | 57/5         | 4.9e-10 *** |
| fedavg_2d_42      | ALL      | 248 | +0.0230   | [+0.0180, +0.0284] | 198/50       | 8.7e-23 *** |
| fedbn_2d_42       | H1       | 62  | +0.0251   | [+0.0157, +0.0361] | 51/11        | 1.8e-08 *** |
| fedbn_2d_42       | H2       | 62  | +0.0196   | [+0.0118, +0.0301] | 56/6         | 1.6e-08 *** |
| fedbn_2d_42       | H3       | 62  | +0.0189   | [+0.0103, +0.0280] | 48/14        | 9.2e-06 *** |
| fedbn_2d_42       | H4       | 62  | +0.0229   | [+0.0131, +0.0333] | 54/8         | 6.5e-07 *** |
| fedbn_2d_42       | ALL      | 248 | +0.0216   | [+0.0170, +0.0265] | 209/39       | 9.8e-25 *** |

`better/worse` counts cases that moved in each direction; the remainder tied. A CI excluding zero and a small p mean the improvement survives case-level noise.

Per-case rows read from stage: final, rescore. These can differ from section 2 — section 2 averages rounds of ordinary inference, while a `final` stage is the selected model re-scored with TTA. The two answer different questions and are expected to disagree in size.

> **⚠ Different stages on each side:** `centralized_2d_42`: rescore → final; `fedavg_2d_42`: rescore → final; `fedbn_2d_42`: rescore → final; `local_2d_42`: rescore → final

