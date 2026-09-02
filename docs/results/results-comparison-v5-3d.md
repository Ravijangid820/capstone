# Baseline vs rerun — 3D

- **before:** `artifacts\baseline`
- **after:** `artifacts\runs\v5`
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
| eval_batch_size     | —      | 64       |
| eval_test_every     | —      | 3        |
| lr_anneal_rounds    | —      | 25       |
| lr_min_factor       | —      | 0.05     |
| lr_schedule         | —      | cosine   |
| postproc_min_voxels | —      | 50       |
| report_last_k       | —      | 5        |
| rounds              | 25     | 40       |
| select_by           | —      | best_val |
| sw_batch_size       | —      | 4        |
| sw_overlap          | —      | 0.25     |
| train_per_hospital  | 150    | 230      |
| tta                 | —      | True     |
| val_per_hospital    | —      | 20       |

## 2. Dice, before → after

| Run               | WT before | WT after | WT Δ    | TC before | TC after | TC Δ    | ET before | ET after | ET Δ    |
|-------------------|-----------|----------|---------|-----------|----------|---------|-----------|----------|---------|
| centralized_3d_42 | 0.8768    | 0.8879   | +0.0111 | 0.8326    | 0.8496   | +0.0169 | 0.7911    | 0.8100   | +0.0190 |
| local_3d_42       | 0.8490    | 0.8527   | +0.0037 | 0.7718    | 0.7834   | +0.0115 | 0.7502    | 0.7613   | +0.0111 |
| fedavg_3d_42      | 0.8525    | 0.8618   | +0.0094 | 0.7922    | 0.8144   | +0.0222 | 0.7627    | 0.7779   | +0.0152 |
| fedbn_3d_42       | 0.8422    | 0.8642   | +0.0220 | 0.7896    | 0.8101   | +0.0205 | 0.7570    | 0.7798   | +0.0228 |

> **⚠ The two sides were not scored the same way.** A delta across different estimators measures the estimators as much as the models.
>
> - `centralized_3d_42`: before = _last 5 rounds (up to 25)_, after = _last 5 rounds (up to 40)_
> - `local_3d_42`: before = _last 5 rounds (up to 25)_, after = _last 5 rounds (up to 40)_
> - `fedavg_3d_42`: before = _last 5 rounds (up to 25)_, after = _last 5 rounds (up to 40)_
> - `fedbn_3d_42`: before = _last 5 rounds (up to 25)_, after = _last 5 rounds (up to 40)_
>
> Give the baseline comparable rows with `scripts/rescore.py` (no retraining), or compare with `--select last-k`, which both sides can always satisfy.

### Per-hospital WT Dice

| Run               | H1                        | H2                        | H3                        | H4                        |
|-------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| centralized_3d_42 | 0.8879 → 0.9018 (+0.0138) | 0.8871 → 0.9025 (+0.0154) | 0.8761 → 0.8998 (+0.0237) | 0.8561 → 0.8474 (-0.0087) |
| local_3d_42       | 0.8692 → 0.8657 (-0.0035) | 0.8497 → 0.8589 (+0.0091) | 0.8457 → 0.8475 (+0.0018) | 0.8315 → 0.8387 (+0.0072) |
| fedavg_3d_42      | 0.8568 → 0.8745 (+0.0177) | 0.8617 → 0.8750 (+0.0133) | 0.8521 → 0.8679 (+0.0158) | 0.8392 → 0.8300 (-0.0092) |
| fedbn_3d_42       | 0.8449 → 0.8760 (+0.0312) | 0.8456 → 0.8742 (+0.0286) | 0.8452 → 0.8689 (+0.0237) | 0.8332 → 0.8378 (+0.0046) |

## 3. Is the difference real? (paired, per case)

| Run               | Hospital | n   | Δ mean WT | 95% CI (bootstrap) | better/worse | Wilcoxon p  |
|-------------------|----------|-----|-----------|--------------------|--------------|-------------|
| centralized_3d_42 | H1       | 62  | -0.0014   | [-0.0090, +0.0071] | 35/27        | 0.81 ns     |
| centralized_3d_42 | H2       | 62  | +0.0070   | [-0.0072, +0.0260] | 43/19        | 0.033 *     |
| centralized_3d_42 | H3       | 62  | +0.0230   | [+0.0094, +0.0397] | 48/14        | 0.00022 *** |
| centralized_3d_42 | H4       | 62  | +0.0175   | [+0.0085, +0.0275] | 49/13        | 1.2e-05 *** |
| centralized_3d_42 | ALL      | 248 | +0.0115   | [+0.0054, +0.0187] | 175/73       | 4.3e-07 *** |
| local_3d_42       | H1       | 62  | -0.0001   | [-0.0141, +0.0133] | 34/27        | 0.46 ns     |
| local_3d_42       | H2       | 62  | -0.0067   | [-0.0226, +0.0059] | 39/23        | 0.33 ns     |
| local_3d_42       | H3       | 62  | +0.0077   | [-0.0016, +0.0168] | 43/19        | 0.0083 **   |
| local_3d_42       | H4       | 62  | +0.0208   | [+0.0036, +0.0381] | 45/17        | 0.008 **    |
| local_3d_42       | ALL      | 248 | +0.0054   | [-0.0016, +0.0125] | 161/86       | 0.00022 *** |
| fedavg_3d_42      | H1       | 62  | +0.0208   | [+0.0066, +0.0349] | 43/18        | 5e-05 ***   |
| fedavg_3d_42      | H2       | 62  | +0.0159   | [-0.0021, +0.0378] | 37/25        | 0.12 ns     |
| fedavg_3d_42      | H3       | 62  | +0.0122   | [-0.0008, +0.0283] | 42/20        | 0.013 *     |
| fedavg_3d_42      | H4       | 62  | -0.0196   | [-0.0392, -0.0035] | 33/29        | 0.46 ns     |
| fedavg_3d_42      | ALL      | 248 | +0.0073   | [-0.0012, +0.0160] | 155/92       | 0.0002 ***  |
| fedbn_3d_42       | H1       | 62  | +0.0366   | [+0.0236, +0.0512] | 46/15        | 1.5e-06 *** |
| fedbn_3d_42       | H2       | 62  | +0.0171   | [-0.0028, +0.0346] | 46/16        | 0.00081 *** |
| fedbn_3d_42       | H3       | 62  | +0.0740   | [+0.0428, +0.1095] | 49/13        | 7.1e-08 *** |
| fedbn_3d_42       | H4       | 62  | +0.0094   | [-0.0045, +0.0228] | 42/20        | 0.038 *     |
| fedbn_3d_42       | ALL      | 248 | +0.0343   | [+0.0232, +0.0458] | 183/64       | 8.4e-16 *** |

`better/worse` counts cases that moved in each direction; the remainder tied. A CI excluding zero and a small p mean the improvement survives case-level noise.

Per-case rows read from stage: final, rescore. These can differ from section 2 — section 2 averages rounds of ordinary inference, while a `final` stage is the selected model re-scored with TTA. The two answer different questions and are expected to disagree in size.

> **⚠ Different stages on each side:** `centralized_3d_42`: rescore → final; `fedavg_3d_42`: rescore → final; `fedbn_3d_42`: rescore → final; `local_3d_42`: rescore → final

