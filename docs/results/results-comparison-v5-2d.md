# Baseline vs rerun — 2D

- **before:** `artifacts\baseline`
- **after:** `artifacts\runs\v5`
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
| centralized_2d_42 | 0.8582    | 0.8954   | +0.0372 | 0.8341    | 0.8744   | +0.0403 | 0.7959    | 0.8310   | +0.0350 |
| local_2d_42       | 0.8436    | 0.8699   | +0.0263 | 0.8228    | 0.8432   | +0.0204 | 0.7809    | 0.8050   | +0.0241 |
| fedavg_2d_42      | 0.8502    | 0.8629   | +0.0127 | 0.8274    | 0.8427   | +0.0154 | 0.7762    | 0.7959   | +0.0197 |
| fedbn_2d_42       | 0.8514    | 0.8756   | +0.0242 | 0.8315    | 0.8463   | +0.0148 | 0.7838    | 0.8023   | +0.0185 |

> **⚠ The two sides were not scored the same way.** A delta across different estimators measures the estimators as much as the models.
>
> - `centralized_2d_42`: before = _last 5 rounds (up to 25)_, after = _last 5 rounds (up to 40)_
> - `local_2d_42`: before = _last 5 rounds (up to 25)_, after = _last 5 rounds (up to 40)_
> - `fedavg_2d_42`: before = _last 5 rounds (up to 25)_, after = _last 5 rounds (up to 40)_
> - `fedbn_2d_42`: before = _last 5 rounds (up to 25)_, after = _last 5 rounds (up to 40)_
>
> Give the baseline comparable rows with `scripts/rescore.py` (no retraining), or compare with `--select last-k`, which both sides can always satisfy.

### Per-hospital WT Dice

| Run               | H1                        | H2                        | H3                        | H4                        |
|-------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| centralized_2d_42 | 0.8735 → 0.9048 (+0.0313) | 0.8780 → 0.9118 (+0.0338) | 0.8516 → 0.8963 (+0.0448) | 0.8299 → 0.8688 (+0.0389) |
| local_2d_42       | 0.8508 → 0.8837 (+0.0329) | 0.8638 → 0.8779 (+0.0142) | 0.8226 → 0.8655 (+0.0429) | 0.8371 → 0.8524 (+0.0153) |
| fedavg_2d_42      | 0.8800 → 0.8936 (+0.0136) | 0.8852 → 0.8995 (+0.0143) | 0.8601 → 0.8740 (+0.0139) | 0.7754 → 0.7843 (+0.0089) |
| fedbn_2d_42       | 0.8662 → 0.8916 (+0.0254) | 0.8654 → 0.8919 (+0.0265) | 0.8405 → 0.8743 (+0.0337) | 0.8334 → 0.8447 (+0.0113) |

## 3. Is the difference real? (paired, per case)

| Run               | Hospital | n   | Δ mean WT | 95% CI (bootstrap) | better/worse | Wilcoxon p  |
|-------------------|----------|-----|-----------|--------------------|--------------|-------------|
| centralized_2d_42 | H1       | 62  | +0.0456   | [+0.0303, +0.0631] | 58/4         | 4.9e-10 *** |
| centralized_2d_42 | H2       | 62  | +0.0489   | [+0.0339, +0.0670] | 60/2         | 4.9e-11 *** |
| centralized_2d_42 | H3       | 62  | +0.0599   | [+0.0407, +0.0820] | 61/1         | 1e-11 ***   |
| centralized_2d_42 | H4       | 62  | +0.0496   | [+0.0307, +0.0714] | 53/9         | 1.9e-07 *** |
| centralized_2d_42 | ALL      | 248 | +0.0510   | [+0.0420, +0.0609] | 232/16       | 3.1e-35 *** |
| local_2d_42       | H1       | 62  | +0.0401   | [+0.0265, +0.0555] | 55/7         | 2.5e-09 *** |
| local_2d_42       | H2       | 62  | +0.0186   | [+0.0087, +0.0285] | 48/14        | 1.3e-05 *** |
| local_2d_42       | H3       | 62  | +0.0324   | [+0.0179, +0.0492] | 51/11        | 5.7e-07 *** |
| local_2d_42       | H4       | 62  | +0.0070   | [-0.0025, +0.0157] | 45/17        | 0.004 **    |
| local_2d_42       | ALL      | 248 | +0.0245   | [+0.0184, +0.0311] | 199/49       | 3.4e-20 *** |
| fedavg_2d_42      | H1       | 62  | +0.0133   | [+0.0073, +0.0207] | 47/15        | 5.6e-06 *** |
| fedavg_2d_42      | H2       | 62  | +0.0194   | [+0.0127, +0.0280] | 49/13        | 8e-08 ***   |
| fedavg_2d_42      | H3       | 62  | +0.0390   | [+0.0256, +0.0544] | 58/4         | 4.1e-11 *** |
| fedavg_2d_42      | H4       | 62  | +0.0518   | [+0.0410, +0.0631] | 62/0         | 7.6e-12 *** |
| fedavg_2d_42      | ALL      | 248 | +0.0309   | [+0.0255, +0.0365] | 216/32       | 5.8e-34 *** |
| fedbn_2d_42       | H1       | 62  | +0.0293   | [+0.0184, +0.0424] | 55/7         | 2.8e-09 *** |
| fedbn_2d_42       | H2       | 62  | +0.0300   | [+0.0201, +0.0425] | 59/3         | 4.5e-10 *** |
| fedbn_2d_42       | H3       | 62  | +0.0256   | [+0.0156, +0.0365] | 48/14        | 1.2e-06 *** |
| fedbn_2d_42       | H4       | 62  | +0.0238   | [+0.0141, +0.0345] | 53/9         | 7.6e-07 *** |
| fedbn_2d_42       | ALL      | 248 | +0.0272   | [+0.0219, +0.0328] | 215/33       | 6.4e-28 *** |

`better/worse` counts cases that moved in each direction; the remainder tied. A CI excluding zero and a small p mean the improvement survives case-level noise.

Per-case rows read from stage: final, rescore. These can differ from section 2 — section 2 averages rounds of ordinary inference, while a `final` stage is the selected model re-scored with TTA. The two answer different questions and are expected to disagree in size.

> **⚠ Different stages on each side:** `centralized_2d_42`: rescore → final; `fedavg_2d_42`: rescore → final; `fedbn_2d_42`: rescore → final; `local_2d_42`: rescore → final

