# Baseline vs rerun — 2D

- **before:** `artifacts\rescore_tta`
- **after:** `artifacts\runs\v2`
- **estimator (both sides):** selected model, final evaluation
- **runs compared:** centralized_2d_42, local_2d_42, fedavg_2d_42, fedbn_2d_42

## 1. What changed

| Field            | before   | after    |
|------------------|----------|----------|
| aug_intensity_p  | 0.0      | 0.3      |
| aug_noise_std    | 0.0      | 0.02     |
| aug_rot90_p      | 0.0      | 0.5      |
| augment          | False    | True     |
| lr_schedule      | constant | cosine   |
| report_last_k    | 1        | 5        |
| rounds           | 20       | 25       |
| select_by        | last     | best_val |
| val_per_hospital | 0        | 20       |

## 2. Dice, before → after

| Run               | WT before | WT after | WT Δ    | TC before | TC after | TC Δ    | ET before | ET after | ET Δ    |
|-------------------|-----------|----------|---------|-----------|----------|---------|-----------|----------|---------|
| centralized_2d_42 | 0.8595    | 0.8906   | +0.0311 | 0.8380    | 0.8593   | +0.0213 | 0.8002    | 0.8172   | +0.0169 |
| local_2d_42       | 0.8596    | 0.8631   | +0.0035 | 0.8406    | 0.8394   | -0.0012 | 0.7964    | 0.8000   | +0.0036 |
| fedavg_2d_42      | 0.8388    | 0.8474   | +0.0086 | 0.8267    | 0.8263   | -0.0005 | 0.7779    | 0.7786   | +0.0007 |
| fedbn_2d_42       | 0.8587    | 0.8664   | +0.0077 | 0.8371    | 0.8420   | +0.0049 | 0.7930    | 0.7996   | +0.0067 |

> **⚠ The two sides were not scored the same way.** A delta across different estimators measures the estimators as much as the models.
>
> - `centralized_2d_42`: before = _final (rescore)_, after = _final (final)_
> - `local_2d_42`: before = _final (rescore)_, after = _final (final)_
> - `fedavg_2d_42`: before = _final (rescore)_, after = _final (final)_
> - `fedbn_2d_42`: before = _final (rescore)_, after = _final (final)_
>
> Give the baseline comparable rows with `scripts/rescore.py` (no retraining), or compare with `--select last-k`, which both sides can always satisfy.

### Per-hospital WT Dice

| Run               | H1                        | H2                        | H3                        | H4                        |
|-------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| centralized_2d_42 | 0.8768 → 0.9015 (+0.0247) | 0.8761 → 0.9071 (+0.0310) | 0.8526 → 0.8897 (+0.0371) | 0.8326 → 0.8643 (+0.0317) |
| local_2d_42       | 0.8570 → 0.8751 (+0.0181) | 0.8695 → 0.8688 (-0.0006) | 0.8508 → 0.8531 (+0.0023) | 0.8610 → 0.8553 (-0.0057) |
| fedavg_2d_42      | 0.8871 → 0.8763 (-0.0107) | 0.8896 → 0.8863 (-0.0033) | 0.8420 → 0.8621 (+0.0201) | 0.7365 → 0.7647 (+0.0282) |
| fedbn_2d_42       | 0.8743 → 0.8841 (+0.0098) | 0.8717 → 0.8779 (+0.0061) | 0.8549 → 0.8724 (+0.0174) | 0.8341 → 0.8313 (-0.0028) |

## 3. Is the difference real? (paired, per case)

| Run               | Hospital | n   | Δ mean WT | 95% CI (bootstrap) | better/worse | Wilcoxon p  |
|-------------------|----------|-----|-----------|--------------------|--------------|-------------|
| centralized_2d_42 | H1       | 62  | +0.0247   | [+0.0099, +0.0393] | 49/13        | 3.4e-06 *** |
| centralized_2d_42 | H2       | 62  | +0.0310   | [+0.0176, +0.0468] | 55/7         | 4.4e-08 *** |
| centralized_2d_42 | H3       | 62  | +0.0371   | [+0.0218, +0.0558] | 51/11        | 6.1e-08 *** |
| centralized_2d_42 | H4       | 62  | +0.0317   | [+0.0164, +0.0492] | 43/19        | 0.00012 *** |
| centralized_2d_42 | ALL      | 248 | +0.0311   | [+0.0235, +0.0394] | 198/50       | 8.9e-22 *** |
| local_2d_42       | H1       | 62  | +0.0181   | [+0.0055, +0.0305] | 38/24        | 0.0012 **   |
| local_2d_42       | H2       | 62  | -0.0006   | [-0.0126, +0.0103] | 36/26        | 0.37 ns     |
| local_2d_42       | H3       | 62  | +0.0023   | [-0.0108, +0.0162] | 29/33        | 0.96 ns     |
| local_2d_42       | H4       | 62  | -0.0057   | [-0.0164, +0.0034] | 34/28        | 0.92 ns     |
| local_2d_42       | ALL      | 248 | +0.0035   | [-0.0024, +0.0096] | 137/111      | 0.029 *     |
| fedavg_2d_42      | H1       | 62  | -0.0107   | [-0.0186, -0.0038] | 22/40        | 0.0027 **   |
| fedavg_2d_42      | H2       | 62  | -0.0033   | [-0.0094, +0.0026] | 28/34        | 0.43 ns     |
| fedavg_2d_42      | H3       | 62  | +0.0201   | [+0.0095, +0.0330] | 39/23        | 0.0012 **   |
| fedavg_2d_42      | H4       | 62  | +0.0282   | [+0.0198, +0.0369] | 52/10        | 3.8e-08 *** |
| fedavg_2d_42      | ALL      | 248 | +0.0086   | [+0.0039, +0.0134] | 141/107      | 0.0016 **   |
| fedbn_2d_42       | H1       | 62  | +0.0098   | [+0.0041, +0.0166] | 44/18        | 0.00031 *** |
| fedbn_2d_42       | H2       | 62  | +0.0061   | [-0.0007, +0.0141] | 41/21        | 0.021 *     |
| fedbn_2d_42       | H3       | 62  | +0.0174   | [+0.0070, +0.0308] | 43/19        | 0.00057 *** |
| fedbn_2d_42       | H4       | 62  | -0.0028   | [-0.0129, +0.0065] | 39/23        | 0.13 ns     |
| fedbn_2d_42       | ALL      | 248 | +0.0077   | [+0.0031, +0.0125] | 167/81       | 7.2e-08 *** |

`better/worse` counts cases that moved in each direction; the remainder tied. A CI excluding zero and a small p mean the improvement survives case-level noise.

Per-case rows read from stage: final, rescore. These can differ from section 2 — section 2 averages rounds of ordinary inference, while a `final` stage is the selected model re-scored with TTA. The two answer different questions and are expected to disagree in size.

> **⚠ Different stages on each side:** `centralized_2d_42`: rescore → final; `fedavg_2d_42`: rescore → final; `fedbn_2d_42`: rescore → final; `local_2d_42`: rescore → final

