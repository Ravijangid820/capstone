# Baseline vs rerun — 3D

- **before:** `artifacts\baseline`
- **after:** `artifacts\rescore_tta_3d`
- **estimator (both sides):** selected model, final evaluation
- **runs compared:** centralized_3d_42, local_3d_42, fedavg_3d_42, fedbn_3d_42

## 1. What changed

| Field               | before | after |
|---------------------|--------|-------|
| postproc_min_voxels | 0      | 50    |
| tta                 | False  | True  |

## 2. Dice, before → after

| Run               | WT before | WT after | WT Δ    | TC before | TC after | TC Δ    | ET before | ET after | ET Δ    |
|-------------------|-----------|----------|---------|-----------|----------|---------|-----------|----------|---------|
| centralized_3d_42 | 0.8801    | 0.8834   | +0.0033 | 0.8358    | 0.8417   | +0.0058 | 0.7926    | 0.8062   | +0.0136 |
| local_3d_42       | 0.8515    | 0.8559   | +0.0045 | 0.7778    | 0.7810   | +0.0033 | 0.7508    | 0.7527   | +0.0019 |
| fedavg_3d_42      | 0.8589    | 0.8586   | -0.0002 | 0.8010    | 0.7994   | -0.0016 | 0.7699    | 0.7661   | -0.0037 |
| fedbn_3d_42       | 0.8340    | 0.8350   | +0.0009 | 0.7811    | 0.7821   | +0.0010 | 0.7461    | 0.7469   | +0.0008 |

### Per-hospital WT Dice

| Run               | H1                        | H2                        | H3                        | H4                        |
|-------------------|---------------------------|---------------------------|---------------------------|---------------------------|
| centralized_3d_42 | 0.8988 → 0.9010 (+0.0022) | 0.8956 → 0.8991 (+0.0035) | 0.8779 → 0.8872 (+0.0093) | 0.8479 → 0.8463 (-0.0016) |
| local_3d_42       | 0.8709 → 0.8724 (+0.0016) | 0.8719 → 0.8758 (+0.0039) | 0.8437 → 0.8470 (+0.0034) | 0.8194 → 0.8285 (+0.0091) |
| fedavg_3d_42      | 0.8623 → 0.8629 (+0.0005) | 0.8655 → 0.8628 (-0.0027) | 0.8593 → 0.8626 (+0.0033) | 0.8483 → 0.8464 (-0.0019) |
| fedbn_3d_42       | 0.8438 → 0.8435 (-0.0003) | 0.8625 → 0.8649 (+0.0024) | 0.7972 → 0.7982 (+0.0010) | 0.8327 → 0.8332 (+0.0005) |

## 3. Is the difference real? (paired, per case)

| Run               | Hospital | n   | Δ mean WT | 95% CI (bootstrap) | better/worse | Wilcoxon p  |
|-------------------|----------|-----|-----------|--------------------|--------------|-------------|
| centralized_3d_42 | H1       | 62  | +0.0022   | [-0.0006, +0.0050] | 42/20        | 0.00062 *** |
| centralized_3d_42 | H2       | 62  | +0.0035   | [-0.0002, +0.0083] | 47/15        | 0.0017 **   |
| centralized_3d_42 | H3       | 62  | +0.0093   | [+0.0028, +0.0192] | 47/15        | 6.7e-05 *** |
| centralized_3d_42 | H4       | 62  | -0.0016   | [-0.0067, +0.0031] | 36/26        | 0.85 ns     |
| centralized_3d_42 | ALL      | 248 | +0.0033   | [+0.0008, +0.0065] | 172/76       | 1.6e-07 *** |
| local_3d_42       | H1       | 62  | +0.0016   | [-0.0006, +0.0036] | 45/16        | 0.003 **    |
| local_3d_42       | H2       | 62  | +0.0039   | [+0.0016, +0.0063] | 48/14        | 2.1e-05 *** |
| local_3d_42       | H3       | 62  | +0.0034   | [-0.0002, +0.0076] | 48/14        | 0.001 **    |
| local_3d_42       | H4       | 62  | +0.0091   | [+0.0055, +0.0129] | 49/13        | 1.5e-06 *** |
| local_3d_42       | ALL      | 248 | +0.0045   | [+0.0030, +0.0061] | 190/57       | 2e-14 ***   |
| fedavg_3d_42      | H1       | 62  | +0.0005   | [-0.0017, +0.0028] | 38/23        | 0.15 ns     |
| fedavg_3d_42      | H2       | 62  | -0.0027   | [-0.0072, +0.0006] | 37/25        | 0.56 ns     |
| fedavg_3d_42      | H3       | 62  | +0.0033   | [+0.0007, +0.0065] | 42/20        | 0.0083 **   |
| fedavg_3d_42      | H4       | 62  | -0.0019   | [-0.0048, +0.0005] | 26/36        | 0.2 ns      |
| fedavg_3d_42      | ALL      | 248 | -0.0002   | [-0.0018, +0.0013] | 143/104      | 0.1 ns      |
| fedbn_3d_42       | H1       | 62  | -0.0003   | [-0.0023, +0.0016] | 37/24        | 0.15 ns     |
| fedbn_3d_42       | H2       | 62  | +0.0024   | [-0.0001, +0.0051] | 46/16        | 0.0021 **   |
| fedbn_3d_42       | H3       | 62  | +0.0010   | [-0.0022, +0.0043] | 38/23        | 0.12 ns     |
| fedbn_3d_42       | H4       | 62  | +0.0005   | [-0.0029, +0.0035] | 31/31        | 0.68 ns     |
| fedbn_3d_42       | ALL      | 248 | +0.0009   | [-0.0005, +0.0023] | 152/94       | 0.0022 **   |

`better/worse` counts cases that moved in each direction; the remainder tied. A CI excluding zero and a small p mean the improvement survives case-level noise.

Per-case rows read from stage: rescore. These can differ from section 2 — section 2 averages rounds of ordinary inference, while a `final` stage is the selected model re-scored with TTA. The two answer different questions and are expected to disagree in size.

