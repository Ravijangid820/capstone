# v5 results — what changed, and what it does to the headline claim

> Naming follows [conventions.md](../conventions.md): hospitals are **Site A–D**
> (`H1`–`H4` in code and logs, Site D = the outlier); `H1`–`H3` mean **hypotheses**.

Seed 42, both backbones, all four methods. Runs in `artifacts/runs/v5`; every number below comes
from a script in `scripts/` and can be regenerated with the commands in §7.

## 1. What v5 is

v5 = the v3 recipe (augmentation without intensity jitter or noise, 230 train cases/hospital,
`best_val` selection, TTA, component filtering) trained for **40 rounds while annealing the LR on
v3's 25-round timetable** (`rounds=40`, `lr_anneal_rounds=25`).

The reason it matters beyond the accuracy bump: **v5 is the first recipe run in both 2D and 3D.**
Before it, the only config-matched 2D/3D pair was the weak original baseline, so any 2D-vs-3D
statement was entangled with the training recipe. It no longer is.

## 2. Headline: the 2D/3D reversal does not survive v5

H3 is "FedBN recovers the shifted hospital". Paired per-case test on Site D (`H4` in the logs), WT Dice, FedBN − FedAvg,
62 volumes, Wilcoxon + paired bootstrap. Holm correction is applied within each report's own
family of per-hospital tests — 12 for the single-seed reports, 36 for the three-seed 2D ones —
so a p-value here is not comparable across rows as a raw number, only as a verdict:

| Recipe   | Dim | FedAvg | FedBN  | Δ (FedBN−FedAvg) | 95% CI             | p (Holm) | Verdict |
|----------|-----|--------|--------|------------------|--------------------|----------|---------|
| Baseline | 2D  | 0.7373 | 0.8290 | **+0.0917**      | [+0.0694, +0.1149] | 6.4e-10  | FedBN wins |
| Baseline | 3D  | 0.8483 | 0.8327 | **−0.0156**      | [−0.0271, −0.0043] | 0.0105   | FedAvg wins |
| v2       | 2D  | 0.7647 | 0.8313 | +0.0666          | [+0.0499, +0.0846] | 5.1e-10  | FedBN wins |
| **v5**   | 2D  | 0.7891 | 0.8528 | **+0.0637**      | [+0.0463, +0.0821] | 5.4e-10  | FedBN wins |
| **v5**   | 3D  | 0.8287 | 0.8421 | **+0.0134**      | [+0.0021, +0.0259] | 0.53     | no significant difference |

The reversal was real under the baseline recipe — both directions were significant. Under v5 it is
gone: FedBN is no longer worse than FedAvg in 3D, it is directionally *better*.

**Do not overclaim the v5 3D cell in the other direction.** The bootstrap CI excludes zero but the
Wilcoxon test does not reach significance even before correction (uncorrected p = 0.066; the sign
split is 39/23). The two disagree because the mean shift comes from magnitude on a minority of
cases rather than a consistent per-case win. The defensible sentence is *"no significant
difference in 3D"*, not *"FedBN wins in 3D"*.

## 3. Hypothesis verdicts under v5

| Hypothesis | 2D | 3D |
|---|---|---|
| **H1** — federation beats going alone on average | ❌ not supported (FedAvg 0.8629 < Local 0.8699) | ✅ supported (FedAvg 0.8618 > Local 0.8527) |
| **H2** — the global model underperforms on the outlier | ✅ supported | ✅ supported |
| **H3** — FedBN recovers the outlier | ✅ supported | ✅ supported |

There is still a dimension-dependent difference, but **it has moved from H3 to H1**. If the paper's
contribution is "the backbone changes which conclusion you reach", that claim survives — it just
attaches to a different hypothesis, and the mechanism has to be re-argued from scratch.

### Why H1 flips in 2D (Local − FedAvg, WT, paired)

*Hospitals are Site A–D; `H1`–`H4` in the logs. Site D is the outlier. See [conventions.md](../conventions.md).*

| Dim | Site A | Site B | Site C | Site D | Pooled |
|-----|--------|--------|--------|--------|--------|
| 2D  | −0.0076 ns | −0.0219 *** | −0.0023 ns | **+0.0749 ***** | +0.0108 ns |
| 3D  | −0.0123 *** | −0.0162 *** | −0.0202 *** | +0.0115 ns | −0.0093 (FedAvg better) |

In 2D, FedAvg is at least as good as local-only on all three typical hospitals; the pooled mean
only favours local because H4 alone swings it by +0.0749. So "federation doesn't help in 2D" is
really "federation doesn't help *at the shifted hospital*, and that one site drags the average" —
which is H2, restated. In 3D, FedAvg beats local-only on all three typical hospitals with
significance, and the pooled result follows.

## 4. Baseline → v5, per method (WT Dice, mean of last 5 rounds)

| Method | 2D before | 2D after | Δ | 3D before | 3D after | Δ |
|---|---|---|---|---|---|---|
| Centralized | 0.8582 | 0.8954 | +0.0372 | 0.8768 | 0.8879 | +0.0111 |
| Local-only  | 0.8436 | 0.8699 | +0.0263 | 0.8490 | 0.8527 | +0.0037 |
| FedAvg      | 0.8502 | 0.8629 | +0.0127 | 0.8525 | 0.8618 | +0.0094 |
| FedBN       | 0.8514 | 0.8756 | +0.0242 | 0.8422 | 0.8642 | +0.0220 |

Every method improves in both backbones. Caveat carried from the tool's own warning: the baseline
averages its last 5 of 25 rounds and v5 its last 5 of 40, so this column pair is not a pure
model-vs-model delta. The per-case paired sections of the comparison reports are, and they agree.

## 5. Caveats to state in the paper

1. **v5 is one seed (42).** The 2D H3 result replicates at 3 seeds under v2, but nothing in v5
   does. The 3D side has never had more than one seed.
2. **2D and 3D differ in capacity**, not only dimensionality: `base_channels` 32 vs 16 and batch
   size 8 vs 1, to fit 3D in memory. `aug_rot90_p` is also 0 in 3D by design (the three axes are
   anatomically distinct). Any "2D vs 3D" claim inherits these.
3. **The reversal claim as currently written in `methodology.md` §2.1 and
   `capstone_project_report.md` §9 is now contradicted by v5** and needs rewriting before
   submission.

## 6. Figures

Recipe-tagged, `artifacts/figures/<recipe>_<dim>_<seed>_<hospital>_<case>/`, each with
`mri.png`, `ground_truth.png`, `fedavg.png`, `fedbn.png`, a labelled `panel.png`, and a
`caption.json` recording provenance and how the case was chosen:

- `v5_2d_42_H4_BraTS2021_01163/` — FedAvg 0.832 → FedBN 0.868 WT
- `v5_3d_42_H4_BraTS2021_00104/` — FedAvg 0.949 → FedBN 0.955 WT
- `v2_2d_42_H4_BraTS2021_00727/`, `baseline_3d_42_H4_BraTS2021_00233/` — the earlier recipes

Cases are chosen by `--pick representative` (gap closest to the hospital median), not by best
result. `baseline_3d_42_H4_BraTS2021_00727/` is a hand-picked matched-case figure and is **not
representative of the 3D table** — don't use it as one.

## 7. Regenerating everything here

```bash
python scripts/compute_significance.py --dim 2d --runs artifacts/runs/v5 --regions wt tc et --pooled
python scripts/compute_significance.py --dim 3d --runs artifacts/runs/v5 --regions wt tc et --pooled
python scripts/compute_significance.py --dim 2d --runs artifacts/runs/v5 --a fedavg --b local --regions wt --pooled
python scripts/compare_runs.py --dim 2d --seed 42 --new artifacts/runs/v5
python scripts/compare_runs.py --dim 3d --seed 42 --new artifacts/runs/v5
python scripts/export_predictions.py --dim 2d --seed 42 --runs artifacts/runs/v5
python scripts/export_predictions.py --dim 3d --seed 42 --runs artifacts/runs/v5
```

Full tables: [2D](results-significance-2d-v5.md) · [3D](results-significance-3d-v5.md) ·
[baseline→v5 2D](results-comparison-v5-2d.md) · [baseline→v5 3D](results-comparison-v5-3d.md)
