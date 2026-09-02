# Experiments

The operational plan: what runs, how they're evaluated, and exactly how each hypothesis is measured.

> Hospitals are **Site A–D** in prose (`H1`–`H4` in code and logs; **Site D** is the outlier);
> `H1`–`H3` mean **hypotheses**. See [conventions.md](../conventions.md). Sections 4–5 quote raw log
> names inside their superseded v1 tables.

## 1. Experiment matrix

Run for the **2D** backbone first; repeat for **3D** if the feasibility spike passes.

| # | Run | Trains on | Produces |
|---|---|---|---|
| E0 | **Centralized** | pooled ~1000 train | ceiling reference |
| E1 | **Local-only** ×4 | each hospital's own train | floor (one model per hospital) |
| E2 | **FedAvg** | federated across 4 hospitals | global model (tests H1, H2) |
| E3 | **FedBN** | federated, BN kept local | personalized model (tests H3) |

All four share the **same committed split**, the same seed, and the same test sets — only the training
*procedure* differs, so the comparison is clean.

## 2. Evaluation protocol

```mermaid
flowchart TD
    M["a trained model"] --> T1["Hospital 1 test"]
    M --> T2["Site B test"]
    M --> T3["Site C test"]
    M --> T4["Site D test (outlier)"]
    T1 --> D["Dice WT / TC / ET<br/>per hospital"]
    T2 --> D
    T3 --> D
    T4 --> D
    D --> AVG["mean across hospitals"]
    D --> OUT["outlier score"]
```

- **Metric:** Dice on the three regions **WT** (whole tumor), **TC** (tumor core), **ET** (enhancing).
- **Per volume, never per slice.** For 2D we predict every axial slice, stack them into a volume, and
  score *that*. The mean of per-slice Dice is not the per-case Dice — it inflates scores, because
  empty slices score 1.0 for free.
- **Empty ground truth (matters for ET).** BraTS convention: empty prediction on empty GT scores 1.0;
  any false positive scores 0.0. Without this the ET column is meaningless.
- **Two views:** per-hospital Dice (for the outlier claims) and the mean across hospitals (for the average
  claims). Reported per method.
- **FedBN eval:** each hospital uses the shared weights + *its own* BN layers.
- **Timing:** score **after aggregation, before the next round's local training** — see
  [federated-learning](federated-learning.md).
- Every score is appended to `metrics.jsonl`; tables/plots are generated from that file.

### Scope — diagonal, plus a cross-matrix for local-only

The headline numbers are the **diagonal**: each hospital's model on its own test set
(`model_hospital == test_hospital`). That is all H1/H2/H3 require.

Additionally, at the **final round only**, every local-only model is scored on all four test sets — a
4×4 matrix. Its off-diagonal cells are direct evidence the synthetic shift creates a genuine domain
gap (Site D's model should collapse on Sites A–C). It is run once rather than per round, since it costs 16×
a diagonal evaluation.

## 3. How each hypothesis is measured

| Hyp. | Claim | Concrete test |
|---|---|---|
| **H1** | collaboration helps on average | `mean_dice(FedAvg) ≥ mean_dice(Local-only)` |
| **H2** | the global model fails the outlier | `dice(FedAvg, Site D) < dice(Local-only, Site D)` |
| **H3** | personalization recovers the outlier | `mean_dice(FedBN) ≥ mean_dice(FedAvg)` **and** `dice(FedBN, Site D) ≥ dice(FedAvg, Site D)` (closing the H2 gap) |

### At which round — and why the answer is not "the last one"

Each test above is an inequality between two numbers, so **which round supplies them is part of
the test**, not a detail. The curves plateau by roughly round 15 and then oscillate by more than
the gaps being tested: on 2D seed 42 FedAvg's mean WT is 0.8611 at round 24 and 0.8354 at round 25,
against local-only's 0.8491 and 0.8526. H1 is decided in opposite directions by adjacent rounds.

Scoring the last round alone therefore reports one draw from that oscillation. Averaging the final
five reduces the variance without touching a model, and on the frozen baseline it moves H1 from
supported in 1/3 seeds to 3/3 — same logs, same models, better estimator. Full evidence in
[improvements.md](improvements.md) §1.

```bash
python scripts/analyze.py --dim 2d --select last-k --last-k 5
```

Two consequences worth carrying into the write-up:

- **H3 should be reported as its two halves.** "FedBN recovers the outlier" holds in every seed
  under either estimator. "FedBN ≥ FedAvg on the mean" is inside the noise. Bundling them into one
  verdict lets a coin flip decide the fate of a real finding.
- **The estimator must be fixed before the verdicts are read**, and applied identically to
  anything it is compared against — the same latitude used afterwards would let either answer be
  justified. `scripts/compare_runs.py` enforces this across two runs; `analyze.py` does not.

The 3D runs were re-checked the same way and all three verdicts are estimator-independent, so the
3D picture does not rest on this choice — the 2D H1 verdict does. It is, however, *recipe*-
sensitive: what v1 reported as a 2D/3D reversal did not survive v5. See
[methodology.md](methodology.md#21-the-2d--3d-comparison--what-survives).

Centralized (E0) frames all of the above as "how close to the pooled ceiling did we get."

> **Matched compute — what makes H1 a real test.** FedAvg gives each hospital `R × E` local epochs.
> Local-only therefore also trains `R × E` epochs, so the *only* difference between them is
> aggregation. Had local-only trained for `E` epochs, H1 would be near-guaranteed and would merely be
> measuring FedAvg's ~R× larger training budget. Centralized likewise trains `R × E` epochs on the
> pooled set.

> **Identical init.** All four methods start from the same seeded random weights (`build_model` seeds
> `torch` before construction), so no comparison is confounded by initialization luck.

## 4. Results — 2D backbone, **v1 baseline only** (R=25, E=1, seed 42, 150 train/hospital)

> **⚠ Superseded numbers.** Sections 4 and 5 report the **v1 baseline**. They are kept because the
> gates and the protocol above were validated against them, but the verdicts changed under v2–v5.
> **Current results: [iteration-report.md](iteration-report.md).** Hospital columns below use the
> raw log names `H1`–`H4`; in prose these are Sites A–D — see [conventions.md](../conventions.md).

Run on the RTX 3050; `metrics.jsonl` per run under `artifacts/runs/`. Regenerate the verdicts with
`python scripts/analyze.py --dim 2d` and the figures with `python scripts/plot_results.py`.

### Mean Dice across hospitals (diagonal)

| Method | WT | TC | ET |
|---|---|---|---|
| Centralized (ceiling) | 0.852 | 0.835 | 0.794 |
| Local-only (floor) | 0.853 | 0.831 | 0.787 |
| FedAvg | 0.835 | 0.817 | 0.764 |
| FedBN | 0.852 | 0.828 | 0.779 |

### Per-hospital WT Dice (outlier = H4)

| Method | H1 | H2 | H3 | H4 (outlier) |
|---|---|---|---|---|
| Local-only | 0.848 | 0.863 | 0.842 | **0.857** |
| FedAvg | **0.883** | **0.884** | 0.838 | **0.737** |
| FedBN | 0.866 | 0.866 | 0.849 | **0.829** |

### Verdicts

| Hyp. | Test | Observed (WT) | Verdict |
|---|---|---|---|
| **H1** | mean(FedAvg) ≥ mean(Local) | 0.835 vs 0.853 | **not supported** |
| **H2** | dice(FedAvg, H4) < dice(Local, H4) | 0.737 vs 0.857 | **supported** |
| **H3** | mean(FedBN) ≥ mean(FedAvg) **and** dice(FedBN,H4) ≥ dice(FedAvg,H4) | 0.852 ≥ 0.835 and 0.829 ≥ 0.737 | **supported** |

**Reading of the result.** FedAvg *improves* the three typical hospitals over local-only
(H1 0.848→0.883, H2 0.863→0.884) — collaboration helps where scanners are alike — but **collapses on
the outlier** (H4 0.857→0.737). That single collapse drags the mean below local-only, so H1 fails *as
stated*, not because federation is useless but because one compromise global model cannot serve both
the cluster and the outlier. That is precisely the gap FedBN closes: keeping BatchNorm local **recovers
H4 (0.737→0.829)** while retaining the collaboration gains on the cluster, landing a mean (0.852) that
ties local-only and the centralized ceiling and beats FedAvg — **without pooling any data**. The
cross-hospital matrix corroborates the domain gap: the H1 model scores only 0.671 on H4, its worst cell.

Figures: `artifacts/figures/{learning_curves_wt_2d,per_hospital_wt_2d,outlier_h4_wt_2d}.png`.
## 5. Results — 3D backbone, **v1 baseline only** (R=25, E=1, seed 42, 150 train/hospital)

> **⚠ The 3D verdicts below were later refuted.** Under `--preset v5` both H2 and H3's inequalities
> hold in 3D. What v1 reported as a dimension-dependent reversal was an artefact of its training
> recipe. See [conventions.md](../conventions.md) §6.

### Mean Dice across hospitals (diagonal)

| Method | WT | TC | ET |
|---|---|---|---|
| Centralized (ceiling) | 0.880 | 0.836 | 0.793 |
| Local-only (floor) | 0.852 | 0.778 | 0.751 |
| FedAvg | 0.859 | 0.801 | 0.770 |
| FedBN | 0.834 | 0.781 | 0.746 |

### Per-hospital WT Dice (outlier = H4)

| Method | H1 | H2 | H3 | H4 (outlier) |
|---|---|---|---|---|
| Local-only | 0.871 | 0.872 | 0.844 | **0.819** |
| FedAvg | 0.862 | 0.866 | 0.859 | **0.848** |
| FedBN | 0.844 | 0.863 | 0.797 | **0.833** |

### Verdicts (3D)

| Hyp. | Test | Observed (WT) | Verdict |
|---|---|---|---|
| **H1** | mean(FedAvg) ≥ mean(Local) | 0.859 vs 0.852 | **supported** |
| **H2** | dice(FedAvg, H4) < dice(Local, H4) | 0.848 vs 0.819 | **not supported** |
| **H3** | mean(FedBN) ≥ mean(FedAvg) **and** dice(FedBN,H4) ≥ dice(FedAvg,H4) | 0.834 ≥ 0.859 and 0.833 ≥ 0.848 | **not supported** |

**Reading of the 3D result.** Unlike 2D, the 3D backbone results show that **FedAvg does not fail the outlier** (H2 is not supported). On the outlier (H4), FedAvg (0.848) outperforms local-only (0.819) by a wide margin, and FedAvg beats local-only on average (0.859 vs 0.852, supporting H1). Consequently, local personalization via FedBN is counterproductive (mean WT 0.834 vs FedAvg 0.859).

This divergence is explained by two factors:
1. **Data scarcity & overfitting in 3D:** Training a 3D model with only 150 local cases (batch size 1) is highly prone to overfitting. The collaborative pooling in FedAvg provides a strong regularizing effect that greatly improves performance across all clients, including the outlier.
2. **Poor local BN estimation:** Keeping BatchNorm layers local (FedBN) requires clients to estimate running statistics on small local datasets. In 3D, 150 samples are insufficient to robustly estimate these statistics, causing local BN layers to degenerate and degrade overall model performance.

Figures: `artifacts/figures/{learning_curves_wt_3d,per_hospital_wt_3d,outlier_h4_wt_3d}.png`.

## 6. 3D feasibility spike (gate before E0–E3 in 3D)

Before running the full 3D matrix, one measurement decides go/no-go:

- Train a single 3D U-Net on the T4; record VRAM, per-epoch time, and projected full-study wall-clock vs.
  Colab's session limit.
- **Pass →** run E0–E3 in 3D and add a "does the story hold in 3D?" comparison.
- **Fail →** report the spike numbers; 2D remains the deliverable.

Local probe (RTX 3050) already shows 3D *fits in memory* at 96³/128³; the spike is about *speed at scale*.
See [specs](specs.md) for the measured numbers.
