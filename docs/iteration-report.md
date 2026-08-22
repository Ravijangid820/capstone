# Iteration report: v1 → v5

Complete record of the five training iterations — what each changed, what it produced, which two
failed and why, and what may and may not be claimed from the results.

Companions: [improvements.md](improvements.md) for the methodology narrative,
[experiments.md](experiments.md) for the evaluation protocol, [specs.md](specs.md) for
hyperparameters. Every figure below comes from run logs in `artifacts/snapshots/`, each frozen with
a SHA-256 manifest.

> **Naming collision — settle this before writing.** `H1–H4` are **hospitals**; `H1–H3` are
> **hypotheses**. They collide throughout the code and logs. In the paper, rename the hospitals
> (suggestion: **Site A–D**) and reserve H1–H3 for hypotheses. **Site D = H4** is the outlier
> carrying the strongest synthetic scanner shift, and nearly every result here is about it.

---

## 1. The five iterations at a glance

| # | Tag | What it changed | Outcome |
|---|---|---|---|
| **v1** | `baseline` | The original study | Reference. Its flaw was in *scoring*, not training |
| **v2** | `v2` | Cosine LR, augmentation (incl. intensity + noise), TTA, component filtering, validation split, best-val selection | **Partial regression** — broke FedAvg in 2D and 3 of 4 methods in 3D |
| **v3** | `v3` | v2 minus intensity/noise augmentation; 150 → 230 training cases per hospital | **Success** — every method improved in both backbones |
| **v4** | `v4` | v3 with 25 → 40 rounds, cosine stretched across all 40 | **Regression** — stopped after 1 run |
| **v5** | `v5` | v3 with 40 rounds but cosine still annealing by round 25 | **Best** — every method improved again, and H3 finally holds in 3D |

Two of five iterations failed. Both are kept with their evidence, because each produced a finding:
v2 showed that augmenting along the experimental variable is harmful, and v4 showed that a longer
run needs its schedule re-tuned rather than stretched.

**Constant across all five:** the committed split manifest, seed 42, `local_epochs=1`, base LR
1e-3, batch size 8 (2D) / 1 (3D), `base_channels` 32 (2D) / 16 (3D), 8 slices or 2 patches per
case, 192² crop / 96³ patch, `tumor_frac` 0.7, 5σ clipping, and 62 test cases per hospital (248
total). Only the training and inference *procedure* varied.

---

## 2. Configuration, iteration by iteration

| Knob | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| `rounds` | 25 | 25 | 25 | **40** | **40** |
| `lr_anneal_rounds` | — | — | — | (40) | **25** |
| `lr_schedule` | constant | cosine | cosine | cosine | cosine |
| `train_per_hospital` | 150 | 150 | **230** | 230 | 230 |
| `val_per_hospital` | 0 | 20 | 20 | 20 | 20 |
| `augment` | off | on | on | on | on |
| `aug_flip_p` / `aug_rot90_p` | — | 0.5 / 0.5 | 0.5 / 0.5 | 0.5 / 0.5 | 0.5 / 0.5 |
| `aug_intensity_p` | — | 0.3 | **0.0** | 0.0 | 0.0 |
| `aug_noise_std` | — | 0.02 | **0.0** | 0.0 | 0.0 |
| `tta` | off | on | on | on | on |
| `postproc_min_voxels` | 0 | 50 | 50 | 50 | 50 |
| `select_by` | last round | best val | best val | best val | best val |
| `eval_test_every` | 1 | 1 | 1 | 3 | 3 |
| `eval_batch_size` / `sw_batch_size` | 8 / 1 | 8 / 1 | 8 / 1 | 8 / 1 | **64 / 4** |

`rot90` applies in 2D only — the three 3D axes are anatomically distinct, so rotating between them
produces orientations no scanner emits. The config records `aug_rot90_p=0.0` for 3D so the recorded
configuration matches what actually ran.

---

## 3. Results — mean WT Dice across hospitals

Pre-registered estimator: mean over the final five rounds (21–25 for v1–v3, 36–40 for v4–v5).
Seed 42. No TTA in these figures, so they isolate the training recipe.

### 2D backbone

| Method | v1 | v2 | v3 | v4 | **v5** | **v5 − v1** |
|---|---|---|---|---|---|---|
| Centralized *(ceiling)* | 0.8582 | 0.8835 | 0.8931 | 0.8832 | **0.8954** | **+0.0372** |
| Local-only *(floor)* | 0.8436 | 0.8583 | 0.8661 | — | **0.8699** | **+0.0263** |
| FedAvg | 0.8502 | 0.8444 | 0.8533 | — | **0.8629** | **+0.0127** |
| FedBN | 0.8514 | 0.8610 | 0.8653 | — | **0.8756** | **+0.0242** |

### 3D backbone

| Method | v1 | v2 | v3 | **v5** | **v5 − v1** |
|---|---|---|---|---|---|
| Centralized *(ceiling)* | 0.8768 | 0.8770 | 0.8836 | **0.8879** | **+0.0111** |
| Local-only *(floor)* | 0.8490 | 0.8475 | 0.8489 | **0.8527** | **+0.0037** |
| FedAvg | 0.8525 | 0.8399 | 0.8555 | **0.8618** | **+0.0094** |
| FedBN | 0.8422 | 0.8333 | 0.8550 | **0.8642** | **+0.0220** |

v4 ran only 2D centralized before being stopped; 3D was never attempted.

### All three regions, v1 → v5

| | 2D WT | 2D TC | 2D ET | 3D WT | 3D TC | 3D ET |
|---|---|---|---|---|---|---|
| Centralized | +0.0372 | +0.0403 | +0.0350 | +0.0111 | +0.0169 | +0.0190 |
| Local-only | +0.0263 | +0.0204 | +0.0241 | +0.0037 | +0.0115 | +0.0111 |
| FedAvg | +0.0127 | +0.0154 | +0.0197 | +0.0094 | +0.0222 | +0.0152 |
| FedBN | +0.0242 | +0.0148 | +0.0185 | +0.0220 | +0.0205 | +0.0228 |

All 24 region × method × backbone cells improved.

---

## 4. Statistical evidence

Paired per-case comparison against the frozen v1 baseline: the same 248 test volumes scored by both
models, differenced within each case. Pairing removes case difficulty — the dominant variance
source, since a hard tumour is hard for every model — leaving the method effect.

### v5 vs v1

| Method | 2D Δ WT | 95% CI | Wilcoxon p | better/worse | 3D Δ WT | 95% CI | p |
|---|---|---|---|---|---|---|---|
| Centralized | **+0.0510** | [+.0420, +.0609] | 3.1e−35 | 232/16 | **+0.0115** | [+.0054, +.0187] | 4.3e−07 |
| Local-only | **+0.0245** | [+.0184, +.0311] | 3.4e−20 | 199/49 | **+0.0054** | [−.0016, +.0125] | 2.2e−04 |
| FedAvg | **+0.0309** | [+.0255, +.0365] | 5.8e−34 | 216/32 | **+0.0073** | [−.0012, +.0160] | 2.0e−04 |
| FedBN | **+0.0272** | [+.0219, +.0328] | 6.4e−28 | 215/33 | **+0.0343** | [+.0232, +.0458] | 8.4e−16 |

### v3 vs v1 (previous best, for reference)

| Method | 2D Δ | 3D Δ |
|---|---|---|
| Centralized | +0.0438 (1.8e−31) | +0.0066 (3.0e−04) |
| Local-only | +0.0204 (3.5e−15) | +0.0028 (5.4e−03) |
| FedAvg | +0.0230 (8.7e−23) | +0.0024 (4.6e−03) |
| FedBN | +0.0216 (9.8e−25) | +0.0255 (1.6e−10) |

Two 3D CIs under v5 cross zero while their p-values are small. That combination means the mean
shift is modest but the *direction* is consistent across cases. **Quote the CI alongside the p,
never the p alone.**

---

## 5. Per-hospital detail, v1 → v5

| | Site A (H1) | Site B (H2) | Site C (H3) | **Site D (H4, outlier)** |
|---|---|---|---|---|
| **2D** Centralized | 0.8735 → 0.9048 | 0.8780 → 0.9118 | 0.8516 → 0.8963 | 0.8299 → **0.8688** |
| **2D** Local-only | 0.8508 → 0.8837 | 0.8638 → 0.8779 | 0.8226 → 0.8655 | 0.8371 → **0.8524** |
| **2D** FedAvg | 0.8800 → 0.8936 | 0.8852 → 0.8995 | 0.8601 → 0.8740 | 0.7754 → **0.7843** |
| **2D** FedBN | 0.8662 → 0.8916 | 0.8654 → 0.8919 | 0.8405 → 0.8743 | 0.8334 → **0.8447** |
| **3D** Centralized | 0.8879 → 0.9018 | 0.8871 → 0.9025 | 0.8761 → 0.8998 | 0.8561 → **0.8474** |
| **3D** Local-only | 0.8692 → 0.8657 | 0.8497 → 0.8589 | 0.8457 → 0.8475 | 0.8315 → **0.8387** |
| **3D** FedAvg | 0.8568 → 0.8745 | 0.8617 → 0.8750 | 0.8521 → 0.8679 | 0.8392 → **0.8300** |
| **3D** FedBN | 0.8449 → 0.8760 | 0.8456 → 0.8742 | 0.8452 → 0.8689 | 0.8332 → **0.8378** |

---

## 6. Multi-seed coverage

Only v1 and v2 have three 2D seeds (42, 7, 123). Mean ± population SD:

| Method | v1 mean WT | v2 mean WT | v1 Site D | v2 Site D |
|---|---|---|---|---|
| Local-only | 0.8433 ± 0.0025 | **0.8541 ± 0.0036** | 0.8299 ± 0.0106 | **0.8400 ± 0.0048** |
| FedAvg | 0.8498 ± 0.0007 | 0.8449 ± 0.0025 | 0.7695 ± 0.0095 | 0.7574 ± 0.0098 |
| FedBN | 0.8475 ± 0.0028 | **0.8590 ± 0.0021** | 0.8152 ± 0.0164 | **0.8248 ± 0.0064** |

Worth a sentence in the paper: **reproducibility improved alongside accuracy** — FedBN's
across-seed spread on the outlier fell 2.6× (±0.0164 → ±0.0064), local-only's 2.2×.

**v3, v4 and v5 are seed 42 only, in both backbones.**

---

## 7. Hypothesis verdicts across iterations

| Configuration | H1 · federation helps on average | H2 · global model fails the outlier | H3 · personalization recovers it |
|---|---|---|---|
| 2D v1 | ✅ | ✅ | ✅ |
| 2D v2 | ❌ | ✅ | ✅ |
| 2D v3 | ❌ | ✅ | ✅ |
| **2D v5** | ❌ | ✅ | ✅ |
| 3D v1 | ✅ | ❌ | ❌ |
| 3D v2 | ❌ | ✅ | ❌ |
| 3D v3 | ❌ | ✅ | ❌ |
| **3D v5** | ✅ | ✅ | **✅** |

**H2 is the durable claim.** Supported in every configuration except the 3D baseline — and that
single exception disappears once the recipe is properly tuned. It is the most defensible result in
the study.

**H3 strengthened enormously.** In v1 (2D, three seeds) it held in only **1 of 3 seeds**, because
FedBN actually *trailed* FedAvg on the mean (0.8475 vs 0.8498). Under v5 FedBN leads FedAvg by
**+0.0128 on the mean and +0.0604 on the outlier** in 2D, and by +0.0024 / +0.0078 in 3D — where it
had never held before.

**H1 is a knife edge and should be reported as one.** It has flipped in both directions during this
work: the estimator change moved it 1/3 → 3/3 seeds on v1's own logs, and the training recipe moved
it 3/3 → 0/3. Both movements are smaller than v1's round-to-round noise. H1 compares two nearly
equal numbers; the durable claim beneath it is H2.

### The "3D reversal" claim must be rewritten

The original finding was that *all three* verdicts reverse in 3D, presented as a novel
contribution. Under a properly tuned recipe **neither the H2 nor the H3 half survives** — both now
hold in 3D. What survives is narrower and better supported:

> The backbone does not change *whether* a single global model fails an outlier site. Both
> backbones agree on that. The apparent reversal was an artefact of the original training setup.

---

## 8. Why each iteration succeeded or failed

### v1 — the flaw was in scoring, not training

`analyze.py` scored `max(round)`. But the curves plateau by ~round 15 and then oscillate by more
than the effects being compared:

```
2D seed 42, mean WT Dice, adjacent rounds
  FedAvg      round 24: 0.8611    round 25: 0.8354
  Local-only  round 24: 0.8491    round 25: 0.8526
  -> H1 decided in opposite directions by two consecutive rounds

Worst single swing (FedBN, seed 123, Site D)
  round 25 alone: 0.676    mean of rounds 21-25: 0.794    difference 0.118
```

Re-scoring the **same frozen logs** over the last five rounds — no retraining — moved H1 from
supported in 1 of 3 seeds to 3 of 3. That estimator was then fixed in advance and applied
identically to both sides of every later comparison.

### v2 — augmenting along the experimental variable

v2's intensity jitter and Gaussian noise perturb gamma, bias field and blur — **the same channels
the synthetic scanner shift uses.** That dilutes the signal FedBN's BatchNorm layers exist to
capture and forces FedAvg's single global model into a worse compromise. Where it lost, in 2D:

| Method | Sites A–C | Site D (outlier) |
|---|---|---|
| Centralized | +0.0245 | +0.0277 |
| Local-only | +0.0112 | +0.0101 |
| FedBN | +0.0121 | +0.0096 |
| **FedAvg** | **−0.0025** | **−0.0120** |

FedAvg — the only method reduced to a single global model — was the only loser, and lost most on
the outlier. In 3D, where 2 patches per case makes each epoch small enough for noise to dominate,
three of four methods regressed.

### v3 — geometric augmentation only, plus the unused data

Flips and rotations are label-preserving and leave intensity statistics untouched, so they
regularize without competing with the effect under test. v3 also raised training data from 150 to
230 cases per hospital — 40% of the available pool had never been used, and it costs almost nothing
in wall clock since evaluation dominates a round.

### v4 — a longer run needs its schedule re-tuned, not stretched

v3's runs were all still improving at round 25, so v4 extended to 40 rounds. It came back **worse**:
2D centralized fell to 0.8832 test / 0.8748 validation against v3's 0.8931 / 0.8830.

The inference was wrong. v3's late-curve rise was the **cosine anneal consolidating the model** as
LR fell to 5e-5 — not headroom from extra optimization steps. Stretching the same cosine across 40
rounds keeps LR high far longer (round 20: 5.4e-4 versus v3's 5.0e-5), so the model never gets that
consolidation until much later, by which point it has drifted. Stopped after one run.

### v5 — two probes, then the winner

Rather than spend ~74 h on two full matrices, both remaining hypotheses were tested with
single-method probes on 2D FedBN for ~7 h total:

| Probe | Test (last-5) | **Validation** | Verdict |
|---|---|---|---|
| v3 reference — 25 rounds, base 32 | 0.8653 | 0.8475 | — |
| **40 rounds, anneal by 25** | **0.8756** (+0.0103) | **0.8514** (+0.0039) | **adopted** |
| 25 rounds, `base_channels` 48 (3.61M params) | 0.8701 (+0.0048) | 0.8457 (**−0.0018**) | rejected |

The capacity probe is a useful negative result: a test gain that **validation contradicts** is the
signature of test-set noise, and validation is the signal that never sees the test set. It is the
same standard that rejected v4.

**These models were round-limited, not size-limited.** Confirmed independently when 40 rounds fixed
3D local-only — the one cell flat under both v2 and v3, which had been wrongly attributed to a
sampling or capacity limit.

**Reproducibility check:** v5's FedBN 2D scored **0.8756**, identical to the probe's 0.8756 to four
decimals — two runs days apart, one under `--preset v3` with manual flags, one under `--preset v5`.

---

## 9. Two mechanisms, one of them a correction

### Evaluation noise was the real obstacle (methods contribution)

Cosine LR annealing across rounds cut plateau oscillation substantially. Range of mean WT over the
final five rounds:

| | v1 | **v5** | reduction |
|---|---|---|---|
| 2D FedBN | 0.0344 | **0.0042** | 8.2× |
| 2D FedAvg | 0.0257 | **0.0050** | 5.1× |
| 2D Local-only | 0.0182 | **0.0035** | 5.2× |
| 2D Centralized | 0.0234 | **0.0130** | 1.8× |
| 3D FedBN | 0.0328 | **0.0064** | 5.1× |
| 3D FedAvg | 0.0191 | **0.0027** | 7.1× |

A fresh Adam is built every round — standard for FL, since optimizer state is not transmitted — so
a decaying *per-round* LR is the only place a schedule can live. Verdicts are no longer hostage to
which round the loop stops on.

### The outlier effect is not caused by weight averaging — correction

An earlier reading, based on 2D FedAvg alone, attributed the outlier's decline under better
training to FedAvg's weight averaging. **The 3D v5 results refute that.** Grouping by whether
training mixes sites:

| Method (3D) | Sites A–C | **Site D (outlier)** |
|---|---|---|
| Centralized — pools all sites | +0.0176 | **−0.0087** |
| FedAvg — averages all weights | +0.0156 | **−0.0092** |
| FedBN — keeps BatchNorm local | +0.0278 | **+0.0046** |
| Local-only — never mixes sites | +0.0025 | **+0.0072** |

Centralized has no aggregation step at all and shows the effect just as strongly as FedAvg. The
common factor is training over a **majority-dominated distribution**: optimize harder and you fit
the 3-of-4 majority better and the minority worse.

This makes H2 a *broader* claim than originally stated — not an artefact of FedAvg's aggregation,
but what happens whenever one model is optimized across heterogeneous sites. And it places
BatchNorm locality directly in the explanatory role FedBN's thesis claims: FedBN patterns with
local-only, not with FedAvg.

> A third mechanism was proposed and **retracted**. A monotone ordering of outlier degradation by
> "how much weight averaging a method does" looked clean at seed 42 but failed to replicate at seed
> 7, where FedBN's outlier rose while local-only's fell. It was over-fitted to one seed and must not
> appear in the paper.

---

## 10. Compute

Measured on one RTX 3050 Laptop (4 GB), fp32 throughout.

| Iteration | Scope | Wall clock |
|---|---|---|
| v1 | 2D matrix, seed 42 | 7.2 h |
| v2 | 2D × 3 seeds + 3D × 1 seed | ~52 h |
| v3 | 2D + 3D, seed 42 | 33 h |
| v4 | 1 run, then stopped | 8 h |
| Probes | 2 single-method runs | 7 h |
| v5 | 2D + 3D, seed 42 | ~44 h |

Two speed findings worth recording:

- **Evaluation, not training, dominated.** For 2D FedBN a round was 48% training, 52% evaluation.
  Raising inference batch size from 8 → 64 (2D) and sliding-window batch 1 → 4 (3D) gave **2.7×**
  and **6.9×** on the forward pass, using 0.39 GB and 0.73 GB of a 4 GB card — the GPU had been
  idling through the most expensive phase. Per-case Dice changes by at most 1.5e−4.
- **A larger *training* batch would not help in 2D.** Training ran at 195 ms/step against a 48 ms
  GPU step, so ~75% is data loading, and batching reads the same number of samples. 3D is the
  opposite — genuinely GPU-bound at `batch_size=1` — and is where that headroom should go next.

---

## 11. Limitations — what the data will not support

1. **v3 and v5 changed more than one thing versus v1.** Both bundle the recipe change with +53%
   training data. The deltas are a combined effect and must not be written up as a clean ablation.
   Separating them costs one run: `--preset v5 --train-per-hospital 150`.
2. **v3, v4 and v5 are seed 42 only, in both backbones.** Only v1 and v2 have three 2D seeds. No
   iteration has 3D multi-seed coverage. **Any ± figure for v5 would be fabricated.**
3. **The H3-in-3D result rests on a single run.** It is the newest and most interesting finding and
   the least replicated. Flag it as provisional.
4. **The full "3D reversal" claim is refuted** — do not repeat it in its original form.
5. **The averaging-based outlier mechanism is retracted** (§9).
6. **Never quote a single-round number.** If one is unavoidable, state the round and its
   neighbours' values.
7. **Gains differ hugely by backbone** — 2D +0.020…+0.051 paired, 3D +0.005…+0.034. The recipe is
   not uniformly beneficial and should not be described as such.
8. **Validation gains are sometimes smaller than test gains** (2D local +0.0009 val vs +0.0038
   test). Where they disagree in magnitude, the test figure is the optimistic end.

---

## 12. Reproducing

```bash
# the winning recipe, both backbones
python scripts/run_matrix.py --dim 2d 3d --seed 42 --preset v5 --tag v5

# compare any iteration against the frozen baseline, same estimator on both sides
python scripts/compare_runs.py --baseline artifacts/baseline --new artifacts/runs/v5 \
       --dim 2d --seed 42 --select last-k --last-k 5

# hypothesis verdicts
python scripts/analyze.py --dim 2d --runs-dir artifacts/runs/v5 --select last-k --last-k 5

# confirm the baseline has not drifted
python scripts/freeze_baseline.py --verify
```

| Location | Contents |
|---|---|
| `artifacts/baseline/` | v1, frozen, SHA-256 manifest, 42/42 verified |
| `artifacts/snapshots/{v2,v3,v4,v5,probe_*}/` | every later run, same treatment |
| `metrics.jsonl` | one row per round × model × test set |
| `per_case.jsonl` | one row per **volume** — the input to every CI and p-value |
| `summary.json` | selected round and its validation score |
| `docs/results-comparison-v5-{2d,3d}.md` | generated comparison reports |

Checkpoints are excluded from snapshots — large, and regenerable from these logs plus the code.
