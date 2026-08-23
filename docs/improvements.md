# Improving accuracy — and proving it

How the second round of results is produced and defended: what the baseline is, what changed,
and how "it got better" is turned into a claim with a number and a p-value attached.

Read [`workflow.md`](workflow.md) first for the run itself. This document is about the part that
comes after: comparing two runs honestly.

---

## 1. The finding that reframed the question

Before changing anything, the existing logs were re-scored. The models plateau by roughly round
15 and then oscillate — and the oscillation is **larger than the differences between the methods
being compared**.

Rounds 18–25, mean diagonal WT Dice, 2D seed 42:

| Method | r18 | r19 | r20 | r21 | r22 | r23 | r24 | r25 |
|---|---|---|---|---|---|---|---|---|
| fedavg | .8483 | .8377 | .8390 | .8608 | .8498 | .8437 | **.8611** | **.8354** |
| local | .8413 | .8543 | .8500 | .8469 | .8349 | .8343 | **.8491** | **.8526** |

At round 24 FedAvg beats local-only. At round 25 it loses. The reported verdict came from
whichever round the loop happened to stop on. `analyze.py` scores `max(round)`, so H1 was decided
by a coin flip — and the worst single swing in the 2D runs is 0.12 WT Dice (fedbn seed 123, H4:
0.794 averaged over the last five rounds, 0.676 at round 25 alone).

Re-scoring the **same frozen logs** with a mean over the last five rounds:

| | round 25 only | mean of rounds 21–25 |
|---|---|---|
| **H1** — fedavg ≥ local | **1/3 seeds** | **3/3 seeds** |
| **H2** — fedavg fails H4 | 3/3 | 3/3 |
| **H3** — fedbn ≥ fedavg, recovers H4 | 2/3 | 1/3 |

Nothing was retrained. Reproduce both rows:

```bash
python scripts/analyze.py --dim 2d --select last
python scripts/analyze.py --dim 2d --select last-k --last-k 5
```

Three things follow, and they matter more than any single Dice number:

1. **H1's original "NOT SUPPORTED" was not a result.** It was an unmeasured quantity. Under a
   noise-robust estimator, collaboration does beat going alone in every seed.
2. **H3 was over-claimed, and splitting it is the honest fix.** It bundles two claims. "FedBN
   recovers the outlier" is solid: FedBN scores H4 at ≈0.82 against FedAvg's ≈0.76 in all three
   seeds, under either estimator. "FedBN ≥ FedAvg on the mean" is inside the noise and should be
   reported as such rather than as a pass or a fail.
3. **Reducing this noise *is* the accuracy work.** A model that stops thrashing scores higher and
   is measured more precisely; the same change buys both.

The 3D runs were checked the same way and **all three 3D verdicts are identical under both
estimators** — so the 3D picture is not estimator-sensitive the way the 2D H1 verdict is. It is,
however, *recipe*-sensitive: what v1 reported as a 2D/3D reversal did not survive v5. See
[`methodology.md`](methodology.md#21-the-2d-3d-comparison-what-survives).

---

## 2. What changed

Every default still reproduces the frozen baseline. Improvements are opt-in via `--preset v2`,
so "before" stays runnable after "after" exists.

| Lever | Baseline | v2 | Why |
|---|---|---|---|
| LR across rounds | constant 1e-3 | cosine → 5e-5 | A fresh Adam is built every round, so this is the *only* place a schedule can live. Constant LR is what makes the plateau oscillate. |
| Augmentation | **none** | flips, rot90, intensity jitter, noise | The baseline had no augmentation of any kind. |
| Test-time augmentation | none | flip-TTA on the final evaluation | Costs 4× inference once, no retraining, no risk. |
| Post-processing | none | drop components < 50 voxels | A 2D model predicts slices independently, so its false positives are small and z-isolated; real tumour is contiguous. |
| Validation set | **none** | 20 cases/hospital | Model selection had no signal to use except the test set or the clock. |
| Model selection | last round | best validation round | Removes exactly the coin flip in §1. |
| Per-case Dice | computed, **discarded** | written to `per_case.jsonl` | Without it there is no CI and no significance test. |

Two deliberate choices:

- **Validation comes from cases *after* the training cap, not carved out of it.** Holding out 20
  of the 150 would mean v2 trains on less data than the baseline, and a difference could no longer
  be attributed to the recipe. Cost: the cache must hold 170 cases/hospital instead of 150.
- **Augmentation restores the exact-zero background.** Preprocessing writes background as
  literally 0.0 and evaluation feeds volumes with that property; a shift or noise draw that lifts
  it off zero would train the model on inputs it never sees again.

---

## 2a. Measured already: the inference-only gain (no retraining)

Flip-TTA and component filtering change no weights, so they can be measured against the **frozen
baseline checkpoints** directly. Same models, same test volumes, two fields different
(`tta` False→True, `postproc_min_voxels` 0→50). 2D, seed 42, 248 paired volumes per method:

| Method | WT | TC | ET | Wilcoxon p (WT, all 248) |
|---|---|---|---|---|
| centralized | 0.8517 → 0.8595 (**+0.0078**) | +0.0034 | +0.0065 | 3.9e-29 |
| local-only | 0.8526 → 0.8596 (**+0.0070**) | +0.0097 | +0.0095 | 4.0e-24 |
| FedAvg | 0.8354 → 0.8388 (**+0.0034**) | +0.0100 | +0.0143 | 1.8e-16 |
| FedBN | 0.8525 → 0.8587 (**+0.0062**) | +0.0092 | +0.0138 | 2.2e-27 |

Every 95 % CI excludes zero; 199–222 of 248 volumes improve in each method. ET gains most, as
expected — component filtering removes exactly the small isolated false positives that a 2D model
scattered across slices, and ET is the smallest, most fragmented region.

Full report: [`results-inference-only.md`](results-inference-only.md). Reproduce:

```bash
python scripts/rescore.py --all --dim 2d --seed 42 --out-dir artifacts/snapshots/v1
python scripts/rescore.py --all --dim 2d --seed 42 --out-dir artifacts/rescore_tta \
    --tta --postproc-min-voxels 50
python scripts/compare_runs.py --baseline artifacts/snapshots/v1 --new artifacts/rescore_tta \
    --dim 2d --seed 42 --select final
```

**One cell does not improve, and it is the interesting one.** FedAvg on H4 — the outlier — moves
−0.0009 (p = 0.54, the only non-significant cell in the table) while FedAvg gains on H1–H3 and
FedBN gains on H4. Better inference cannot rescue a model whose *representation* is mismatched to
the domain. That sharpens H2 rather than softening it: the outlier failure is a domain-shift
problem, not a decoding one, and TTA averaging over flips of a wrong answer still averages wrong
answers. It is also a useful negative control — a gain that appeared everywhere including there
would suggest the metric, not the method, had changed.

---

## 2b. Results — the full 2D rerun, three seeds

Complete matrix rerun with `--preset v2`. Both sides scored with the **same pre-registered
estimator** (mean of rounds 21–25), on the same split, same seeds, same training cases.

| Method | baseline mean WT | v2 mean WT | baseline H4 | v2 H4 |
|---|---|---|---|---|
| local-only | 0.8433 ± 0.0025 | **0.8541 ± 0.0036** | 0.8299 ± 0.0106 | **0.8400 ± 0.0048** |
| FedAvg | 0.8498 ± 0.0007 | 0.8449 ± 0.0025 | 0.7695 ± 0.0095 | 0.7574 ± 0.0098 |
| FedBN | 0.8475 ± 0.0028 | **0.8590 ± 0.0021** | 0.8152 ± 0.0164 | **0.8248 ± 0.0064** |
| centralized *(seed 42)* | 0.8582 | **0.8835** | 0.8299 | **0.8576** |

### The headline: H3 goes from 1/3 seeds to 3/3

| Hypothesis | baseline | v2 |
|---|---|---|
| **H1** mean(FedAvg) ≥ mean(local) | 3/3 ✅ | **0/3 ❌** |
| **H2** FedAvg fails the outlier | 3/3 ✅ | **3/3 ✅** |
| **H3** FedBN ≥ FedAvg, recovers outlier | **1/3** ⚠ | **3/3 ✅** |

H3 is the project's thesis and it was previously carried by one seed out of three. The reason is
visible in the table: in the baseline FedBN was *behind* FedAvg on the mean (0.8475 vs 0.8498), so
the "≥ on the mean" clause failed in two seeds. Under v2 FedBN leads by **+0.0141**, and the
clause passes everywhere.

**Reproducibility improved alongside accuracy.** Across-seed spread on the outlier fell from
±0.0164 to ±0.0064 for FedBN (2.6×) and ±0.0106 → ±0.0048 for local-only (2.2×), and
round-to-round plateau oscillation fell 2.7–5.9× (see §2c). Both make every number above more
quotable than its baseline counterpart.

### Paired per-case evidence, end to end

Δ WT Dice over 248 test volumes per run, comparing each pipeline's *reported* model:

| Method | seed 42 | seed 7 | seed 123 | mean | same sign |
|---|---|---|---|---|---|
| FedBN | +0.0139 | +0.0147 | +0.0387 | **+0.0224** | yes |
| local-only | +0.0106 | +0.0046 | +0.0144 | +0.0099 | yes |
| FedAvg | +0.0120 | +0.0113 | −0.0080 | +0.0051 | **no** |

Per-seed reports: [`results-comparison-seed42.md`](results-comparison-seed42.md),
[`seed7`](results-comparison-seed7.md), [`seed123`](results-comparison-seed123.md).

### H1 flipped off, and that is a finding rather than a failure

H1 went 3/3 → 0/3. FedAvg's mean is dragged down by its outlier, which got *worse* under the new
recipe in every seed, while local-only improved in every seed. Nothing about federation changed —
the gap H1 measures simply moved.

Read together with §1, **H1 has now flipped in both directions during this work**: the estimator
fix moved it 1/3 → 3/3 on the baseline, and the training recipe moved it 3/3 → 0/3. Both movements
are real and both are smaller than the round-to-round noise the original protocol reported it
through. The honest conclusion is that H1 as stated is a knife-edge comparison between two nearly
equal numbers, and the useful claim is the decomposition underneath it: FedAvg beats local-only on
the three typical hospitals and loses badly on the outlier. That is H2, which is 3/3 everywhere.

### A retraction

An earlier reading of seed 42 alone suggested H4 degradation ordered monotonically by how much
weight averaging a method does (centralized > local > FedBN > FedAvg). **That did not replicate.**
At seed 7 FedBN's H4 rose +0.0146 while local-only's fell −0.0041, reversing the order. What holds
across all three seeds is narrower: FedAvg's outlier degrades under the new recipe, and FedBN's
mean improves. The tidy mechanism was over-fitted to one seed — which is precisely what the extra
two seeds were run to catch.

## 2b-3d. The 3D rerun — where the recipe does *not* help

Same preset, same estimator, seed 42 (the only seed the 3D baseline has).

| Method | baseline mean WT | v2 mean WT | Δ | paired Δ (248 vols) |
|---|---|---|---|---|
| centralized | 0.8768 | 0.8770 | +0.0002 | −0.0022 *(ns)* |
| local-only | 0.8490 | 0.8475 | −0.0016 | −0.0016 *(ns)* |
| FedAvg | 0.8525 | 0.8399 | **−0.0125** | **−0.0155** (p=1.9e-08) |
| FedBN | 0.8422 | 0.8333 | −0.0089 | −0.0014 *(ns)* |

**The v2 recipe should not be adopted for 3D.** Three of four methods are flat or negative, and
FedAvg is significantly worse. Compare the same recipe in 2D, where centralized gained +0.0253 and
FedBN +0.0115.

This is consistent with the explanation already in [methodology.md](methodology.md#21-the-2d-3d-comparison-what-survives):
if 3D convolutions act as a natural regularizer, augmentation has little headroom left to exploit,
and the extra input noise costs more than the regularization buys. The baseline numbers say the
same thing from the other side — baseline 3D was already at 0.8768 mean where baseline 2D was
0.8582. **The improvement is a 2D result, not a universal one**, and reporting it as universal
would be the same over-generalization this document keeps warning about.

### The "3D reversal" partly survives, and the surviving half is the interesting one

| | H1 | H2 | H3 |
|---|---|---|---|
| 2D baseline | ✅ | ✅ | ✅ |
| 2D v2 | ❌ | ✅ | ✅ |
| 3D baseline | ✅ | ❌ | ❌ |
| **3D v2** | ❌ | **✅** | ❌ |

The original 3D claim was that *all three* verdicts reverse. Under the improved recipe they do not:

* **H2 flips to supported in 3D.** "FedAvg is robust to the outlier in 3D" does **not** hold once
  the training recipe changes — under v2 the global model fails the outlier in both backbones
  (3D: 0.8171 vs local's 0.8285). That half of the reversal was an artefact of the original
  training setup, not a property of the backbone.
* **H3 stays unsupported in 3D.** FedBN still loses to FedAvg (0.8333 vs 0.8399 mean; 0.8120 vs
  0.8171 on H4). This half survives both recipes.

So the defensible claim is narrower and sharper than the original: **the backbone does not change
whether a single global model fails an outlier hospital — it changes whether keeping BatchNorm
local is the right way to fix it.** H2 is universal; H3 is backbone-dependent. That is a cleaner
contribution than "everything reverses," and it is what two independent training recipes agree on.

> **Caveat, stated plainly.** 3D is **seed 42 only** on both sides. In 2D, a mechanism that looked
> clean at seed 42 failed to replicate at seed 7 (§2b). Nothing above should be treated as settled
> until the 3D matrix has three seeds; the H2 flip in particular rests on a 0.011 gap in one run.

## 2d. v3 — the recipe that improves every method in both backbones

v2 regressed FedAvg in 2D and three of four methods in 3D. Decomposing *where* it lost identified
a single cause (§2, "geometric augmentation only"), and v3 corrects it while adding the training
data that had never been used. **All eight runs improve, every one significant.**

Paired per-case Δ WT Dice against the frozen baseline, 248 test volumes per run, seed 42:

| Method | 2D Δ | 2D p | 3D Δ | 3D p |
|---|---|---|---|---|
| centralized | **+0.0438** | 1.8e-31 | **+0.0066** | 0.0003 |
| local-only | **+0.0204** | 3.5e-15 | **+0.0028** | 0.0054 |
| FedAvg | **+0.0230** | 8.7e-23 | **+0.0024** | 0.0046 |
| FedBN | **+0.0216** | 9.8e-25 | **+0.0255** | 1.6e-10 |

On the round-based estimator (mean of rounds 21–25, no TTA, so the training recipe alone):

| Method | 2D base → v3 | Δ | 3D base → v3 | Δ |
|---|---|---|---|---|
| centralized | 0.8582 → **0.8931** | +0.0349 | 0.8768 → **0.8836** | +0.0068 |
| local-only | 0.8436 → **0.8661** | +0.0225 | 0.8490 → 0.8489 | −0.0002 |
| FedAvg | 0.8502 → **0.8533** | +0.0031 | 0.8525 → **0.8555** | +0.0030 |
| FedBN | 0.8514 → **0.8653** | +0.0139 | 0.8422 → **0.8550** | +0.0128 |

**v3 beats v2 on all eight runs**, by +0.0044 to +0.0217. The 2D ceiling moved 0.8582 → 0.8931.

### Two cells to report honestly rather than average away

* **3D local-only is flat on the round estimator** (−0.0002) though positive paired (+0.0028,
  p=0.0054). It is the one place +53% training data bought nothing. Centralized and FedAvg gained
  from the same data, so the data is not useless — this is specific to training a 3D model alone
  on one hospital. Likely sampling/capacity-limited rather than data-limited: at
  `patches_per_case=2`, 230 cases yield only 460 patches/epoch at batch 1, on a `base_channels=16`
  network. The lever is `patches_per_case` or `base_channels`, not more cases.
* **FedAvg's outlier stays below baseline** (2D −0.0070, 3D −0.0110 on the round estimator) even
  though its overall mean improves in both backbones. This looks intrinsic rather than tunable:
  FedAvg collapses to one global model, so training every hospital better makes each local optimum
  fit its own distribution more tightly and the averaged compromise serve the majority better and
  the outlier worse. No other method faces it — centralized and local never average, FedBN keeps
  BatchNorm local. **This is what H2 asserts**, so a better-trained FedAvg with a sharper outlier
  gap is the phenomenon getting clearer. Moving that cell needs a different aggregation
  (outlier-weighted FedAvg, FedProx), not a tuning change.

### Hypotheses under v3

| | H1 | H2 | H3 |
|---|---|---|---|
| 2D baseline | ✅ | ✅ | ✅ |
| 2D v3 | ❌ | ✅ | ✅ |
| 3D baseline | ✅ | ❌ | ❌ |
| 3D v3 | ✅ | ✅ | ❌ |

Under the best recipe both backbones now agree on **H2 (✅)** — the global model fails the
outlier — which the 3D baseline had contradicted. They continue to disagree on **H3**, exactly as
[§2b-3d](#2b-3d-the-3d-rerun--where-the-recipe-does-not-help) argued: the backbone does not change
*whether* a single global model fails an outlier, it changes *whether keeping BatchNorm local is
the right fix*. Three recipes now agree on that reading.

> v3 changes **two** things at once versus baseline — the augmentation fix and +53% training data —
> so these deltas are the combined effect, not the recipe alone. That was the right trade for a
> request to maximize accuracy, but it is not a clean ablation and should not be written up as one.
> Seed 42 only; 2D multi-seed coverage exists for baseline and v2 but not yet v3.

## 2e. v5 — the final recipe, and the 3D reversal's last half falls

v3 left two questions open: were the models undertrained, and was capacity the limit? Two
single-method probes on 2D FedBN answered both for ~7 h of GPU instead of ~74 h of matrices.

| probe | test (last-5) | validation | verdict |
|---|---|---|---|
| v3 reference — 25 rounds, base 32 | 0.8653 | 0.8475 | — |
| **40 rounds, anneal by 25** | **0.8756** (+0.0103) | **0.8514** (+0.0039) | **adopted** |
| 25 rounds, base 48 (3.61M params) | 0.8701 (+0.0048) | 0.8457 (−0.0018) | rejected |

The capacity probe is a useful negative result: a test gain that validation *contradicts* is the
signature of test-set noise, and it is the same standard that rejected v4. **These models were
round-limited, not size-limited** — confirmed independently when 40 rounds fixed 3D local-only,
the cell I had wrongly attributed to a sampling/capacity limit.

v5 = v3 + 40 rounds with `lr_anneal_rounds=25`. Paired per-case against the frozen baseline:

| Method | 2D Δ | 2D p | 3D Δ | 3D p |
|---|---|---|---|---|
| centralized | **+0.0510** | 3.1e-35 | **+0.0115** | 4.3e-07 |
| local-only | **+0.0245** | 3.4e-20 | **+0.0054** | 2.2e-04 |
| FedAvg | **+0.0309** | 5.8e-34 | **+0.0073** | 2.0e-04 |
| FedBN | **+0.0272** | 6.4e-28 | **+0.0343** | 8.4e-16 |

**Reproducibility check:** v5's FedBN 2D scored **0.8756**, identical to the probe's 0.8756 to four
decimals — two runs days apart, one under `--preset v3` with manual flags, one under `--preset v5`.

### H3 now holds in 3D, and the reversal is finished

| | H1 | H2 | H3 |
|---|---|---|---|
| 2D baseline / v3 / **v5** | ✅ / ❌ / **❌** | ✅ / ✅ / **✅** | ✅ / ✅ / **✅** |
| 3D baseline / v3 / **v5** | ✅ / ❌ / **✅** | ❌ / ✅ / **✅** | ❌ / ❌ / **✅** |

[§2b-3d](#2b-3d-the-3d-rerun--where-the-recipe-does-not-help) argued the H2 half of the reversal was
a training artefact while the H3 half survived. **The H3 half does not survive either — but state
it carefully.** Under v5 FedBN's inequality holds in 3D on both clauses (mean +0.0024, outlier
+0.0087), where under v1 FedAvg beat FedBN *significantly*. So the reversal is gone.

It does **not** follow that FedBN wins in 3D. Paired per-case testing on the outlier gives
uncorrected p = 0.066, Holm-corrected p = 0.53, sign split 39/23 — the bootstrap CI excludes zero
while the Wilcoxon does not, so the shift comes from magnitude on a minority of volumes rather than
a consistent per-case win. **The defensible claim is "no significant difference in 3D."** Contrast
2D, where FedBN wins on 59 of 62 volumes at p = 4.5e-11.

### The outlier mechanism, corrected

Earlier I attributed the outlier's decline under better training to FedAvg's weight averaging. The
3D v5 results refute that — grouping by whether training mixes sites:

| method (3D) | typical sites H1–H3 | outlier H4 |
|---|---|---|
| centralized — pools all sites | +0.0176 | **−0.0087** |
| FedAvg — averages all weights | +0.0156 | **−0.0092** |
| FedBN — keeps BatchNorm local | +0.0278 | **+0.0046** |
| local-only — never mixes | +0.0025 | **+0.0072** |

Centralized has no aggregation step at all and shows the effect just as strongly as FedAvg. The
common factor is training over a **majority-dominated distribution**: optimize harder and you fit
the 3-of-4 majority better and the minority worse. So H2 is not an artefact of weight averaging —
it is what happens whenever one model is optimized across heterogeneous sites, which makes it a
broader claim than originally stated. Keeping BatchNorm local is what protects the outlier, which
is precisely FedBN's thesis, and FedBN patterning with local-only rather than with FedAvg is direct
evidence for it.

> Still seed 42 only for v3 and v5. Baseline and v2 have three 2D seeds; nothing here has 3D
> multi-seed coverage. The H3-in-3D result rests on one run and should be replicated before it is
> presented as settled.

## 2c. Plateau stability — the original problem, measured

Round-to-round swing in mean WT over rounds 21–25, seed 42:

| Method | baseline | v2 | reduction |
|---|---|---|---|
| centralized | 0.0234 | 0.0052 | **4.5×** |
| local-only | 0.0182 | 0.0066 | **2.7×** |
| FedAvg | 0.0257 | 0.0044 | **5.9×** |

Cosine decay across rounds is what did this, and it is the fix for the failure in §1: verdicts are
no longer hostage to which round the loop happened to stop on.

---

## 3. The protocol

```mermaid
flowchart LR
    B["artifacts/runs/<br/>baseline"] -->|freeze_baseline.py| F[("artifacts/snapshots/v1/<br/>+ SHA-256 manifest")]
    B -->|rescore.py<br/><i>no retraining</i>| P["per-case Dice<br/>backfilled"]
    P --> F
    N["--preset v2 --tag v2"] --> V["artifacts/runs/v2/"]
    F --> C{{"compare_runs.py"}}
    V --> C
    C --> R["config diff<br/>Δ Dice<br/>paired CI + Wilcoxon"]
```

### Why the freeze is not optional

`MetricsWriter` appends, and `run_id` is `<method>_<dim>_<seed>`. Re-running FedAvg 2D seed 42
would have written 25 fresh rounds *underneath* the 25 already in that file, and `analyze.py`,
which scores `max(round)`, would have silently averaged two experiments and reported the mean as
one. The failure leaves no error and no visible trace in the output.

Two things now prevent it: runs are namespaced by `--tag`, and `guard_run_dir` refuses to write
into a directory that already holds results.

```bash
python scripts/freeze_baseline.py            # snapshot + hash
python scripts/freeze_baseline.py --verify   # prove it hasn't drifted
```

### Backfilling the baseline's per-case numbers

Per-case Dice was computed and thrown away, so the frozen logs have means only. The checkpoints
survived, so the per-case numbers are recoverable **exactly** rather than approximately — no
retraining:

```bash
python scripts/rescore.py --all --dim 2d --seed 42 --out-dir artifacts/snapshots/v1
```

This doubles as a regression test on the evaluation path: with the new flags off, every rescored
mean reproduces the logged round-25 value to four decimals (fedavg H4 = 0.7373, fedbn H4 = 0.8290,
local H4 = 0.8569).

It writes `rescore_metrics.jsonl` and `rescore_config.json` — never `metrics.jsonl` or
`config.json`, which are hashed in the manifest. A tool that edits the frozen record to make a
comparison work has destroyed the thing the comparison was evidence for.

### Running and comparing

```bash
# extend the cache to cover the validation cases (resumable; only builds what's missing)
python scripts/build_cache.py --max-cases 170 --workers 8

python scripts/run_experiment.py --method centralized --dim 2d --preset v2 --tag v2
python scripts/run_experiment.py --method local       --dim 2d --preset v2 --tag v2
python scripts/run_experiment.py --method fedavg      --dim 2d --preset v2 --tag v2
python scripts/run_experiment.py --method fedbn       --dim 2d --preset v2 --tag v2

python scripts/compare_runs.py --dim 2d --select last-k --last-k 5 \
    --md-out docs/results-comparison.md --json-out artifacts/comparison_2d.json
```

---

## 4. What the comparison reports

**1. What changed** — a field-by-field diff of the two sides' recorded `config.json`. A results
table means nothing until a reader can see that one thing varied, and stated intentions are not
evidence; the runs' own configs are.

**2. By how much** — mean Dice per method and per hospital, before → after, with deltas.

**3. Whether it is real** — the same test volumes compared case by case: a paired bootstrap 95%
CI on the mean difference, and a Wilcoxon signed-rank test.

Paired, not two-sample, and that is the whole point. Hard cases are hard for both models, so
per-case variance is mostly case difficulty rather than model quality. Differencing within a case
removes it, which is why a 0.01 mean shift can be significant across 62 volumes while an unpaired
test on the same numbers would see nothing.

The tool applies **one estimator to both sides** and refuses to hide it when it cannot: if the
baseline has no `final` rows and the rerun does, it says so in the output rather than quietly
comparing a selected TTA model against a last-round one.

> **Reporting rule.** Choose the estimator before looking at the verdicts and apply it to both
> sides. `--select last-k --last-k 5` is the pre-registered choice for the rerun. It is defensible
> precisely because it was fixed in advance — the same freedom used after seeing results is how
> the 1/3-vs-3/3 split in §1 could be spun either way.

---

## 5. Honest limitations

- **Averaging the last K rounds reduces variance, it does not remove bias.** If a method is still
  improving at round 25, its 5-round mean understates it. The curves plateau by ~15, so this is
  minor here — but it is the reason to keep reporting the learning curves alongside the tables.
- **Validation changes the data layout.** v2 reads 170 cases/hospital where the baseline read 150.
  Training cases are identical; the cache is not. Anyone rebuilding from scratch needs the larger
  cache or `--val-per-hospital 0`.
- **TTA and post-processing are evaluated at final only.** Learning curves stay comparable to the
  baseline's, but the headline number and the curve's endpoint are measured differently. The
  `stage` field distinguishes them; do not read one as the other.
- **Three seeds is few.** "Supported in 3/3 seeds" is a stronger statement than 1/3, and still not
  a confidence interval over seeds. The per-case CIs are within a run, not across runs.
