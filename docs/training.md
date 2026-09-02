# Training

The research question, the four methods, the federated round loop, the evaluation protocol, and
the full v1→v5 record of what was tried and what it produced.

> Hospitals are keyed `H1`–`H4` in code and logs. In prose they are **Site A–D**, with
> **Site D = `H4` = the outlier**. `H1`–`H3` unqualified always mean **hypotheses**.
> See [conventions.md](conventions.md).

**Related:** [data.md](data.md) for the dataset and pipeline · [code.md](code.md) for the
implementation · [results/](results/) for the generated per-run reports.

---

## 1. The question

Federated learning lets hospitals train a shared model without pooling patient data. But real
hospitals are **non-IID**: different scanners, field strengths and acquisition protocols shift the
image distribution. A single global model is pulled toward the "average" hospital and can serve an
unusual site poorly.

> **Question.** When heterogeneity comes from per-hospital scanner differences, does keeping the
> normalization statistics local (FedBN) recover the worst-served hospital while keeping the
> average collaboration gain?

### The three hypotheses, pre-registered

| # | Claim | Test |
|---|---|---|
| **H1** | Collaboration beats going alone on average | `mean_dice(FedAvg) ≥ mean_dice(Local-only)` |
| **H2** | The single global model fails the outlier | `dice(FedAvg, Site D) < dice(Local-only, Site D)` |
| **H3** | Personalization recovers the outlier | `mean(FedBN) ≥ mean(FedAvg)` **and** `dice(FedBN, D) ≥ dice(FedAvg, D)` |

Stating these before the runs is what makes the answers meaningful. Two of the three did not come
out as expected.

---

## 2. The four methods

| Method | What is shared / kept local | Role |
|---|---|---|
| **Centralized** | not federated — one model on pooled data | **ceiling** reference; not deployable |
| **Local-only** ×4 | nothing shared; each site trains alone | **floor**; anything federated must beat this |
| **FedAvg** | **all** weights averaged each round | the global-model baseline (H1, H2) |
| **FedBN** | all weights averaged **except BatchNorm**, kept per site | the personalization under test (H3) |

**Why FedBN is the hypothesis.** Under a scanner shift the dominant mismatch between sites is in
the *feature statistics* BatchNorm captures. Averaging BN across four scanners produces statistics
that describe none of them. Keeping BN local lets each site normalize to itself while still sharing
everything the federation learned about tumour shape. In the 2D model that is **2,310 of 1,607,562
parameters — 0.14%**.

**Matched compute, deliberately.** Local-only and centralized train for `rounds × local_epochs`
total epochs — the same number a federated site sees across the whole run. Give local-only fewer
and H1 would merely measure FedAvg's longer training budget.

---

## 3. The round loop

One communication round: the server sends the current global weights out, each site trains locally
on its own shifted data, returns its weights, and the server aggregates. Sites run **sequentially**
on one GPU, so peak VRAM is one model regardless of how many hospitals there are.

```text
global_w = init_random(seed)            # identical init for every method
for r in 1..R:
    for h in hospitals:                 # sequential, shared GPU
        w = global_w + bn[h]            # fedbn: own BN; fedavg: pure global
        updates[h] = train(w, cache[h], epochs=E)
    if aggregate:
        global_w.update(weighted_average(updates, n_samples,
                                         skip=bn_keys if fedbn else {}))
        if fedbn: bn[h] = batchnorm_state(updates[h])   for each h
    else:
        own_w = updates                 # local / centralized keep their own full model
    evaluate(global_w + bn[h], test[h]) for each h       # → metrics.jsonl
```

All four methods are that one loop with two switches — which is why
[`federated.py`](../src/fedbrats/federated.py) has no per-method branches beyond a table:

| method | `aggregate` | `keep_bn_local` | `pooled` |
|---|---|---|---|
| `centralized` | no | – | **yes** |
| `local` | no | – | no |
| `fedavg` | **yes** | no | no |
| `fedbn` | **yes** | **yes** | no |

Averaging is weighted by each site's number of training samples (standard FedAvg).

### Three implementation traps

1. **State is `state_dict()`, not `parameters()`.** BN running statistics are *buffers*;
   `model.parameters()` excludes them. They are the entire mechanism of FedBN — miss them and
   FedBN silently degrades to FedAvg.
2. **`num_batches_tracked` is `int64`.** Every BN layer carries one. A weighted mean turns it into
   a float and `load_state_dict` rejects or corrupts it. Integer buffers are **copied from the
   highest-weighted client**, never averaged.
3. **Identify BN keys by module *type*, not by name.** MONAI emits keys like
   `net.model.0.conv.unit0.adn.N.bias` — there is no `"bn"` substring to match. We walk
   `named_modules()` for `nn.modules.batchnorm._BatchNorm`. 65 of the U-Net's 114 state keys are BN.

### Evaluate after aggregation, before local training

The order is fixed: `train → aggregate → evaluate → next round`. Evaluation must score the **true
federated model** — pure global for FedAvg, global body plus own BN for FedBN. Score *after* a
round of local training and FedAvg quietly gains a round of local adaptation, which is precisely
the personalization H2 claims it lacks. H2 would vanish for a purely procedural reason.

### The model

A dimension-parametric residual U-Net (MONAI), `num_res_units=2`, BatchNorm normalization.

| | 2D | 3D |
|---|---|---|
| Base channels | 32 | 16 |
| Parameters | **1,607,562** | **1,191,516** |
| BatchNorm affine | 2,310 (0.144%) | 1,158 (0.097%) |

**fp32 throughout.** Mixed precision was tested and rejected: it corrupts BatchNorm running
statistics on the strongly-shifted data. In a study whose central variable *is* the BatchNorm
layer, a speed-up that perturbs it is not a speed-up.

---

## 4. Evaluation protocol

Most of this project is about not fooling ourselves. Early on we found the round-to-round noise was
**larger than the effects under test**, and everything below exists because of that.

- **Metric.** Dice on WT / TC / ET, computed **per volume**, with the BraTS empty-ground-truth
  convention. 62 test volumes per site, **248 total**.
- **Scope.** The headline is the **diagonal** — each site scored by the model that serves it. A
  4×4 cross-hospital matrix is also run for local-only, to show what a site's own model does on
  everyone else's patients.
- **The estimator, fixed in advance.** A headline figure is the **mean of the final five rounds** —
  rounds 21–25 for v1–v3, 36–40 for v4–v5. **Never a single round.**
- **Paired statistics.** Comparisons are paired on the same case, with a bootstrap 95% CI over
  10,000 resamples and a Wilcoxon signed-rank test. `compare_runs.py` reports uncorrected p-values;
  `compute_significance.py` applies **Holm correction** within each family of per-site tests. Always
  label which.
- **The sign split is reported** — how many of the 62 volumes moved each way. A large mean driven
  by a handful of cases is a different claim from a consistent win.

### Why the estimator matters

At seed 123, a **single-round** reading says FedBN trails FedAvg by 0.098. The pre-registered
estimator on the same run says FedBN **leads** by 0.0376. Same data, opposite conclusions. That
finding is why the rest of the protocol exists.

---

## 5. The five iterations

| # | What it changed | Outcome |
|---|---|---|
| **v1** | The original study | Reference. Its flaw was in *scoring*, not training |
| **v2** | Cosine LR, augmentation (incl. intensity + noise), TTA, component filtering, validation split, best-val selection | **Partial regression** — broke FedAvg in 2D and 3 of 4 methods in 3D |
| **v3** | v2 minus intensity/noise augmentation; 150 → 230 training cases per site | **Success** — every method improved in both backbones |
| **v4** | v3 with 25 → 40 rounds, cosine stretched across all 40 | **Regression** — stopped after 1 run |
| **v5** | v3 with 40 rounds but cosine still annealing by round 25 | **Best** — every method improved again |

Two of five failed. Both are kept with their evidence, because each produced a finding.

**Constant across all five:** the committed split manifest, seed 42, `local_epochs=1`, base LR
1e-3, batch size 8 (2D) / 1 (3D), `base_channels` 32 (2D) / 16 (3D), 8 slices or 2 patches per
case, 192² crop / 96³ patch, `tumor_frac` 0.7, 5σ clipping, 62 test cases per site. Only the
training and inference *procedure* varied.

### Configuration, iteration by iteration

| Knob | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| `rounds` | 25 | 25 | 25 | **40** | **40** |
| `lr_schedule` | constant | cosine | cosine | cosine | cosine |
| `lr_anneal_rounds` | — | — | — | (40) | **25** |
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

### Why each iteration succeeded or failed

**v1 — the flaw was in scoring, not training.** Constant learning rate, no augmentation, headline
read off the final round. The plateau oscillated by more than the differences we wanted to measure.

**v2 — augmenting along the experimental variable.** Cosine LR, validation selection, TTA and
component filtering all helped. But the intensity jitter and Gaussian noise perturb gamma, bias and
blur — *the exact channels the scanner shift uses*. Augmenting along the experimental variable
dilutes the signal FedBN exists to capture and forces FedAvg's one global model into a worse
compromise. FedAvg was the only 2D method to lose ground, and in 3D — where 2 patches per case
makes an epoch small enough for noise to dominate — three of four methods went backwards.

**v3 — geometric augmentation only, plus the unused data.** Flips and rotations are
label-preserving and leave intensity statistics untouched, so they regularize without competing
with the thing under test. The training cap went from 150 to 230, using data that had been sitting
unread. Every method improved.

**v4 — a longer run needs its schedule re-tuned, not stretched.** All eight v3 runs were still
rising at the end (late-curve slope +0.0051 to +0.0110, with validation rising too), so we ran 40
rounds. But stretching the cosine over 40 rounds meant the model never consolidated at a low
learning rate. It lost 0.0099 against v3 and was stopped after one run. **The diagnosis was right
and the prescription was wrong.**

**v5 — decouple length from anneal rate.** 40 rounds, but the cosine still bottoms out at round 25;
the extra 15 rounds are spent at the learning-rate floor, consolidating. A head-to-head probe on 2D
FedBN scored 0.8756 test / 0.8514 validation against v3's 0.8653 / 0.8475 — both signals moving
together, which is what separates a real effect from test-set noise. A capacity probe
(`base_channels` 48) was rejected: +0.0048 on test but −0.0018 on validation.

---

## 6. Results

Pre-registered estimator, seed 42, no TTA — so these isolate the training recipe.

### 2D backbone — mean WT Dice across sites

| Method | v1 | v2 | v3 | v4 | **v5** | **v5 − v1** |
|---|---|---|---|---|---|---|
| Centralized *(ceiling)* | 0.8582 | 0.8835 | 0.8931 | 0.8832 | **0.8954** | **+0.0372** |
| Local-only *(floor)* | 0.8436 | 0.8583 | 0.8661 | — | **0.8699** | **+0.0263** |
| FedAvg | 0.8502 | 0.8444 | 0.8533 | — | **0.8629** | **+0.0127** |
| FedBN | 0.8514 | 0.8610 | 0.8653 | — | **0.8756** | **+0.0242** |

### 3D backbone — mean WT Dice across sites

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

### Site D — the table the project is about

Whole-tumour Dice at the outlier site, v5:

| Method | 2D | 3D |
|---|---|---|
| Centralized | 0.8688 | 0.8474 |
| Local-only | 0.8524 | 0.8387 |
| **FedAvg** | **0.7843** | **0.8300** |
| **FedBN** | **0.8447** | **0.8378** |

In 2D, FedAvg is the **best** method at Sites A–C and the **worst** at Site D. Joining the
federation left that hospital worse off than training alone by **0.0681**.

### Fairness — best site minus worst site, 2D at v5

| Method | Spread |
|---|---|
| Local-only | **0.0313** |
| Centralized | 0.0430 |
| FedBN | 0.0472 |
| **FedAvg** | **0.1152** |

Local-only is the *fairest* method — every site fits itself — but has the lowest mean. FedBN
approaches local-only's fairness at a much higher mean, which is the argument for personalization.

---

## 7. Statistical evidence

Paired per case across all 248 test volumes, v5 against v1 (v5 side includes TTA, so these measure
end-to-end gain rather than retraining alone):

| Method | Δ mean WT | 95% CI | moved ↑/↓ | p |
|---|---|---|---|---|
| Centralized 2D | +0.0510 | [+0.0420, +0.0609] | 232/16 | 3.1e−35 |
| Local-only 2D | +0.0245 | [+0.0184, +0.0311] | 199/49 | 3.4e−20 |
| FedAvg 2D | +0.0309 | [+0.0255, +0.0365] | 216/32 | 5.8e−34 |
| FedBN 2D | +0.0272 | [+0.0219, +0.0328] | 215/33 | 6.4e−28 |
| Centralized 3D | +0.0115 | [+0.0054, +0.0187] | 175/73 | 4.3e−07 |
| Local-only 3D | +0.0054 | [−0.0016, +0.0125] ⚠ | 161/86 | 2.2e−04 |
| FedAvg 3D | +0.0073 | [−0.0013, +0.0161] ⚠ | 155/92 | 2.0e−04 |
| FedBN 3D | +0.0343 | [+0.0235, +0.0457] | 183/64 | 8.4e−16 |

⚠ marks a confidence interval that includes zero — the mean gain is not backed by a consistent
per-case win. Those two are the weakest results in the set and are reported as such.

### FedBN vs FedAvg at v5, per site

| Backbone | Site | Δ | 95% CI | FedBN ↑/↓ |
|---|---|---|---|---|
| 2D | A | −0.0006 | [−0.0032, +0.0018] | 32/30 |
| 2D | B | −0.0073 | [−0.0122, −0.0027] | 23/39 |
| 2D | C | −0.0021 | [−0.0043, −0.0001] | 23/39 |
| **2D** | **D** | **+0.0637** | **[+0.0463, +0.0821]** | **59/3** |
| 3D | A | −0.0028 | [−0.0118, +0.0064] | 27/34 |
| 3D | B | −0.0018 | [−0.0062, +0.0034] | 25/37 |
| 3D | C | −0.0003 | [−0.0094, +0.0075] | 27/35 |
| **3D** | **D** | **+0.0134** | **[+0.0021, +0.0261]** | **39/23** |

At the three typical sites the difference is nil. At Site D it is the whole result — and in 2D it
is decisive (p = 4.5e−11). In 3D it is **not significant**: p = 0.066 uncorrected, **0.53 after
Holm correction**, with a near-even sign split.

---

## 8. Hypothesis verdicts at v5

| Hypothesis | 2D | 3D |
|---|---|---|
| **H1** — collaboration helps | ❌ **fails** (FedAvg 0.8629 vs Local 0.8699) | ✅ holds (0.8618 vs 0.8527) |
| **H2** — global model fails the outlier | ✅ holds (−0.0681 at Site D) | ✅ holds (−0.0087) |
| **H3** — personalization recovers it | ✅ holds, significant | ✅ inequality holds, **not significant** |

**H2 is the durable claim.** A single global model underperforms the outlier site in both
backbones once the recipe is sound.

**H1 flipping off in 2D is a finding, not a failure.** Under v1, FedAvg beat local-only. Under v5 —
with 230 training cases per site instead of 150 — it no longer does in 2D. More local data raised
the *floor* faster than federating raised the average. FedBN still clears it, which makes the case
for *personalized* federation rather than for federation as such.

**H3 differs by backbone in strength, not in sign.** In 2D FedBN beats FedAvg on the outlier
decisively. In 3D the inequality holds but the margin is not significant — which is itself a change
from v1, where FedAvg beat FedBN in 3D *significantly*. The defensible sentence is *"no significant
difference in 3D"*.

---

## 9. Two mechanisms

### Evaluation noise was the real obstacle — the methods contribution

The plateau moved more between adjacent rounds than the methods differed from each other. Cosine
annealing cut that oscillation **2.7–5.9×**. Combined with the pre-registered estimator, that is
what made every later comparison readable. This is arguably the project's most transferable result:
before optimizing a federated method, check that your measurement is quieter than the effect.

### The outlier effect is *not* caused by weight averaging — a correction

An earlier version of this study attributed Site D's degradation to FedAvg's averaging step, and
ordered the methods by "how much averaging each does". Both claims are withdrawn:

- **Centralized shows the same penalty**, and it has no aggregation step at all. The cause is
  optimizing over a **majority-dominated distribution**, not the averaging mechanism.
- The monotone ordering held at seed 42 and **failed to replicate at seed 7**.

Keeping BatchNorm local is what protects the minority site — which places FedBN's mechanism in
exactly the explanatory role its thesis claims, just for a different reason than first stated.

Also withdrawn: the explanation that FedBN's weaker 3D result came from 150 local cases being too
few for stable 3D BatchNorm statistics. A capacity probe was rejected on validation, and 3D
local-only — flat under v2 and v3 — improved as soon as the round budget rose to 40. The binding
constraint was **training length**, not sample size or model size.

Full list of retracted claims: [conventions.md §6](conventions.md#6-claims-that-are-retracted-or-restricted).

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

Per-stage costs: a training step is 48 ms; full-volume evaluation is 0.41 s per volume, so 248
volumes cost **1.7 min per round**. Preprocessing one case takes 2.3 s with 4 workers.

Two speed findings worth recording:

- **Evaluation, not training, dominated.** For 2D FedBN a round was 48% training, 52% evaluation.
  Raising inference batch size 8 → 64 (2D) and sliding-window batch 1 → 4 (3D) gave **2.7×** and
  **6.9×** on the forward pass, using 0.39 GB and 0.73 GB of a 4 GB card — the GPU had been idling
  through the most expensive phase. Per-case Dice changes by at most 1.5e−4.
- **A larger *training* batch would not help in 2D.** Training ran at 195 ms/step against a 48 ms
  GPU step, so ~75% is data loading. 3D is the opposite — genuinely GPU-bound at `batch_size=1` —
  and is where that headroom should go next.

---

## 11. Limitations — what the data will not support

1. **v3 and v5 changed more than one thing versus v1.** Both bundle the recipe change with +53%
   training data. The deltas are a combined effect and **must not** be written up as a clean
   ablation. Separating them costs one run: `--preset v5 --train-per-hospital 150`.
2. **v3, v4 and v5 are seed 42 only, in both backbones.** Only v1 and v2 have three 2D seeds. No
   iteration has 3D multi-seed coverage. **Any ± figure for v5 would be fabricated.**
3. **The H3-in-3D result rests on a single run** — the newest, most interesting and least
   replicated finding. Flag it as provisional.
4. **The full "3D reversal" claim is refuted.** Do not repeat it in its original form.
5. **The averaging-based outlier mechanism is retracted** (§9).
6. **Never quote a single-round number.** If one is unavoidable, state the round and its
   neighbours.
7. **Gains differ hugely by backbone** — 2D +0.020…+0.051 paired, 3D +0.005…+0.034. The recipe is
   not uniformly beneficial.
8. **Validation gains are sometimes smaller than test gains** (2D local +0.0009 val vs +0.0038
   test). Where they disagree, the test figure is the optimistic end.
9. **The shift is synthetic.** Gamma, bias field and blur are a defensible model of scanner
   variation, not a substitute for four real hospitals.
10. **No privacy guarantee.** No image leaves a site, but raw weights can leak information about
    training data and no differential privacy is implemented.
11. **Every test site was in training.** Generalization to an unseen hospital is unmeasured.

---

## 12. Reproducing

```bash
# the winning recipe, both backbones
uv run python scripts/run_matrix.py --dim 2d 3d --seed 42 --preset v5 --tag v5

# compare any iteration against the frozen baseline, same estimator on both sides
uv run python scripts/compare_runs.py --baseline artifacts/snapshots/v1 \
    --new artifacts/runs/v5 --dim 2d --seed 42 --select last-k --last-k 5

# hypothesis verdicts
uv run python scripts/analyze.py --dim 2d --runs-dir artifacts/runs/v5 --select last-k --last-k 5

# confirm nothing has drifted
uv run python scripts/freeze_baseline.py --verify --dest artifacts/snapshots/v5
uv run python scripts/check_docs.py
```

| Location | Contents |
|---|---|
| `artifacts/snapshots/v1/ … v5/` | every run, frozen, SHA-256 manifest + git commit |
| `<run>/metrics.jsonl` | one row per round × model × test set |
| `<run>/per_case.jsonl` | one row per **volume** — the input to every CI and p-value |
| `<run>/config.json` | every knob the run actually used, typed |
| `<run>/summary.json` | selected round and its validation score |
| [`results/`](results/) | generated comparison and significance reports |

Checkpoints are excluded from the snapshots — large, and regenerable from these logs plus the code.
They live in `artifacts/runs/<tag>/<run>/checkpoints/`.
