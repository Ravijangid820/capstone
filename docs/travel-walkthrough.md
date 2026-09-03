# FedBraTS travel walkthrough

This is the offline-friendly briefing for the capstone presentation and viva. It reflects the final **v5** results. Read it once in order, then use it as a speaking guide.

## The one-minute version

**This project tests whether personalized federated learning protects a hospital with an unusual MRI scanner without losing the benefit of collaboration.**

Brain-tumour MRI scans are privacy-sensitive, so hospitals cannot pool them. In federated learning, every hospital trains locally and shares model tensors rather than patient scans. The difficulty is that scanners differ: one global model can work well at common sites and harm an unusual one.

I simulate four hospitals from BraTS 2021 MRI data, give each a controlled scanner-style shift, and make Site D the outlier. I compare centralized training, local-only training, FedAvg, and FedBN.

**Headline:** in 2D at Site D, FedAvg scores **0.7843** whole-tumour Dice, local-only scores **0.8524**, and FedBN recovers to **0.8447**. Standard federation harms the outlier; local BatchNorm recovers most of that loss.

## Question and hypotheses

> When client data are non-IID because scanners differ, does keeping normalization local in FedBN recover the hospital a shared model serves worst?

Use **Site A–D** in speech. `H1`–`H4` are code/log labels; Site D is `H4`. Do not confuse these with hypotheses H1–H3.

| Hypothesis | Claim |
|---|---|
| H1 | FedAvg improves mean Dice over local-only: collaboration helps on average. |
| H2 | FedAvg is worse than local-only at Site D: a global model fails the outlier. |
| H3 | FedBN matches/beats FedAvg overall and recovers Site D. |

## Data and design

- **BraTS 2021:** 1,251 de-identified glioma MRI cases.
- Inputs: four co-registered modalities — FLAIR, T1, T1ce, T2.
- Outputs: three nested masks — whole tumour (WT), tumour core (TC), enhancing tumour (ET). They use independent sigmoids, not softmax, because the regions nest.
- Four deterministic site partitions: 313 / 313 / 313 / 312 cases.
- **62 held-out volumes per site, 248 total.** From v2 onward each site also has 20 validation cases.
- The committed split manifest is why every version compares the same patients.

BraTS is pooled, so it has no usable real hospital identities. The hospitals and network are therefore simulated, but the FL algorithm is real: a site's images and labels never enter another site's training loop.

### Scanner shift

Each site receives fixed nonlinear gamma contrast, a smooth spatial bias field, and blur. This survives per-case z-normalization; a simple brightness offset would not. Site D is deliberately strongest: gamma 1.85, bias amplitude 0.34, blur sigma 1.7. Site A is mild: 1.06 / 0.06 / 0.3.

**Defense sentence:** “Site D is a controlled distribution outlier, not a data-quantity outlier; all sites have nearly equal numbers of cases.”

## Pipeline and methods

The pipeline loads NIfTI volumes, maps labels to WT/TC/ET, applies the site shift, crops and normalizes, caches arrays, trains a MONAI residual U-Net, then computes Dice on reconstructed full volumes. Volume-level evaluation avoids the artificially easy Dice=1 scores from empty slices.

| Method | What it does | Role |
|---|---|---|
| Centralized | Pools all training data | Accuracy ceiling; privacy is not permitted |
| Local-only | One model per site | Collaboration floor |
| FedAvg | Averages every parameter | Standard global-FL baseline |
| FedBN | Shares body weights; keeps BatchNorm local | Personalized-FL method |

FedBN changes only what is uploaded: all non-BatchNorm model state is averaged, but BatchNorm affine values and running statistics remain at each site. In 2D this is only **2,310 of 1,607,562 parameters (0.14%)**. That small fraction matters because BatchNorm captures scanner-dependent feature means and variances.

The implementation recognizes BatchNorm by PyTorch module type, preserving parameters and buffers without fragile name matching.

## Training and statistics

- 2D U-Net: 32 base channels, batch size 8. 3D U-Net: 16 channels, batch size 1, constrained by memory.
- v5: 230 training cases/site, geometric augmentation, validation checkpoint selection, flip TTA, small-component filtering, 40 rounds.
- The cosine LR reaches its floor by round 25 while training continues to round 40. This corrects v4, whose schedule stayed too high for too long.
- Local-only and centralized receive matched total training exposure, so collaboration is not given extra compute.
- Primary estimator: **mean of the final five rounds**, chosen before interpreting results.
- Comparisons are paired per case. Reports include bootstrap 95% CI (10,000 resamples), Wilcoxon signed-rank tests, Holm correction, and better/worse sign split.

## Numbers to remember

Mean WT Dice across all four sites, seed 42, final-five-round estimator:

| Method | 2D v5 | 3D v5 |
|---|---:|---:|
| Centralized | **0.8954** | **0.8879** |
| Local-only | 0.8699 | 0.8527 |
| FedAvg | 0.8629 | 0.8618 |
| FedBN | **0.8756** | **0.8642** |

Site D is the table that matters:

| Method | 2D Site D | 3D Site D |
|---|---:|---:|
| Centralized | 0.8688 | 0.8474 |
| Local-only | 0.8524 | 0.8387 |
| FedAvg | 0.7843 | 0.8300 |
| FedBN | 0.8447 | 0.8378 |

### Hypothesis verdicts

| Hypothesis | 2D v5 | 3D v5 |
|---|---|---|
| H1 | Not supported: FedAvg 0.8629 < local 0.8699 | Supported: 0.8618 > 0.8527 |
| H2 | Supported | Supported |
| H3 | Supported | Directionally higher, **not statistically significant** |

For the central 2D comparison, FedBN improves Site D over FedAvg by **+0.0604** in the headline table. The paired analysis gives 95% CI **[+0.0463, +0.0821]**, p = **4.5e-11**, and **59/62** volumes improve.

In 3D, the Site D difference is +0.0078, but the Wilcoxon p is 0.066 before correction and 0.53 after Holm correction (39/23 sign split). **Never say “FedBN wins in 3D.” Say “there is no significant difference in 3D.”**

## What changed over five iterations

- **v1:** original baseline.
- **v2:** added improvements, but intensity augmentation perturbed the same channels used to simulate scanner differences; it regressed.
- **v3:** removed intensity/noise augmentation and used 230 cases/site with validation selection, TTA, and post-processing.
- **v4:** extended to 40 rounds but stretched cosine annealing over all 40; it did not consolidate well.
- **v5:** 40 rounds, but anneal reaches its floor by 25; this is the final recipe.

The early claim that 3D reversed all conclusions did not survive v5. The early explanation blaming unstable 3D BatchNorm statistics is not supported. Training recipe and budget were the meaningful change.

## Interpretation

The result is not “FedBN always wins.” It is that a single model optimized over a majority-dominated mixed distribution can fail the unusual site. Centralized training also shows an outlier penalty, so it is not solely an artefact of FedAvg averaging. FedBN shares tumour-shape learning while retaining a tiny site-specific normalization component.

The contribution is a reproducible controlled study showing:

1. A mean can hide a serious hospital-level failure.
2. Standard federation can make a distributional outlier worse off than training alone.
3. FedBN materially reduces that failure in 2D.
4. The 3D evidence is uncertain and should be reported conservatively.

## Limitations — say these proactively

1. **v5 is one seed (42).** Earlier 2D evidence has multi-seed support, but v5 and all 3D work lack multi-seed replication.
2. **2D versus 3D differs in more than dimension:** capacity, batch size, samples per epoch, and rotation behavior vary for memory reasons.
3. **Sites/scanners are simulated.** This isolates heterogeneity but does not prove the effect on a real multi-institution dataset.
4. **No formal privacy guarantee.** Images remain local, but model updates can leak information; there is no DP or secure aggregation.
5. **No unseen-site evaluation.** Every test site participates in training.
6. **Limited FL baselines.** FedProx is a natural next comparator.
7. **v3 is confounded.** It changes augmentation and training cap together, so it is not a clean ablation.

## Likely questions

**Why not pool the data?** Centralized is the ceiling, but it requires privacy-sensitive images to leave their hospital.

**Is this really federated learning?** Yes: each client trains only on its own partition and sends tensors, not image data. The multi-hospital deployment is simulated.

**Why make Site D an outlier?** Scanner shift is the controlled independent variable. Equal site sizes ensure it is not a sample-size effect.

**Why local BatchNorm?** Its statistics are scanner-sensitive. The claim is not that it solves all heterogeneity, only that it is a low-cost, motivated personalization point for this setting.

**Why full-volume Dice?** Slice-level scoring is inflated by empty slices; the clinical output is a 3D tumour volume.

**Does federation help?** It helps typical sites. In 2D, Site D's severe loss makes the four-site FedAvg mean lower than local-only; in 3D FedAvg beats local-only.

**Is 3D FedBN significant?** No. It is directionally higher but statistically inconclusive.

**Next work?** Multi-seed v5, leave-one-site-out tests, FedProx/personalized-FL baselines, and differential privacy or secure aggregation.

## Presentation order and demo

1. Privacy problem: data cannot be pooled.
2. Explain centralized ceiling and local-only floor.
3. Explain a federated round: local train, model upload, aggregate, broadcast.
4. Cover BraTS, modalities, nested masks, deterministic split, and scanner shift.
5. Contrast FedAvg and FedBN: “shared body, local normalization.”
6. Explain fair compute, volume Dice, final-five-round estimator, and paired tests.
7. Show the Site D table before the overall mean.
8. State the 2D result, then the cautious 3D result.
9. Finish with limitations and next steps.

The interactive deck is [artifacts/showcase/showcase.html](../artifacts/showcase/showcase.html). Left/Right navigates; `O` opens overview; `F` toggles fullscreen.

For the local server:

```powershell
uv run python scripts/demo_server.py --port 8010
```

Open `http://127.0.0.1:8010/showcase` for the deck and `http://127.0.0.1:8010/` for live inference. The standalone deck is safer: it needs no model checkpoints, cache, GPU, or network.

## Source material

Use these documents for full detail:

- [conventions.md](conventions.md) — safe wording and retracted/restricted claims.
- [data.md](data.md) — dataset, split, scanner shift, and preprocessing.
- [training.md](training.md) — complete experiment record and statistics.
- [code.md](code.md) — modules, command flow, and implementation detail.
- [results-v5-summary.md](results/results-v5-summary.md) — final result tables.
- [review/QUESTIONS.md](../review/QUESTIONS.md) — extended defence answers.

Before a presentation, verify the evidence with:

```powershell
uv run python scripts/check_docs.py
uv run python scripts/freeze_baseline.py --verify --dest artifacts/snapshots/v5
```

Use current `docs/` rather than `docs/archive/`: archive files preserve earlier drafts and superseded claims.
