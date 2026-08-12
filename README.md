# Personalized Federated Learning for Brain Tumor Segmentation

**Capstone project.** Can *personalized* federated learning (FedBN) give hospitals with
unusual scanners a better tumor-segmentation model than either a single shared global
model (FedAvg) or training alone (local-only) — without hurting everyone else?

We simulate several "hospitals" from the **BraTS 2021** MRI dataset, make them
heterogeneous with a controlled per-hospital scanner shift, and compare local-only,
FedAvg, and FedBN on brain-tumor segmentation.

## Research question & hypotheses

> When client data is non-IID because each hospital's scanner differs, does keeping the
> normalization layers **local** (FedBN) recover the hospitals that a single global model
> serves worst, while preserving the collaboration gain on average?

- **H1 — collaboration helps on average.** FedAvg ≥ local-only in mean Dice across hospitals.
- **H2 — the global model fails outliers.** FedAvg underperforms local-only on the hospital(s)
  whose scanner deviates most from the federation.
- **H3 — personalization recovers outliers.** FedBN matches/beats FedAvg on average *and*
  closes the outlier gap from H2.

## Status

| Phase | State |
|---|---|
| 0. Data acquisition & prep | ✅ done — compressed + unzipped in Google Drive; unzipped locally on D: |
| 1. Docs & design | ✅ done — full structured doc set (see below) |
| 2. Model choice | ✅ decided — build **both** 2D & 3D (dimension-parametric, 2D first, 3D feasibility-gated) |
| 3. Hospital partition (4 hospitals, 1 outlier) | ✅ done — deterministic, verified, manifest committed |
| 4. Data pipeline — preprocess + cache | ✅ done — fp16 memmap cache, resumable, ~35 MB/case |
| 5. Model + train/eval loop | ✅ done — MONAI U-Net (2D/3D), per-volume Dice |
| 6. FL engine (centralized / local / FedAvg / FedBN) | ✅ done — custom sequential loop, smoke-tested |
| 7. Experiments — full 2D matrix | ✅ done — ran locally (RTX 3050), all four methods, R=25 |
| 8. Analysis (H1/H2/H3) → report | ✅ done — **H2 & H3 supported, H1 not** (see below) |
| 9. 3D feasibility spike → 3D matrix | ✅ done — full 3D matrix completed, hypothesis reversal observed |
| 10. *(optional)* NVIDIA FLARE port | ⬜ Linux/Colab only |

## Results (2D)

Full 2D matrix, R=25, seed 42, 150 train/hospital. Final-round WT Dice on each hospital's own test set:

| Method | mean | H1 | H2 | H3 | H4 (outlier) |
|---|---|---|---|---|---|
| Centralized (ceiling) | 0.852 | 0.866 | 0.868 | 0.844 | 0.828 |
| Local-only (floor) | 0.853 | 0.848 | 0.863 | 0.842 | **0.857** |
| FedAvg | 0.835 | 0.883 | 0.884 | 0.838 | **0.737** |
| FedBN | 0.852 | 0.866 | 0.866 | 0.849 | **0.829** |

- **H2 ✅** — FedAvg's one global model **fails the outlier** (H4 0.737 vs local 0.857).
- **H3 ✅** — FedBN **recovers the outlier** (0.737 → 0.829) and beats FedAvg on the mean, tying
  local-only and the centralized ceiling **without pooling data** — the headline result.
- **H1 ❌ (as stated)** — mean(FedAvg) 0.835 < mean(local) 0.853. FedAvg *helps* the three typical
  hospitals but the outlier's collapse drags the average down. Not "federation is useless" — it is the
  motivation for FedBN, which then delivers.

> **⚠ These are single-final-round numbers and the verdicts above are not stable.** The curves
> plateau by ~round 15 then oscillate by more than the gaps being tested — FedAvg's mean WT is
> 0.861 at round 24 and 0.835 at round 25. Scoring the last five rounds instead, on the same logs,
> flips H1 to supported in 3/3 seeds. See **Results (2D, improved)** below for the superseding run.

Details and figures: [experiments.md](docs/experiments.md#4-results--2d-backbone-r25-e1-seed-42-150-trainhospital) · `artifacts/figures/`. Regenerate: `python scripts/analyze.py --dim 2d`.

## Results (2D, improved recipe — supersedes the table above)

Full rerun with `--preset v2` (cosine LR across rounds, augmentation, flip-TTA, component
filtering, validation-based checkpoint selection). Three seeds, same split and same training cases
as the baseline, both sides scored with the same pre-registered estimator (mean of rounds 21–25):

| Method | baseline mean WT | **v2 mean WT** | baseline H4 | **v2 H4** |
|---|---|---|---|---|
| Local-only (floor) | 0.8433 ± 0.0025 | **0.8541 ± 0.0036** | 0.8299 ± 0.0106 | **0.8400 ± 0.0048** |
| FedAvg | 0.8498 ± 0.0007 | 0.8449 ± 0.0025 | 0.7695 ± 0.0095 | 0.7574 ± 0.0098 |
| **FedBN** | 0.8475 ± 0.0028 | **0.8590 ± 0.0021** | 0.8152 ± 0.0164 | **0.8248 ± 0.0064** |
| Centralized *(seed 42)* | 0.8582 | **0.8835** | 0.8299 | **0.8576** |

| Hypothesis | baseline | v2 |
|---|---|---|
| H1 — collaboration helps on average | 3/3 ✅ | **0/3 ❌** |
| H2 — the global model fails the outlier | 3/3 ✅ | **3/3 ✅** |
| **H3 — personalization recovers the outlier** | **1/3** ⚠ | **3/3 ✅** |

- **H3 — the project's thesis — goes from 1/3 seeds to 3/3.** In the baseline FedBN trailed FedAvg
  on the mean (0.8475 vs 0.8498), so the "≥ on the mean" clause failed in two seeds. It now leads
  by +0.0141 and passes in all three.
- **Paired over 248 test volumes per run**, FedBN improves in every seed (+0.0139, +0.0147,
  +0.0387; all CIs exclude zero).
- **H1 flips off** because FedAvg's outlier degrades further while local-only improves. H1 has now
  flipped in *both* directions during this work, which is itself the finding: it is a knife-edge
  between two nearly equal numbers, and the durable claim underneath it is H2.
- **Reproducibility improved too** — across-seed spread on H4 fell 2.6× for FedBN, and
  round-to-round plateau oscillation fell 2.7–5.9×.

Full write-up, protocol, and the frozen "before" snapshot: **[improvements.md](docs/improvements.md)**.
Regenerate: `python scripts/compare_runs.py --dim 2d --seed 42 --select last-k --last-k 5`.

## Results (3D)

Full 3D matrix, R=25, seed 42, 150 train/hospital. Final-round WT Dice on each hospital's own test set:

| Method | mean | H1 | H2 | H3 | H4 (outlier) |
|---|---|---|---|---|---|
| Centralized (ceiling) | 0.880 | 0.899 | 0.896 | 0.878 | 0.848 |
| Local-only (floor) | 0.851 | 0.871 | 0.872 | 0.844 | 0.819 |
| FedAvg | 0.859 | 0.862 | 0.866 | 0.859 | **0.848** |
| FedBN | 0.834 | 0.844 | 0.862 | 0.797 | **0.833** |

- **H1 ✅** — FedAvg mean (0.859) > local mean (0.851): federation helps on average.
- **H2 ❌** — FedAvg on H4 (0.848) ≥ local on H4 (0.819): the global model does **not** fail the outlier.
- **H3 ❌** — FedBN on H4 (0.833) < FedAvg on H4 (0.848): personalization does **not** recover — it's worse.

**Hypothesis reversal vs. 2D:** 3D spatial convolutions act as a natural regularizer,
making FedAvg robust to scanner shift even on the outlier hospital. FedBN actually
suffers in 3D because 150 local cases per hospital are insufficient to estimate stable
3D batch-normalization running statistics — the higher-dimensional feature maps amplify
the variance, so keeping BN layers local becomes a liability rather than an advantage.

All three 3D verdicts survive the last-5-round **estimator** unchanged (the 3D curves are steadier —
H4 spans 0.008 over rounds 21–25 against swings up to 0.12 in 2D). They do **not** all survive a
change of training **recipe**:

> **⚠ Half the reversal is an artefact of the training setup.** Rerunning 3D with `--preset v2`
> flips **H2 to supported** — the global model *does* fail the outlier in 3D (0.8171 vs local's
> 0.8285), contrary to the claim above. **H3 stays unsupported**, so FedBN really does underperform
> in 3D under both recipes. The defensible claim is therefore narrower: the backbone does not change
> *whether* a global model fails an outlier, it changes *whether keeping BatchNorm local fixes it*.
> Seed 42 only — see [improvements.md](docs/improvements.md).

The v2 recipe also **should not be adopted for 3D**: three of four methods are flat or worse, and
FedAvg drops significantly (−0.0155 paired, p=1.9e-08). The accuracy gains are a 2D result.

Details and figures: `artifacts/figures/`. Regenerate: `python scripts/analyze.py --dim 3d`.

## Web Demo

An interactive web dashboard lets you visualize and compare segmentations across all four FL methods in real time.

```bash
uv run python scripts/demo_server.py
# Open http://localhost:8000
```

Features:
- **2D slice viewer** — browse any axial slice across FLAIR, T1, T1ce, T2 modalities
- **3D rotatable mesh** — isosurface rendering of brain + tumor regions (WT/TC/ET) with orbit controls
- **Live inference** — run segmentation with any trained model and see Dice scores instantly
- **Scanner shift simulation** — toggle hospital-specific scanner distortions to see how models respond

## Documentation

Start at the [documentation index](docs/README.md). The set is split by concern so each doc stays focused:

| Doc | Scope |
|---|---|
| [methodology.md](docs/methodology.md) | Research design — question, hypotheses, methods, evaluation (the *why*) |
| [workflow.md](docs/workflow.md) | **Start here to run it** — the four runs, pipeline order, measured costs, decision gates |
| [data.md](docs/data.md) | Dataset spec, labels, and the reproducible data-prep pipeline |
| [architecture.md](docs/architecture.md) | System architecture, end-to-end flow, module layout, logging strategy |
| [data-pipeline.md](docs/data-pipeline.md) | Case → hospital partition, synthetic shift, preprocessing, caching, sampling |
| [federated-learning.md](docs/federated-learning.md) | The FL round loop; FedAvg / FedBN / local-only aggregation |
| [experiments.md](docs/experiments.md) | Experiment matrix, evaluation protocol, how H1/H2/H3 are measured |
| [specs.md](docs/specs.md) | Reference sheet — hyperparameters, model dims, hardware, seeds, artifact layout |
| [environments.md](docs/environments.md) | Windows / WSL2 / Colab — what runs where, portability contract, recipes |
| [progress-log.md](docs/progress-log.md) | Dated lab notebook of decisions and milestones |

## Repository layout

```
src/fedbrats/        The library: config, partition, shift, data, model, metrics, train, federated
scripts/             Entrypoints: build_partition.py, build_cache.py, run_experiment.py
docs/                Project documentation (see above)
unzip_data.py        Local one-off: decompress .nii.gz -> .nii onto the D: drive
colab_setup.ipynb    Colab: download from Kaggle -> stream-unzip in batches -> Google Drive
pyproject.toml       Python environment (managed with uv)
```

## Quickstart

```bash
uv sync                                   # Linux: add --extra flare for the (optional) FLARE port
uv run python scripts/build_partition.py  # deterministic split -> artifacts/splits/partition.json

# Smoke the whole pipeline in ~2 minutes, on any OS:
uv run python scripts/build_cache.py --max-cases 3 --workers 4
uv run python scripts/run_experiment.py --method fedbn --rounds 2 \
    --max-train-cases 3 --max-test-cases 2

# The real thing (Colab T4; cache dir auto-resolves to /content/cache):
uv run python scripts/build_cache.py --workers 8
for m in centralized local fedavg fedbn; do
    uv run python scripts/run_experiment.py --method $m --dim 2d
done
```

Results stream to `artifacts/runs/<method>_<dim>_<seed>/metrics.jsonl`.

## Compute

- **Local:** WSL2 **or** native Windows, RTX 3050 Laptop (4 GB VRAM) — data prep and quick checks.
  NVIDIA FLARE runs on WSL2 only ([why](docs/environments.md)).
- **Training:** Google Colab **T4 (16 GB VRAM)**, with the dataset staged in Google Drive.
- **Cache:** ~35 MB/case → ~44 GB for all 1251. Set `FEDBRATS_CACHE_DIR` to keep it off `C:`.

## Reproducing the data prep

See [`docs/data.md`](docs/data.md). In short: run `colab_setup.ipynb` in Colab to pull the
dataset from Kaggle and build the unzipped copy in `Drive/MyDrive/capstone/`.
