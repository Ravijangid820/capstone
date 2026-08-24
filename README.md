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

> **Naming — read before writing anything.** `H1`–`H4` are **hospitals** in the code and logs;
> `H1`–`H3` are **hypotheses**. In all prose the hospitals are **Site A–D** (Site D = `H4` = the
> outlier). Full rules, retracted claims and reporting conventions:
> **[docs/conventions.md](docs/conventions.md)**.

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
| 8. Analysis (H1/H2/H3) → report | ✅ done — see **Results** below (verdicts changed after v2–v5) |
| 9. 3D feasibility spike → 3D matrix | ✅ done — full 3D matrix; the early "reversal" reading was later refuted |
| 10. Accuracy iterations v2–v5 | ✅ done — see [iteration-report.md](docs/iteration-report.md) |
| 11. *(optional)* NVIDIA FLARE port | ⬜ Linux/Colab only |

## Results (best recipe, `--preset v5`) — supersedes everything below

Every method improves in **both** backbones, all significant. Paired per-case Δ WT Dice against
the frozen baseline, 248 test volumes per run, seed 42:

| Method | 2D Δ | 2D p | 3D Δ | 3D p |
|---|---|---|---|---|
| Centralized (ceiling) | **+0.0510** | 3.1e-35 | **+0.0115** | 4.3e-07 |
| Local-only (floor) | **+0.0245** | 3.4e-20 | **+0.0054** | 2.2e-04 |
| FedAvg | **+0.0309** | 5.8e-34 | **+0.0073** | 2.0e-04 |
| **FedBN** | **+0.0272** | 6.4e-28 | **+0.0343** | 8.4e-16 |

Mean WT Dice across hospitals (training recipe only, no TTA):

| Method | 2D baseline → v5 | 3D baseline → v5 |
|---|---|---|
| Centralized | 0.8582 → **0.8954** | 0.8768 → **0.8879** |
| Local-only | 0.8436 → **0.8699** | 0.8490 → **0.8527** |
| FedAvg | 0.8502 → **0.8629** | 0.8525 → **0.8618** |
| FedBN | 0.8514 → **0.8756** | 0.8422 → **0.8642** |

### H3 now holds in both backbones

| | H1 | H2 | H3 |
|---|---|---|---|
| 2D baseline | ✅ | ✅ | ✅ |
| **2D v5** | ❌ | ✅ | ✅ |
| 3D baseline | ✅ | ❌ | ❌ |
| **3D v5** | ✅ | ✅ | **holds, not significant** ⚠ |

**The 3D reversal does not survive a properly tuned recipe.** Under v1, FedAvg beat FedBN in 3D
*significantly*; under v5 that is gone. H2 now holds in 3D, and H3's inequality holds too
(2D +0.0128 mean / +0.0604 outlier; 3D +0.0024 / +0.0087).

> **Do not write "FedBN wins in 3D".** The 3D margin on the outlier is **not statistically
> significant** — uncorrected p = 0.066, Holm-corrected p = 0.53, sign split 39/23. The bootstrap CI
> excludes zero but the Wilcoxon test does not, meaning the shift comes from magnitude on a minority
> of volumes. The defensible sentence is **"no significant difference in 3D"**, which is still a
> change from v1. In 2D FedBN wins decisively (59 of 62 volumes). See
> [results-v5-summary.md](docs/results-v5-summary.md).

### Why the outlier moves the way it does

Grouping the 3D methods by whether training mixes sites explains the outlier column:

| | typical sites H1–H3 | outlier H4 |
|---|---|---|
| Centralized (pools all sites) | +0.0176 | **−0.0087** |
| FedAvg (averages all weights) | +0.0156 | **−0.0092** |
| FedBN (keeps BatchNorm local) | +0.0278 | **+0.0046** |
| Local-only (never mixes) | +0.0025 | **+0.0072** |

Better optimization over a majority-dominated distribution fits the majority harder and the
minority worse — and it happens to centralized, which has no aggregation step at all, just as
strongly as to FedAvg. So H2 is not an artefact of weight averaging; it is what happens whenever
one model is optimized across heterogeneous sites. Keeping BatchNorm local is what protects the
outlier, which is exactly FedBN's claim.

**What v5 is:** cosine LR annealing to its floor by round 25 but training for **40 rounds**,
geometric augmentation only, flip-TTA and small-component filtering at inference,
validation-based checkpoint selection, and 230 training cases/hospital.

Full analysis: **[improvements.md](docs/improvements.md)**.

Reproduce: `python scripts/run_matrix.py --dim 2d 3d --seed 42 --preset v5 --tag v5`

## Earlier results (superseded — kept for the record)

Four earlier result sets used to live here: the original v1 2D and 3D tables, and the v2 and v3
reruns. They contained hospital columns labelled `H1`–`H4`, which collide with the hypothesis
names, and two claims that later runs refuted. They have been consolidated into one place rather
than left to contradict each other:

**→ [docs/iteration-report.md](docs/iteration-report.md)** — every iteration v1→v5, every method,
both backbones, all three regions, with the statistics and the limitations.

Two claims from those tables are **retracted** and must not be reused:

| Retracted | Why |
|---|---|
| *"In 3D all three hypothesis verdicts reverse"* — presented as a novel contribution | Refuted. Under v5 both H2 and H3's inequalities hold in 3D. The reversal was an artefact of the v1 training recipe, not a property of the backbone. |
| *"FedBN suffers in 3D because 150 cases cannot estimate stable 3D BatchNorm statistics"* | Not supported. The cause was the round budget: a capacity probe (`base_channels` 48) was rejected, and 3D local-only — flat under v2 and v3 — improved once trained for 40 rounds. |

Full list of retracted and restricted claims: **[docs/conventions.md](docs/conventions.md) §6**.


## Project showcase

A 27-slide walkthrough of the whole study — an animated four-part federated-learning tutorial,
then the dataset, scanner shift, preprocessing chain, the four methods, all five iterations and the
final results — built as one self-contained HTML file for presenting to a reviewer or examiner.

The tutorial runs on a canvas state machine over the phases of a communication round, so the same
engine shows pooled training, FedAvg, FedBN, and a replay of the actual logged run just by changing
what travels on the wires. The replay's round counter and per-site Dice are read from
`artifacts/snapshots/v5/`, not invented.

```bash
uv run python scripts/showcase_assets.py --data-root data/BraTS2021_Training_Data   # images, once
uv run python scripts/build_showcase.py                                             # the deck
# open artifacts/showcase/showcase.html
```

Every figure is re-derived from `artifacts/snapshots/` at build time and every picture is rendered
by the pipeline's own functions, for the same reason [`scripts/check_docs.py`](scripts/check_docs.py)
exists: a slide with a hand-typed Dice number goes stale silently, and nobody re-derives a figure
on a slide before presenting it.

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
