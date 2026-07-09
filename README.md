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
| 9. 3D feasibility spike → 3D matrix | ⬜ next — memory fits; speed is the gate |
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

Details and figures: [experiments.md](docs/experiments.md#4-results--2d-backbone-r25-e1-seed-42-150-trainhospital) · `artifacts/figures/`. Regenerate: `python scripts/analyze.py --dim 2d`.

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
