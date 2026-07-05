# Documentation

Full, detailed documentation for **Personalized Federated Learning for Brain-Tumor
Segmentation** — a capstone project training a segmentation U-Net across simulated
hospitals that never share raw data, and showing that **personalized FL (primarily
FedBN)** beats both a single global **FedAvg** model and **local-only** training,
especially for hospitals whose scanners/protocols differ most.

This `docs/` set is the **authoritative, current** reference. It supersedes and
consolidates the older top-level notes (`README.md`, `PLAN.md`, `DATA.md`,
`WSL.md`, `CLAUDE.md`), and additionally covers work those files predate — the
**synthetic scanner-shift** heterogeneity, the **performance model** (the pipeline
is CPU-bound on data loading), and the current run mechanics.

## Read in this order

| # | Document | What it covers |
|---|----------|----------------|
| 01 | [Overview](01-overview.md) | The idea, motivation, goal, research question, hypotheses **H1–H3**, contributions, and what "done" looks like. |
| 02 | [Background](02-background.md) | Federated learning primer, personalization (FedBN & friends), BraTS segmentation, and why medical FL heterogeneity is the real problem. |
| 03 | [Architecture](03-architecture.md) | Repo layout, every module explained, the 8-step pipeline, end-to-end data flow, and the **design invariants** you must not break. |
| 04 | [Data pipeline](04-data-pipeline.md) | BraTS format, lazy 2D slicing, transforms, case-level splits (no leakage), hospital partitioning, and the **synthetic scanner shift** in full. |
| 05 | [Methods](05-methods.md) | Every method (centralized, local, FedAvg, FedProx, **FedBN**, personal-head, fine-tune), the client-side personalization mechanism, and the FLARE simulator setup. |
| 06 | [Running](06-running.md) | Environment, `run_all.sh`, every env knob, the performance model, scaling, and 4 GB survival tips. |
| 07 | [Evaluation](07-evaluation.md) | Metrics (Dice WT/TC/ET), fairness, personalization gain, the result schema, `analyze.py` outputs, and how to read the **H1–H3** story. |
| 08 | [Design decisions](08-design-decisions.md) | A rationale log for every non-obvious choice, with the reasoning and the failure it prevents. |
| 09 | [Roadmap](09-roadmap.md) | Current status, next steps, the real-FeTS swap, and stretch goals. |

## One-paragraph summary

Hospitals can't pool patient MRIs (privacy), but a model trained at just one
hospital generalizes poorly, and a single federated **average** model quietly
fails on the hospitals whose scanners differ most. This project simulates that
setting on **BraTS 2021** MRI, federates a 2D MONAI U-Net with **NVIDIA FLARE**,
and demonstrates that **personalized FL** — keeping each hospital's BatchNorm
statistics local (**FedBN**) — recovers those outlier hospitals **without** hurting
the average. Because we don't yet have the real FeTS institutional split, we
manufacture the heterogeneity with a deterministic, physically-motivated
**per-hospital scanner shift**. See [Overview](01-overview.md) to start.
