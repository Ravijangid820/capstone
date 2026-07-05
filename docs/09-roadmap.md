# 09 · Roadmap — status, next steps, stretch goals

## 9.1 Current status

| Area | State |
|------|-------|
| **Phase 1 — centralized pipeline** | ✅ Validated. Ceiling ~0.79–0.81 mean Dice (WT ~0.87). The data + model + metric path is proven end-to-end. |
| **Full FL pipeline** | ✅ Runs end-to-end in WSL — FedAvg / FedProx / FedBN + local + centralized + fine-tune + analyze all execute. |
| **Synthetic heterogeneity** | ✅ Implemented + verified — the per-hospital scanner shift (`--synthetic-shift`), threaded through every method. Nonlinear so it survives z-normalization. |
| **Performance** | ✅ Characterized — CPU-bound; `WORKERS` is the speed lever; `MAX_CASES` scopes a run to hours. |
| **Real FeTS split** | ⏳ Not obtained (Synapse/FeTS registration). Code path (`--fets-csv`) is ready and waiting. |
| **Headline result** | ⏳ Pending a completed scaled run with the shift on (the per-hospital H1–H3 table + figure). |

## 9.2 Immediate next steps (priority order)

1. **Complete a scoped validation run and confirm H1–H3.**
   `MAX_CASES=200 ROUNDS=20 EPOCHS=20 WORKERS=12 bash run_all.sh` → check
   `results/per_hospital.csv` for FedBN ≥ FedAvg on the outlier sites (site-4/5/6) and
   the fairness summary in `summary.csv`. This is the go/no-go for the approach.
2. **Scale to the headline run.** Raise `MAX_CASES` (toward all cases), `ROUNDS` (→50),
   `EPOCHS` (→40). Keep each run inside one sitting (no mid-run resume yet). Produce the
   final `per_hospital.csv` + `personalization_gain.png`.
3. **Swap in real FeTS heterogeneity when the CSV lands.** Drop
   `data/fets_partitioning.csv` in place and run with
   `SPLIT="--fets-csv data/fets_partitioning.csv"` (the synthetic shift auto-disables).
   Compare synthetic-shift vs real-institution results — the real split is the
   publishable version.
4. **Write up the result.** Per-hospital table + figure with outliers highlighted, the
   fairness deltas, and the H1–H3 narrative.

## 9.3 Possible robustness / ablation work

- **Shift-strength ablation.** Vary the scanner-profile magnitudes and show the FedBN
  gain scales with heterogeneity — turns the "site gradient" into a clean curve.
- **Personalization comparison.** FedBN vs personal-head vs fine-tune across the same
  hospitals — which recovers outliers best per unit of kept-local parameters.
- **Rounds/epochs sensitivity.** How quickly FedBN's advantage emerges vs FedAvg.
- **More hospitals / finer FeTS split** (`partitioning_2.csv`) — does the story hold as
  the federation gets more fragmented?

## 9.4 Stretch goals

- **Differential privacy** (Phase-4): add DP-SGD (opacus is an optional dep) and study
  the privacy/utility/personalization trade-off.
- **Communication efficiency:** measure/reduce bytes-on-the-wire (FedBN already sends
  fewer params implicitly? — quantify).
- **3D or larger backbone** on a bigger GPU (Colab/Azure) — the code is
  device-agnostic; only VRAM changes.

## 9.5 Known limitations (be honest in the writeup)

- **Synthetic ≠ real heterogeneity.** The scanner shift is physically motivated and
  survives normalization, but it is *engineered*; the real FeTS institutional split is
  the credible version and remains the priority.
- **2D, not 3D.** Sufficient for the FL question, but absolute Dice is below 3D SOTA.
- **Simulated federation.** FLARE Simulator on one machine — correct for the
  *algorithmic* question, but not a real multi-site deployment (no real network,
  stragglers, or system heterogeneity).
- **No mid-run resume.** A long run must finish in one sitting; only between-step
  restart is available.

## 9.6 Environment / operational notes to carry forward

- Runs on **WSL2** with **`uv`**; keep repo + data on native **ext4**.
- The pipeline is **CPU-bound** — set `WORKERS` near your core count; leave `BATCH=8`.
- Scope with `MAX_CASES` before any multi-hour commitment.
- Data (~15 GB) and `*.pt` checkpoints are **gitignored**; result JSON/CSV/PNG are
  tracked (they're the deliverable).
- Git: commits are authored solely by the project owner — **no Claude/Anthropic
  attribution** (see `CLAUDE.md`).
