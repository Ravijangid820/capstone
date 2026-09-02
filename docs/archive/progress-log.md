# Progress log

A dated lab notebook: what was done, what was decided, and *why*. Newest entries at the bottom.

---

### 2026-07-05 — Fresh start
- Reset the workspace to a clean slate, keeping only the raw data, the Python environment, and
  git history. The earlier build had accumulated too many one-off scripts and confusing runs.
- **Goal unchanged:** show personalized FL (FedBN) beats FedAvg and local-only for outlier
  hospitals. Only the *approach* is being rebuilt, step by step.
- Decision: proceed one verified step at a time; no long speculative runs.

### 2026-07-05/06 — Confirmed the dataset is 3D
- Inspected the actual files rather than assuming. BraTS 2021 is **3D volumetric**: 1251 cases,
  240×240×155, int16, 4 modalities + a `{0,1,2,4}` mask, ~0.6% tumor. (See [`data.md`](../data.md).)
- This is the fact that makes the 2D-vs-3D model choice a real decision.

### 2026-07-06 — Local unzip to the D: drive
- Decompressed all 1251 cases (`.nii.gz` → `.nii`) with `unzip_data.py`.
- **Constraint found:** Windows C: had ~41 GB free; the unzipped set is ~114 GB. Output was
  therefore pointed at the **D:** drive (267 GB free).
- Verified: 6255 `.nii` written, 0 partial files, all volumes load as valid NIfTI with the
  expected shape/dtype/labels.

### 2026-07-06 — Hardware assessment → pivot to Colab
- Local GPU is an **RTX 3050 Laptop, 4 GB VRAM** — too small for a multi-model FL study
  (3D barely fits; even 2D leaves little room, and FL means training many models many times).
- **Decision:** run training on **Google Colab (T4, 16 GB VRAM)**. Local machine stays for
  data prep and quick sanity checks.

### 2026-07-07 — Colab data pipeline built and run
- Wrote [`../colab_setup.ipynb`](../../colab_setup.ipynb): download the dataset **on Colab**
  (fast datacenter link, no slow home upload) → extract → **stream-unzip in 100-case batches**
  to Google Drive, deleting each local batch so Colab's ~100 GB disk never fills.
- Made every stage skip-safe: download reused from Colab/Drive if present; extraction skipped if
  already done; unzip skips cases already in Drive (resumable across disconnects).
- **Milestone:** data prep complete — compressed archive **and** the full ~114 GB unzipped set
  now live in `Drive/MyDrive/capstone/`. Ready to build the training pipeline.
- Started the project documentation set (README + methodology + data + this log).

### 2026-07-07 — Decision: build both 2D and 3D
- Instead of choosing one, we make the pipeline **dimension-parametric** and evaluate **both**
  backbones. Rationale: (a) de-risks 3D — 2D is the guaranteed fallback; (b) stronger claim —
  "do H1/H2/H3 hold under both 2D and 3D?"; (c) the FL machinery is shared, so the extra cost is
  mostly one data-sampler + one model.
- **Sequencing:** 2D end-to-end first (guaranteed deliverable) → 3D single-model **feasibility
  spike** on the T4 → run the 3D FL study only if the spike passes; otherwise report the spike and
  keep 2D as the deliverable.

### 2026-07-08 — Split decided + full doc set + hardware measured
- **Hospitals: K = 4** (3 typical + 1 outlier). Split = partition-then-split, ~1000 train / ~251 test,
  `train_per_hospital` as a knob (~120–150) so H1 stays visible.
- **FL framework: custom loop now, FLARE later** — build a custom sequential PyTorch round-loop first
  (clients share one GPU, so peak VRAM = one model regardless of K; FedBN = skip BN keys in the average)
  to get transparent H1/H2/H3 results fast, then port to NVIDIA FLARE as a framework demonstration.
  (Correction: FLARE *can* be configured to share one GPU — it's the extra setup/opacity we defer, not a hard OOM wall.)
- **Hardware probes (RTX 3050):** 3D U-Net *fits in memory* (96³ = 0.88–1.78 GB, 128³/base16 = 2.06 GB);
  per-step 0.2–0.5 s. So local 3D is viable for testing; speed (not memory) is the limiter for full sweeps.
- **Docs:** added a structured set with Mermaid diagrams — `architecture`, `data-pipeline`,
  `federated-learning`, `experiments`, `specs`, plus a docs index. Logging strategy defined (run.log +
  metrics.jsonl + committed split manifest).

### 2026-07-08 — Partition built + verified
- First code: `src/fedbrats/` scaffolding (`config`, `logging_utils`, `partition`) + `scripts/build_partition.py`.
- Ran it: 1251 cases → **H1–H3 = 251 train / 62 test, H4 (outlier) = 250 train / 62 test** → **1003 train / 248 test**.
- Verified **deterministic** (identical md5 on re-run) and consistent (all 1251 assigned once, splits valid).
- Manifest committed at `artifacts/splits/partition.json` — the source-of-truth split for every run.

### 2026-07-08 — Found (and fixed) the blur-halo bug
- Trying to reproduce the shift verification through `preprocess()`, hospitals produced **different
  array shapes**. Cause: `apply_shift`'s Gaussian blur smears brain intensity into BraTS's exactly-zero
  background, and the brain mask was taken *after* the shift — so it grew with blur σ
  (H1 +112k voxels … **H4 +833k, a 57% inflated "brain"**).
- Why it mattered: hospital-dependent crop shapes, z-norm computed over the halo, and **hospital
  identity leaking as geometry** — a FedBN "recovery" of H4 could have been an artifact.
- **Fix:** derive mask + bbox from the *unshifted* volume, then `apply_shift(...) * mask`. All four
  hospitals now share one shape; pairwise post-z-norm diffs 0.095–0.388 σ, H4 margin **+0.149 σ**,
  linear shift still washes out to **0.0000**.
- Lesson: the docs recorded the shift as "verified", but the check had been run on a path that skipped
  the crop. *Verify through the real code path.*

### 2026-07-08 — Approach settled, custom FL loop built
- **Environments: WSL2 *and* native Windows** both supported for the custom loop. FLARE stays
  Linux-only (it imports the POSIX `resource` module) → moved to an optional extra with a platform
  marker so `uv sync` on Windows doesn't install a package that cannot import. New
  [`environments.md`](environments.md) captures the portability contract (`__main__` guards, picklable
  workers, no hardcoded paths, memmap handles, `num_workers=0` on Windows).
- **Matched compute (new decision):** local-only and centralized train `R × E` epochs — the same total
  local epochs a hospital spends across a federated run. Otherwise H1 just measures FedAvg's ~R× larger
  training budget.
- **Eval scope (new decision):** diagonal headline + a 4×4 cross-hospital matrix for local-only at the
  final round. Forced a `metrics.jsonl` schema change: `hospital` → `model_hospital` + `test_hospital`.
- **Per-volume Dice (new decision):** predict all slices → stack → score the volume. Mean-of-per-slice
  Dice inflates scores. Plus the BraTS empty-GT convention (empty pred on empty GT = 1.0), without
  which the ET column is meaningless.
- **Code:** `data` (cache + samplers), `model` (MONAI U-Net + `bn_keys`), `metrics`, `train`,
  `federated` (one round loop, four methods), `scripts/build_cache.py`, `scripts/run_experiment.py`.
- **Three traps handled:** FedBN state lives in `state_dict()` (buffers!), `num_batches_tracked` is
  int64 and is *copied* not averaged, and BN keys are found by **module type** — MONAI names them
  `...adn.N.bias`, so substring matching would have silently collapsed FedBN into FedAvg.
- **Verified:** 18 invariants pass (halo fix, `bn_keys` coverage, Dice conventions, dtype preservation,
  *FedBN with K=1 ≡ local-only*, *averaging identical states ≡ that state*). All four methods smoke-run
  end to end in 2D **and** 3D on the RTX 3050. Cache: 24 cases in 56 s (4 workers), 837 MB → ~35 MB/case.
- **Also fixed:** `pyproject.toml` still pointed hatch at `src/braintumor_fl`, which no longer exists —
  a fresh clone + `uv sync` would have failed. (It only worked here via a stale editable-install `.pth`.)
- **Caught by the smoke run:** `pad_to_multiple` was padding `(W, Z)` instead of `(H, W)` in 2D
  evaluation, so the U-Net's skip connections mismatched on odd in-plane dims.

### 2026-07-08 — Costed the full run: it is eval-bound, not train-bound
- Benchmarked the real code paths on the 3050 before committing a Colab session: **48 ms/training step**,
  **0.41 s per full-volume evaluation**, 2.3 s/case to cache (4 workers), 35 MB/case.
- **Surprise:** scoring 248 test volumes each round costs **1.7 min vs 0.5 min of training — 3.5×**.
  The run is dominated by evaluation, which the design docs gave no hint of. Full 2D matrix
  (4 methods × 25 rounds) ≈ 3.7 h on the 3050, ≈ 2.5 h on a T4 — one session, so no change needed.
  Lever if ever required: subsample per-round eval, full test set at the final round only.
- Added [`workflow.md`](workflow.md) — the pipeline in order, measured costs, and the decision gates —
  as the doc to read before running. Also published as a shareable page for supervisor review.
- **Sequencing decided:** run **E0 (centralized) alone first** (~35 min). It exercises the whole
  pipeline; if WT Dice does not climb past ~0.7 the fault is data/loss, debuggable with one model rather
  than four hospitals and an aggregator.
- Two knobs flagged as *guesses, not evidence*: `R = 25` (chosen before seeing any learning curve) and
  `train_per_hospital = 150` (justified by reasoning about H1's visibility). If H1 comes out weak,
  raising the cap weakens it further; lowering it toward ~80 sharpens it.

### 2026-07-09 — Full 2D matrix run locally + results (H2 & H3 supported, H1 not)
- **Ran the whole 2D matrix on the RTX 3050** (not Colab): built the 848-case cache (29 GB on `D:`,
  ~14 min, 6 workers), then E0→E1→E2→E3 at `R=25, E=1`. Total ~6 h wall-clock.
- **Data-root fix:** the committed `config.py` pointed at `D:/data/unzipped`, which no longer existed
  (the unzipped 114 GB set lives at `data/unzipped/`). Made `_default_data_root` *probe* candidate
  locations instead of hardcoding one, so a fresh run finds the data on any of the machines we use.
- **Sanity gate passed decisively:** E0 centralized hit WT 0.81 at round 1, ceiling ~0.852.
- **Results (final-round diagonal WT):** centralized 0.852, local 0.853, FedAvg 0.835, FedBN 0.852.
  Per-hospital H4: local 0.857, **FedAvg 0.737 (collapse)**, **FedBN 0.829 (recovered)**.
  - **H2 supported** — FedAvg fails the outlier (0.737 < 0.857). The synthetic shift is well
    calibrated; `shift.py` needs no change.
  - **H3 supported** — FedBN recovers H4 (0.737→0.829) *and* beats FedAvg on the mean (0.852 ≥ 0.835),
    tying local-only and the centralized ceiling **without pooling data**.
  - **H1 not supported** — mean(FedAvg) 0.835 < mean(local) 0.853. But FedAvg *improves* the three
    typical hospitals (H1 0.848→0.883, H2 0.863→0.884); the outlier's 0.12 collapse alone drags the
    mean below local. Not "federation is useless" — "one global model can't serve cluster + outlier."
    This is the strongest possible motivation for FedBN, and FedBN delivers.
- **Tooling added:** `scripts/analyze.py` (reads `metrics.jsonl` → H1/H2/H3 verdicts + cross-matrix),
  `scripts/plot_results.py` (3 figures under `artifacts/figures/`), and `colab/` notebooks that mirror
  the run on a T4.
- **Perf side-quest (measured, not guessed):** the run is **I/O-bound** on this box — the per-round
  4-client working set (~22 GB) exceeds page cache (~14 GB), so cases re-read cold from `D:` each round
  (E0: 273 ms/step real vs 40 ms warm). Confirmed the docs' call that `num_workers=0` is fastest on
  Windows (spawn IPC pickling each 9 MB batch costs more than it saves: nw=0 143 ms/step vs nw=4 155);
  larger batches help only warm (~15%, VRAM fine) and would break matched-compute + force a full
  restart, so **kept batch 8 / nw=0**. The real speedup lives on Colab (local SSD + more RAM), which
  the notebooks already use. *(Lesson relearned: a stray orchestrator subshell survived a TaskStop and
  launched E1 early, contaminating the first worker benchmark — always confirm the process is dead.)*

### Next
- **Analysis is done for 2D** — tables in [experiments.md](experiments.md#4-results--2d-backbone-v1-baseline-only-r25-e1-seed-42-150-trainhospital), figures in `artifacts/figures/`.
- **3D feasibility spike** on the T4 (memory is fine; speed is the gate). If it passes, repeat the
  matrix in 3D and add the "does the story hold in 3D?" comparison.
- *(optional)* NVIDIA FLARE port as a framework demonstration — the science is now settled on the
  custom loop.

### 2026-07-21 — 3D training and Web demo
- **3D training completed** — full 3D matrix (centralized, local, fedavg, fedbn), seed 42, R=25. 3D reversal observed: FedAvg robust on outlier, FedBN underperforms.
- **Web demo server built** — interactive dashboard with 2D slice viewer, 3D mesh viewer, live inference, scanner shift simulation. Python ThreadingHTTPServer with model caching.
- **WASM marching cubes replaced with pure JS** — fixed 5 critical bugs (axis indexing, BigInt, buffer overflow, API misuse, flat array handling). Pure JS implementation is cross-browser compatible and has no Rust toolchain dependency.
- **Model caching added** — `_get_model()` cache keyed by (dim, method, hospital). 10-20× speedup on repeated requests.
- **Smoke test suite created** — 17 pytest tests covering dice_binary, labels_to_regions, build_partition, build_model, bn_keys.
- **Three.js bundled locally** — vendor/three.min.js + vendor/OrbitControls.js for offline demos.
- **README updated** — added 3D results table, web demo section, updated phase 9 status.

### 2026-08-10 — Accuracy round 2: evaluation noise, then the levers

Prompted by "can we increase the accuracy?", the first move was to re-read the existing logs
rather than start training. That turned out to matter more than any hyperparameter.

- **Evaluation noise exceeds the effects under test.** The curves plateau by ~round 15 and then
  oscillate; on 2D seed 42, FedAvg's mean WT is 0.8611 at r24 and 0.8354 at r25 while local-only
  sits at 0.8491/0.8526 — H1 is decided in opposite directions by adjacent rounds. `analyze.py`
  scored `max(round)`, so **H1's "NOT SUPPORTED" was a coin flip, not a result.** Re-scoring the
  same frozen logs over the last five rounds moves H1 from 1/3 seeds to **3/3**. Nothing retrained.
- **H3 is two claims wearing one verdict.** "FedBN recovers H4" is solid (≈0.82 vs FedAvg's ≈0.76,
  all seeds, either estimator). "FedBN ≥ FedAvg on the mean" is inside the noise. Bundled, a coin
  flip decides the fate of the real finding — now reported split.
- **3D checked the same way: all three verdicts are estimator-independent.** The 3D reversal does
  not rest on this choice; the 2D H1 verdict did. Worth stating, since it is the difference
  between a robust finding and a fragile one.
- **Found before it bit: a rerun would have silently corrupted the baseline.** `MetricsWriter`
  appends and `run_id` has no run counter, so re-running a method+seed writes fresh rounds
  *underneath* the old ones in one file — and `analyze.py`, scoring `max(round)`, would average two
  experiments into one number with nothing in the output to show for it. Now guarded
  (`guard_run_dir`) and namespaced (`--tag`).
- **Baseline frozen and hash-verified** — `artifacts/snapshots/v1/` + `MANIFEST.json` (SHA-256/file,
  git commit, headline numbers). 42 files across 14 runs, `--verify` re-checks them.
- **Per-case Dice backfilled from checkpoints, no retraining** (`scripts/rescore.py`). It was
  being computed and discarded, which left no basis for a CI or a paired test. Doubles as a
  regression test: with the new flags off, every rescored mean reproduces its logged round-25
  value to 4 dp (fedavg H4 0.7373, fedbn H4 0.8290, local H4 0.8569) — confirming the refactored
  eval path is bit-identical to the baseline's.
- **`--preset v2`** — cosine LR across rounds (the only place a schedule can live when Adam is
  rebuilt each round), augmentation (there was none at all), flip-TTA, small-component filtering
  with WT ⊇ TC ⊇ ET re-nesting, a 20-case/hospital validation split taken *after* the train cap so
  training data is unchanged, and best-val model selection. Defaults untouched, so the baseline
  stays bit-reproducible.
- **`scripts/compare_runs.py`** — config diff, Δ Dice, and paired per-case stats (bootstrap 95% CI
  + Wilcoxon). Applies one estimator to both sides and warns loudly when it cannot.
- **Tests: 17 → 44.** The new ones target failures that stay silent — augmentation desynchronizing
  image from mask, TTA forgetting to un-flip (probed with a flip-equivariant model), the run guard
  not firing, a preset naming a field that does not exist.

### 2026-08-22 — v3, v4 and v5; v5 adopted as the best recipe

- **v3** — corrected v2's regression. Decomposing where v2 lost showed its intensity jitter and
  Gaussian noise perturb gamma, bias and blur, the same channels the synthetic scanner shift uses,
  so it was augmenting along the experimental variable. Dropped those two terms, kept geometric
  augmentation, and raised training data 150 → 230 cases/site (40% of the pool had never been
  used). Every method improved in both backbones.
- **v4 — failed, stopped after one run.** All v3 runs were still improving at round 25, so v4 tried
  40 rounds. 2D centralized came back *worse* (0.8832 test / 0.8748 val vs v3's 0.8931 / 0.8830).
  The inference was wrong: v3's late-curve rise was the cosine anneal consolidating the model, not
  headroom from more steps. Stretching the cosine across 40 rounds kept LR high far longer.
- **Two probes instead of two matrices** (~7 h rather than ~74 h), on 2D FedBN, judged on
  validation as well as test. *40 rounds annealing on v3's 25-round timetable* won (+0.0103 test,
  +0.0039 val). *`base_channels` 48* was rejected — it gained on test but **lost** on validation,
  the signature of test-set noise. These models were round-limited, not size-limited.
- **v5 = v3 + 40 rounds with `lr_anneal_rounds=25`.** All eight runs beat v1 paired per-case
  (2D +0.0245…+0.0510, 3D +0.0054…+0.0343), every one significant. v5's FedBN 2D reproduced the
  probe's 0.8756 to four decimals across two separate invocations days apart.
- **The 3D reversal is finished.** H2 and H3's inequalities now hold in 3D. **But do not write
  "FedBN wins in 3D"** — that margin is not significant (uncorrected p = 0.066, Holm-corrected
  p = 0.53, sign split 39/23). The defensible phrasing is "no significant difference in 3D".
- **Outlier mechanism corrected.** Earlier I blamed FedAvg's weight averaging; centralized, which
  has no aggregation step, shows the same effect just as strongly. The cause is optimizing over a
  majority-dominated distribution, which makes H2 broader than first stated and puts BatchNorm
  locality in exactly the explanatory role FedBN's thesis claims.
- **Documentation consolidated.** `conventions.md` fixes the hospital/hypothesis name collision
  (Sites A–D vs H1–H3) and lists every retracted claim; `iteration-report.md` is the single v1→v5
  record; three older reports carry superseded banners; `check_docs.py` re-derives 16 headline
  figures from the frozen logs and fails if a document drifts.
