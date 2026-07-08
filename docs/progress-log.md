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
  240×240×155, int16, 4 modalities + a `{0,1,2,4}` mask, ~0.6% tumor. (See [`data.md`](data.md).)
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
- Wrote [`../colab_setup.ipynb`](../colab_setup.ipynb): download the dataset **on Colab**
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
- **FL framework: custom sequential PyTorch loop** (not NVIDIA FLARE) — clients share one GPU, so peak
  VRAM = one model regardless of K; FedBN = skip BN keys in the average. FLARE's multi-process clients
  risked OOM on the 4 GB card.
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

### Next
- Preprocessing + data module: load case → scanner shift → crop → z-norm → sample (2D/3D) → cache.
- Then the 2D U-Net + Dice metric + a centralized sanity run on Colab.
