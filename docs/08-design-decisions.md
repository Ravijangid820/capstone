# 08 · Design decisions — the rationale log

Every non-obvious choice, with the reasoning and the concrete failure it prevents.
This is the "why is it like this?" reference — read it before changing anything that
looks arbitrary, because most of it isn't.

## 8.1 Modeling & task

**2D slices, not 3D volumes.**
*Why:* 3D BraTS models need far more than 4 GB VRAM. 2D axial slices fit comfortably,
keep I/O manageable (no pre-slicing 15 GB to disk), and are fully sufficient to
demonstrate the *federated* question — the FL/personalization story is orthogonal to
2D-vs-3D. *Prevents:* CUDA OOM on the 4 GB laptop GPU; a multi-day data-prep step.

**BatchNorm (not InstanceNorm/GroupNorm).**
*Why:* FedBN is the primary method, and it works precisely because a scanner shift
lives in BatchNorm's running statistics. BatchNorm makes the personalization
mechanism meaningful; other norms would blunt it. All methods use the same norm for a
fair comparison. *Prevents:* a FedBN result that's inexplicably weak because the norm
layer doesn't actually capture the scanner difference.

**Sigmoid + per-channel Dice (multi-label), not softmax.**
*Why:* the BraTS regions WT/TC/ET **overlap** (a voxel can be in all three), so it's
multi-label, not mutually exclusive classes. *Prevents:* systematically wrong labels
(softmax would force the regions to compete).

**`BratsUNet` wrapper with all-default args (invariant #1).**
*Why:* FLARE reconstructs the server's initial model from a JSON config by
introspecting constructor args; MONAI's `UNet` has a **required** `spatial_dims` that
FLARE drops, so it can't rebuild the raw net → server crash. A wrapper whose
constructor takes only optional args always reconstructs. *Prevents:* a hard FLARE
server crash at job start. *Corollary:* state-dict keys are all `net.*` — keep them
consistent so checkpoints cross-load between centralized/local/federated.

## 8.2 Data & evaluation integrity

**Case-level splitting is the single source of truth (invariant #2).**
*Why:* splitting by *slice* would put slices from the same patient in both train and
val — classic medical-imaging leakage that inflates Dice. `case_split` (seed 42) splits
by patient, and every method routes through it, so all methods share the identical
per-hospital val set. *Prevents:* leakage (fake-high scores) and unfair
method-to-method comparison.

**Centralized trains on the union of hospital *train* cases only.**
*Why:* if the ceiling trained on data that other methods validate on, its per-hospital
scores would be inflated and the "ceiling" would be a mirage. `partitioned_splits`
enforces train/val separation across the whole comparison. *Prevents:* a dishonest
ceiling.

**Index only tumor-bearing slices (`min_tumor_pixels=100`).**
*Why:* most axial slices contain no tumor; training/eval on them wastes compute and
distorts Dice (trivially "correct" empty predictions). *Prevents:* misleadingly high
Dice and slow epochs. *Caveat:* the slice-index cache assumes this threshold — change
it and you must delete the cache.

**Lazy per-slice `nibabel` reads, no pre-slicing.**
*Why:* pre-slicing 1251×~60 slices to disk is tens of GB and slow on WSL's `/mnt/c`.
Memory-mapped `dataobj[..., z]` reads exactly the needed slice. *Prevents:* disk
blow-up and RAM pressure. *Consequence:* the workload is CPU/IO-bound (see §8.5).

## 8.3 Federation & personalization

**Server always runs plain FedAvg; personalization is client-side (invariant #4).**
*Why:* one clean, uniform mechanism for every strategy — "which keys does the client
keep local when the global model arrives." No custom server aggregators to maintain,
and strategies compose (personal-head = norm ∪ head). *Prevents:* a tangle of
per-method server code and subtle aggregation bugs.

**Eval-before-train each round (invariant #3).**
*Why:* the number we report must be the *received* model's quality — the true global
model for FedAvg, the genuine personalized model for FedBN — not a locally-overfit
model. Evaluating *after* local training would report a one-hospital-adapted model for
*every* method, erasing the distinction. *Prevents:* every method looking like
"fine-tune," washing out the FedAvg-vs-FedBN contrast.

**FedBN local params persist across rounds via `{site}_local.pt` (invariant #5).**
*Why:* FLARE may re-run the client *script* each round, which would reset in-memory
BatchNorm to the global values → FedBN would never accumulate specialization and would
silently equal FedAvg. Saving/restoring the kept-local params fixes this. *Prevents:*
FedBN degenerating into FedAvg with no error — the most dangerous kind of bug (wrong
result, no crash).

**Sequential FL clients (`--threads 1`).**
*Why:* one model on the 4 GB GPU at a time. *Prevents:* OOM from concurrent clients.
*Cost:* wall-clock scales with client count × rounds — which is why FL dominates
runtime.

## 8.4 Training stability (invariant #6)

**LR 5e-4 + cosine decay + grad-clip 1.0 + NaN-skip.**
*Why:* earlier training at `lr=1e-3` produced NaN blow-ups. The combination — a
conservative LR, cosine annealing to near-zero late, gradient clipping, and skipping
any non-finite-loss batch — makes training reliably stable on this data. *Prevents:*
runs that silently produce NaN weights (and a near-zero Dice that looks like a broken
pipeline).

## 8.5 The synthetic scanner shift (invariant #7)

**Manufacture heterogeneity at all.**
*Why:* the contribution (H2/H3) *requires* non-IID hospitals; on an even/IID split
FedBN ≈ FedAvg by construction. The real FeTS institutional split isn't obtained yet,
so a synthetic per-hospital scanner shift unblocks the whole result today and swaps
out cleanly for FeTS later. *Prevents:* being blocked indefinitely on data access, or
worse, reporting a null FedBN result and mistaking it for "FedBN doesn't work."

**The shift must be nonlinear/spatial, never affine.**
*Why:* the pipeline z-normalizes each channel over nonzero voxels, which **erases** any
affine `a·x+b` shift (verified: affine control → `mean|Δ|=0.0000` after norm). Only
gamma (nonlinear histogram reshaping), a spatial bias field, and blur *survive*
normalization and therefore actually move the BatchNorm statistics FedBN specializes
on. *Prevents:* a "heterogeneity" that the normalization silently removes, leaving
FedBN with nothing to do — a subtle null result that looks like the method failing.

**Site 0 = identity (reference scanner); later sites drift progressively.**
*Why:* guarantees one canonical hospital that matches the pooled distribution, and a
monotone-ish gradient of outlier-ness across sites — so H2/H3 show up as a *trend*
(bigger shift → bigger FedBN gain), which is far more convincing than a single data
point. *Prevents:* an all-outliers or all-matched configuration where the story is
ambiguous.

**Apply the identical shift to every method via a deterministic `case→site` map.**
*Why:* if centralized/local/FL/fine-tune saw different (or no) shifts, they'd be
comparing different data and every number would be meaningless. A pure
`site → params` function plus a deterministic `case → site` map (same partitions
everywhere) guarantees a given case gets the same shift in all steps. *Prevents:* an
apples-to-oranges comparison — the easiest way to get a confidently wrong result.

**Off for real FeTS.**
*Why:* FeTS data is *already* non-IID; layering a synthetic shift on top would confound
the real institutional differences. `run_all.sh` auto-disables `SHIFT` when `SPLIT`
uses `--fets-csv`. *Prevents:* double-counting heterogeneity.

## 8.6 Performance (learned empirically this project)

**Workers, not batch size, are the speed lever; the pipeline is CPU-bound.**
*Why:* the small 2D U-Net finishes a batch in milliseconds and then waits on the CPU
(nibabel reads + the per-slice transforms, including the shift's `gaussian_filter` +
percentile). Measured: GPU sat at ~11–33 % util while 8–12 CPU cores were pegged;
batch 32 was **no faster** than batch 8 (same CPU feed rate) and converged **worse**
(fewer gradient steps/epoch at a fixed LR). More DataLoader workers directly increases
throughput until cores run out. *Prevents:* wasting effort tuning batch size / chasing
VRAM headroom when the real fix is `WORKERS`. *Rule of thumb:* `WORKERS ≈ cores − a
few`; keep `BATCH=8`.

**Scope before the big run (`MAX_CASES`).**
*Why:* the full run (all cases, 40 epochs, 50 rounds × 3 FL methods) is multi-day on
this GPU. A scoped run (`MAX_CASES=200 ROUNDS=20 EPOCHS=20`) confirms the H1–H3 signal
in a few hours first. *Prevents:* discovering a config/heterogeneity mistake after a
multi-day burn.

## 8.7 Tooling & environment

**WSL2 + `uv`, not native Windows / pip.**
*Why:* FLARE needs POSIX (`resource`) + UTF-8 source handling absent on native Windows
(two confirmed crashes); `uv` gives a deterministic, fast, reproducible Linux env.
*Prevents:* FLARE import/runtime crashes; dependency drift.

**`winshim/` is dead code — never on `PYTHONPATH` in WSL.**
*Why:* it's a `resource`-module shim from the abandoned Windows attempt; on Linux it
would *shadow* the real module. *Prevents:* a broken `resource` import that masquerades
as a FLARE bug.
