# 05 · Methods — every model in the matrix, and how FL is wired

This chapter details each method in the experiment matrix, the shared model, and the
FLARE mechanics. The unifying design principle: **the server always runs plain
FedAvg**, and everything interesting happens **client-side**.

## 5.1 The model

A **MONAI 2D U-Net**, built by `model.build_unet` and always instantiated via the
`BratsUNet` wrapper:

| Setting | Value | Why |
|---------|-------|-----|
| `spatial_dims` | 2 | 2D slices → fits 4 GB VRAM |
| `in_channels` | 4 | FLAIR/T1/T1ce/T2 stacked |
| `out_channels` | 3 | overlapping regions TC/WT/ET |
| `channels` | (16,32,64,128,256) | 5-level encoder/decoder, small enough for 4 GB |
| `strides` | (2,2,2,2) | downsample per level |
| `num_res_units` | 2 | residual units per level |
| `norm` | **batch** | **the setting FedBN targets** — scanner differences live in BN stats |

- **Loss:** `DiceLoss(sigmoid=True)` — multi-label (regions overlap), so a **sigmoid**
  per channel, not softmax.
- **Metric:** `DiceMetric(include_background=True, reduction="mean_batch")`, thresholded
  at `sigmoid ≥ 0.5`.
- **`BratsUNet` wrapper:** all-default constructor args so FLARE can reconstruct the
  server model from JSON (invariant #1). State-dict keys are all `net.*`.

Every method — centralized, local, FedAvg, FedProx, FedBN, personal-head, fine-tune —
uses this **same** architecture and the **same** `trainer.py` loop, so any Dice
difference is attributable to the *federation strategy*, not the model.

## 5.2 The shared training loop (`trainer.py`)

One code path for all methods:

- **AMP mixed precision** (`torch.autocast` + `GradScaler`) — halves activation memory,
  essential on 4 GB.
- **NaN-skip** — if a batch's loss isn't finite, skip it (don't let one bad batch
  corrupt the weights). Reported as `skipped` counts.
- **Grad clipping** — max-norm 1.0, for stability.
- **Optimizer** — Adam, LR default `5e-4`, **cosine-annealed** over the epochs (the
  centralized/local loops). (LR `1e-3` caused NaN blow-ups earlier — hence 5e-4.)
- **FedProx term** — optional `(μ/2)‖w − w_global‖²` added to the loss (§5.6).

`evaluate()` returns per-region + mean Dice, in eval mode, no grad.

## 5.3 Centralized — the ceiling

`train_centralized.py`. Trains one model on the **union of all hospitals' train
cases** (`partitioned_splits`), validating on the union of their val cases. This is
the privacy-*violating* upper bound: what you could get if the data could be pooled.
Step 2 (`evaluate.py`) then scores that one checkpoint on **each hospital's** val set
so it appears per-hospital in the comparison.

- Under the synthetic shift, the centralized model sees **all scanner profiles mixed**
  and learns an averaged BatchNorm — so, like FedAvg, it is expected to underperform
  on the outlier hospitals. That's the point: even the ceiling isn't personalized.

## 5.4 Local-only — the floor

`run_local_baselines.py`. Each hospital trains its **own** model on **only** its own
cases, no collaboration. Shows what a hospital gets alone. To keep the compute budget
comparable to FL, run `--epochs ≈ (FL rounds × FL local-epochs)`.

## 5.5 FedAvg — the federated baseline

`run_fedavg.py --method fedavg`. Standard federated averaging via FLARE (§5.9). Every
client keeps **nothing** local (`keep_local_keys → {}`), so all clients converge to
the single global model. This is the baseline personalized FL must beat on the
outliers (H2/H3).

## 5.6 FedProx — non-IID-stabilized baseline

`run_fedavg.py --method fedprox --prox-mu 0.01`. FedAvg plus a **proximal term** that
penalizes local weights drifting from the round's incoming global weights
(`trainer._prox_term`). Still **one** global model (not personalization) — it just
trains more stably under mild non-IID. `--prox-mu` controls the strength.

## 5.7 FedBN — the primary personalization method

`run_fedavg.py --method fedbn --personalization fedbn`. Identical FLARE job to FedAvg,
except each client **keeps its normalization layers local** (`keep_local_keys →
norm_param_names`: BatchNorm weight/bias **and** running_mean/running_var/
num_batches_tracked). So:

- The convolutional filters (shared "what a tumor looks like") are federated normally.
- Each hospital's **BatchNorm** — its scanner calibration — never gets averaged away.

**Persistence across rounds (invariant #5).** FLARE may re-run the client script each
round, which would reset the local BN. So `brats_client.py` saves the kept-local
params to `results/<method>/{site}_local.pt` after training and restores them before
eval next round. This lets BN **accumulate** its specialization over rounds. Remove it
and FedBN silently degenerates into FedAvg.

## 5.8 Personal-head and Fine-tune

- **Personal-head** (`--personalization personal_head`) — keep norm layers **plus the
  final output conv** local (`norm ∪ head_param_names`). Shared body, private
  segmentation head (FedRep-style). A comparison against FedBN.
- **Fine-tune** (`finetune.py`) — a *post-hoc* strategy: take the converged FedAvg
  **global** checkpoint and train a few more epochs (`--ft-epochs`) on each hospital's
  own data. Simple and strong; the pragmatic personalization baseline. It starts each
  hospital's eval from the global model's score and keeps the best.

## 5.9 The FLARE job (`run_fedavg.py`) and client (`brats_client.py`)

**The job.** `run_fedavg.py` builds a FLARE `FedJob`:
- Server: `FedAvg(num_clients, num_rounds)` workflow + `PTModel(BratsUNet())` (the
  initial global model — reconstructable because of the `BratsUNet` wrapper).
- One `ScriptRunner(script="fl/brats_client.py", ...)` per client, all pointed at the
  same deterministic partition source (`--n-clients N` or `--fets-csv`), each with its
  own `--client-index`.
- Runs in the **Simulator** with `threads=1` (**one client on the GPU at a time** —
  4 GB-safe) and `gpu=0`.

**The client's per-round loop** (`brats_client.py`), in order — this ordering *is* the
eval-before-train protocol (invariant #3):

1. `flare.receive()` the global model.
2. `load_global(model, params, keep_local)` — overwrite everything **except** the
   kept-local keys (this is where the personalization strategy takes effect).
3. Restore accumulated `{site}_local.pt` (FedBN persistence).
4. **Evaluate the received model** on this hospital's val set → write
   `results/<method>/<site>.json`. *(This is the reported number: the true global
   model for FedAvg, the genuine personalized model for FedBN.)*
5. `local_train(...)` on this hospital's data (with the FedProx anchor if `--prox-mu>0`).
6. Save `{site}_local.pt` (kept-local params for next round).
7. `flare.send()` the updated weights back to the server.

The server averages step-7 weights into the next round's global model. Personalization
never touches the server — it is entirely the choice of *which keys step 2 leaves
alone*.

## 5.10 Method summary table

| Method | Server | Client keeps local | Type | Script / flag |
|--------|--------|--------------------|------|---------------|
| Centralized | — (pooled) | — | ceiling | `train_centralized` + `evaluate.py` |
| Local-only | — (isolated) | everything | floor | `run_local_baselines.py` |
| FedAvg | FedAvg | nothing | baseline | `--method fedavg` |
| FedProx | FedAvg | nothing (+prox term) | baseline | `--method fedprox --prox-mu` |
| **FedBN** | FedAvg | **norm layers** | **personalized (primary)** | `--method fedbn --personalization fedbn` |
| Personal-head | FedAvg | norm + output conv | personalized | `--personalization personal_head` |
| Fine-tune | FedAvg then local | (post-hoc) whole model | personalized | `finetune.py --ft-epochs` |
