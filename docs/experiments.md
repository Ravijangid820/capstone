# Experiments

The operational plan: what runs, how they're evaluated, and exactly how each hypothesis is measured.

## 1. Experiment matrix

Run for the **2D** backbone first; repeat for **3D** if the feasibility spike passes.

| # | Run | Trains on | Produces |
|---|---|---|---|
| E0 | **Centralized** | pooled ~1000 train | ceiling reference |
| E1 | **Local-only** ×4 | each hospital's own train | floor (one model per hospital) |
| E2 | **FedAvg** | federated across 4 hospitals | global model (H1, H2) |
| E3 | **FedBN** | federated, BN kept local | personalized model (H3) |

All four share the **same committed split**, the same seed, and the same test sets — only the training
*procedure* differs, so the comparison is clean.

## 2. Evaluation protocol

```mermaid
flowchart TD
    M["a trained model"] --> T1["Hospital 1 test"]
    M --> T2["Hospital 2 test"]
    M --> T3["Hospital 3 test"]
    M --> T4["Hospital 4 test (outlier)"]
    T1 --> D["Dice WT / TC / ET<br/>per hospital"]
    T2 --> D
    T3 --> D
    T4 --> D
    D --> AVG["mean across hospitals"]
    D --> OUT["outlier score"]
```

- **Metric:** Dice on the three regions **WT** (whole tumor), **TC** (tumor core), **ET** (enhancing).
- **Per volume, never per slice.** For 2D we predict every axial slice, stack them into a volume, and
  score *that*. The mean of per-slice Dice is not the per-case Dice — it inflates scores, because
  empty slices score 1.0 for free.
- **Empty ground truth (matters for ET).** BraTS convention: empty prediction on empty GT scores 1.0;
  any false positive scores 0.0. Without this the ET column is meaningless.
- **Two views:** per-hospital Dice (for the outlier claims) and the mean across hospitals (for the average
  claims). Reported per method.
- **FedBN eval:** each hospital uses the shared weights + *its own* BN layers.
- **Timing:** score **after aggregation, before the next round's local training** — see
  [federated-learning](federated-learning.md).
- Every score is appended to `metrics.jsonl`; tables/plots are generated from that file.

### Scope — diagonal, plus a cross-matrix for local-only

The headline numbers are the **diagonal**: each hospital's model on its own test set
(`model_hospital == test_hospital`). That is all H1/H2/H3 require.

Additionally, at the **final round only**, every local-only model is scored on all four test sets — a
4×4 matrix. Its off-diagonal cells are direct evidence the synthetic shift creates a genuine domain
gap (H4's model should collapse on H1–H3). It is run once rather than per round, since it costs 16×
a diagonal evaluation.

## 3. How each hypothesis is measured

| Hyp. | Claim | Concrete test |
|---|---|---|
| **H1** | collaboration helps on average | `mean_dice(FedAvg) ≥ mean_dice(Local-only)` |
| **H2** | the global model fails the outlier | `dice(FedAvg, H4) < dice(Local-only, H4)` |
| **H3** | personalization recovers the outlier | `mean_dice(FedBN) ≥ mean_dice(FedAvg)` **and** `dice(FedBN, H4) ≥ dice(FedAvg, H4)` (closing the H2 gap) |

Centralized (E0) frames all of the above as "how close to the pooled ceiling did we get."

> **Matched compute — what makes H1 a real test.** FedAvg gives each hospital `R × E` local epochs.
> Local-only therefore also trains `R × E` epochs, so the *only* difference between them is
> aggregation. Had local-only trained for `E` epochs, H1 would be near-guaranteed and would merely be
> measuring FedAvg's ~R× larger training budget. Centralized likewise trains `R × E` epochs on the
> pooled set.

> **Identical init.** All four methods start from the same seeded random weights (`build_model` seeds
> `torch` before construction), so no comparison is confounded by initialization luck.

## 4. Results (to be filled)

### 2D backbone — mean Dice across hospitals

| Method | WT | TC | ET |
|---|---|---|---|
| Centralized (ceiling) | – | – | – |
| Local-only | – | – | – |
| FedAvg | – | – | – |
| FedBN | – | – | – |

### 2D backbone — per-hospital WT Dice (outlier = H4)

| Method | H1 | H2 | H3 | H4 (outlier) |
|---|---|---|---|---|
| Local-only | – | – | – | – |
| FedAvg | – | – | – | – |
| FedBN | – | – | – | – |

*(3D tables added if the feasibility spike passes.)*

## 5. 3D feasibility spike (gate before E0–E3 in 3D)

Before running the full 3D matrix, one measurement decides go/no-go:

- Train a single 3D U-Net on the T4; record VRAM, per-epoch time, and projected full-study wall-clock vs.
  Colab's session limit.
- **Pass →** run E0–E3 in 3D and add a "does the story hold in 3D?" comparison.
- **Fail →** report the spike numbers; 2D remains the deliverable.

Local probe (RTX 3050) already shows 3D *fits in memory* at 96³/128³; the spike is about *speed at scale*.
See [specs](specs.md) for the measured numbers.
