# 07 · Evaluation — metrics, fairness, and reading the H1–H3 story

## 7.1 Metrics

**Dice score** per BraTS region — the standard overlap metric,
`Dice = 2|P∩G| / (|P|+|G|)`, from 0 (no overlap) to 1 (perfect):

- **WT** — whole tumor (labels 1∪2∪4)
- **TC** — tumor core (1∪4)
- **ET** — enhancing tumor (4)
- **mean** — the average of the three

Computed by `trainer.evaluate` with MONAI's `DiceMetric` on sigmoid-thresholded
predictions (`sigmoid ≥ 0.5`), **per hospital**, on that hospital's held-out val set.
Because every method routes through the same `case_split` (seed 42), all methods are
scored on the **identical** per-hospital validation cases — the comparison is fair.

## 7.2 Fairness metrics

Personalization is a *fairness* story, so we don't only report the average:

- **`avg_hospital`** — mean Dice across hospitals (the headline average).
- **`worst_hospital`** — the minimum across hospitals (how bad is the worst-served
  site?). Personalization should *raise* this.
- **`std_across_hospitals`** — spread across hospitals (how uneven is quality?).
  Personalization should *lower* this.

A method can win on `avg_hospital` while failing the hospitals that differ most; the
worst-case and spread are what expose that.

## 7.3 The personalization gain — the core figure

For each hospital: **`gain = Dice(FedBN) − Dice(FedAvg)`**.

This is *the* result. `analyze.py` renders it as `personalization_gain.png` — a bar
per hospital, green for positive (FedBN helped), red for negative (FedBN hurt). **H3
is exactly "these bars are positive, especially for the worst hospitals."**

## 7.4 The result schema (`results.py`)

One JSON per `(method, site)` at `results/<method>/<site>.json`:

```json
{
  "method": "fedbn",
  "site": "site-4",
  "n_train": 1623,
  "n_val": 402,
  "dice_mean": 0.71,
  "dice_TC": 0.68,
  "dice_WT": 0.80,
  "dice_ET": 0.65
}
```

`n_train`/`n_val` are **slice** counts (not cases). Every experiment writes this same
schema so `analyze.py` can compare them uniformly. (`_index_*.csv` files alongside are
slice-index caches, gitignored.)

## 7.5 `analyze.py` outputs

Run automatically as step 8, or by hand: `uv run python analyze.py`.

| File | What it is |
|------|------------|
| `results/per_hospital.csv` | rows = hospitals, cols = methods, values = mean Dice (or pick `--metric dice_WT`) |
| `results/summary.csv` | per method: `avg_hospital`, `worst_hospital`, `std_across_hospitals` |
| `results/comparison.png` | grouped bars — per-hospital Dice by method |
| `results/personalization_gain.png` | FedBN − FedAvg per hospital (the core figure) |

The console also prints the per-hospital table, the summary, and the
per-hospital + mean + worst-hospital personalization gain.

Method display order (floor → ceiling): `local, fedavg, fedprox, finetune,
personal_head, fedbn, centralized`.

## 7.6 How to read it — the H1–H3 checklist

Open `summary.csv` and `per_hospital.csv` and check, in order:

- **H1 — collaboration helps.** In `summary.csv`, `fedavg.avg_hospital ≥
  local.avg_hospital`. If FedAvg isn't at least matching local-only on average,
  federation isn't paying off (check rounds/epochs, or that training converged).
- **H2 — heterogeneity hurts the outliers.** In `per_hospital.csv`, FedAvg should be
  fine on **site-1** (the reference scanner) but visibly **worse on the outlier
  sites** (site-4/5/6 under the synthetic shift). If FedAvg is uniformly fine across
  sites, there isn't enough heterogeneity — the shift is too weak or (fatally) the
  split was IID with no shift.
- **H3 — FedBN recovers them for free.** FedBN ≥ FedAvg on the outlier sites
  (positive bars in `personalization_gain.png`) **and** FedBN's `avg_hospital` ≥
  FedAvg's, `worst_hospital` up, `std_across_hospitals` down. That's the whole claim:
  personalization helps the hurt hospitals **without** hurting the average.

## 7.7 Sanity anchors (is a run even valid?)

Before interpreting anything, confirm the run is real:

- **Centralized ceiling ~0.79–0.81 mean Dice.** If it's near zero, the run is
  under-trained/broken and *nothing* below it means anything. (A historical bad run
  had FL `num_rounds=1` on an IID 2-client split and the ceiling collapsed to 0.079 —
  a cautionary tale.)
- **6 hospitals present** (`site-1`…`site-6`) for a `--n-clients 6` run. If you only
  see `site-1`/`site-2`, those are stale smoke-run results.
- **FedBN ≠ FedAvg exactly.** If they're identical, either there's no heterogeneity
  (IID + no shift) or FedBN's local-persistence broke (invariant #5) — FedBN has
  degenerated into FedAvg.

## 7.8 Interpreting expected patterns under the synthetic shift

- **site-1 (reference scanner):** FedAvg, centralized, and FedBN should all do well and
  be close — there's no shift to personalize, so FedBN ≈ FedAvg here *by design*, and
  that's correct (H3's "without hurting the matched hospital").
- **site-4/5/6 (outliers):** FedAvg and centralized should sag; FedBN should recover
  them. The bigger the profile's shift, the bigger the expected FedBN gain — a gradient
  you can literally see across sites.
- **local-only:** typically below FedAvg on average (H1) but sometimes competitive on a
  strong outlier (its model is at least *specialized*, if data-starved).
- **fine-tune:** usually a strong personalizer too — a useful upper-ish bound on what
  "just adapt the global model" buys, to contextualize FedBN's cheaper, during-training
  gain.
