# Naming and reporting conventions

**Canonical reference. Every document, figure and table in this project follows this page.** If a
document disagrees with this one, this one is right and the document is a bug.

---

## 1. The name collision, and how it is resolved

The code uses `H1`–`H4` for **hospitals** and the study uses `H1`–`H3` for **hypotheses**. They
collide. The collision cannot be removed from the data — `H1`…`H4` are keys inside every
`metrics.jsonl` row and every frozen snapshot, and renaming them would invalidate the SHA-256
manifests and every archived result.

It is therefore resolved **in prose**, with a fixed mapping:

| In code, logs, JSONL | In every document, table and figure | Role |
|---|---|---|
| `H1` | **Site A** | typical scanner |
| `H2` | **Site B** | typical scanner |
| `H3` | **Site C** | typical scanner |
| `H4` | **Site D** | **the outlier** — strongest synthetic scanner shift |

**Hypotheses keep `H1`, `H2`, `H3`** and are never written as bare `H1` next to a hospital column.
When both appear near each other, write "Hypothesis H2" or "**H2**" in bold.

### Rules

1. **Never write a bare `H4` in prose.** Write **Site D**.
2. Any table with a hospital column uses **Site A–D** headers, and carries the mapping in a
   caption or the nearest preceding paragraph if the reader might jump straight to it.
3. `H1`/`H2`/`H3` unqualified always mean hypotheses.
4. Quoting a raw log line or JSON field is the one exception — keep it verbatim (`"test_hospital":
   "H4"`) and gloss it as *(Site D)* on first use in that section.

---

## 2. The three hypotheses

| | Claim | Test |
|---|---|---|
| **H1** | Collaboration helps on average | `mean_dice(FedAvg) ≥ mean_dice(Local-only)` |
| **H2** | The single global model fails the outlier | `dice(FedAvg, Site D) < dice(Local-only, Site D)` |
| **H3** | Personalization recovers the outlier | `mean(FedBN) ≥ mean(FedAvg)` **and** `dice(FedBN, Site D) ≥ dice(FedAvg, Site D)` |

The four methods are always written: **Centralized** (ceiling), **Local-only** (floor),
**FedAvg**, **FedBN**.

---

## 3. Iteration names

| Name | Directory | Status |
|---|---|---|
| **v1** (also "baseline") | `artifacts/snapshots/v1/` | reference |
| **v2** | `artifacts/snapshots/v2/` | partial regression |
| **v3** | `artifacts/snapshots/v3/` | success |
| **v4** | `artifacts/snapshots/v4/` | regression, stopped after 1 run |
| **v5** | `artifacts/snapshots/v5/` | **current best** |

Write **v1**, not "the baseline recipe" and not "the original run", except where the word
*baseline* is being used as the comparison target ("paired against the v1 baseline").

---

## 4. Reporting statistics

Two different questions get asked of the same per-case data, and they must not be confused.

| Question | Tool | What it compares |
|---|---|---|
| Did the **recipe** improve a method? | `scripts/compare_runs.py` | same method, v1 vs vN |
| Does **FedBN beat FedAvg** at one recipe? | `scripts/compute_significance.py` | two methods, same recipe |

### Rules

1. **Always quote the 95% CI beside the p-value**, never the p alone. Several cells here have a CI
   excluding zero with a non-significant p — that combination means the mean shift comes from
   magnitude on a minority of cases, not a consistent per-case win, and it must be visible.
2. **State whether a p-value is corrected.** `compute_significance.py` applies **Holm correction**
   within each report's family of per-hospital tests (12 for single-seed, 36 for three-seed 2D).
   `compare_runs.py` reports **uncorrected** p-values. A corrected and an uncorrected p for the
   same cell are both valid and will differ a lot — label which one is being shown.
3. **Report the sign split** (how many of the 62 volumes moved each way) for any borderline claim.
4. **Never quote a single-round number.** Use the pre-registered estimator: mean over the final
   five rounds (rounds 21–25 for v1–v3, 36–40 for v4–v5). If a single round is unavoidable, state
   the round *and* its neighbours.

---

## 5. Precision and units

- Dice is reported to **four decimals** (`0.8756`). Deltas carry a sign (`+0.0103`).
- p-values in scientific notation to two significant figures (`6.4e-28`), or `p = 0.066` when
  above 0.001.
- "WT / TC / ET" are whole tumour, tumour core, enhancing tumour. Unless stated, a bare Dice figure
  is **WT**.
- Sample sizes: **62 test volumes per site, 248 total.**

---

## 6. Claims that are retracted or restricted

These have been shown wrong or over-stated by later runs. They must not appear in any new document.

| Retracted claim | Status |
|---|---|
| "In 3D all three hypothesis verdicts reverse" | **Refuted.** Both H2 and H3's inequalities hold in 3D under v5. The reversal was an artefact of the v1 training recipe. |
| "FedBN wins in 3D" (v5) | **Over-stated.** The inequality holds but the margin is not significant: uncorrected p = 0.066, Holm-corrected p = 0.53, sign split 39/23. Correct phrasing: *no significant difference in 3D*. |
| "Outlier degradation is caused by FedAvg's weight averaging" | **Refuted.** Centralized has no aggregation step and shows the same effect. The cause is optimizing over a majority-dominated distribution. |
| "Outlier degradation orders monotonically by how much averaging a method does" | **Retracted.** Held at seed 42, failed to replicate at seed 7. |

---

## 7. Which document is authoritative

| Need | Read |
|---|---|
| Everything about v1→v5 in one place | [iteration-report.md](iteration-report.md) |
| FedBN vs FedAvg significance, Holm-corrected | [results-v5-summary.md](results-v5-summary.md) |
| The methodology story behind the changes | [improvements.md](improvements.md) |
| Hyperparameters | [specs.md](specs.md) |
| Evaluation protocol | [experiments.md](experiments.md) |

`project_report.md`, `project_status_report.md` and `capstone_project_report.md` predate v2–v5.
Their results sections describe **v1 only** and are marked as such; do not take numbers from them.
