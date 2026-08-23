# Archived run logs — one directory per iteration

Every training iteration is frozen here with a SHA-256 manifest. **This is the evidence base for
every number in the documentation**, and it is what `scripts/check_docs.py` re-derives figures from.

```
artifacts/
  snapshots/            <- archived, hash-verified, tracked in git
    v1/                    reference baseline        14 runs
    v2/                    partial regression        14 runs
    v3/                    success                    8 runs
    v4/                    regression, stopped        1 run
    v5/                    current best               8 runs
    probe_sched/           schedule probe             1 run
    probe_cap/             capacity probe (rejected)  1 run
  runs/                 <- live working copies, git-ignored
    <run_id>/              untagged runs (the v1 recipe)
    v2/ v3/ v4/ v5/        tagged runs, one subdirectory per --tag
  cache/                <- preprocessed tensors, git-ignored, ~104 GB
  splits/               <- the committed partition manifest
  figures/              <- generated plots
```

## Per-run contents

| File | What it holds |
|---|---|
| `metrics.jsonl` | one row per round × model × test set — the learning curves and the estimator input |
| `per_case.jsonl` | one row per **volume** — the input to every confidence interval and significance test |
| `config.json` | every knob the run actually used, typed |
| `run.log` | timestamped human log |
| `summary.json` | the selected round and its validation score |
| `MANIFEST.json` | *(per snapshot)* SHA-256 per file, git commit, headline numbers |

Checkpoints (`.pt`) are deliberately **not** archived — large, and regenerable from these logs plus
the code. They stay in `artifacts/runs/*/checkpoints/`.

## Verifying

```bash
python scripts/freeze_baseline.py --verify --dest artifacts/snapshots/v1
python scripts/freeze_baseline.py --verify --dest artifacts/snapshots/v5
python scripts/check_docs.py          # re-derives 16 headline figures from these logs
```

Every snapshot is stored with `-text` in `.gitattributes`, so git never rewrites line endings —
otherwise a clone on Linux or Colab would fail every hash on data that had not changed.

## Naming

`v1` is also called *the baseline*, because it is the comparison target every tool defaults to.
The directory is `v1` so the five iterations sort and read uniformly; the word "baseline" survives
in prose only where it means "the thing being compared against". See
[`docs/conventions.md`](../../docs/conventions.md).

Hospitals inside these logs are keyed `H1`–`H4`. In all prose they are **Site A–D**, with
**Site D = `H4`** the outlier.
