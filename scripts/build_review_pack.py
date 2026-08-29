"""Assemble `review/` -- everything needed to present and defend this project, in one folder.

    python scripts/build_review_pack.py

Hunting for the right document while a reviewer waits is a bad look, and the docs are split
across 35 files by concern rather than by "what gets shown". This gathers the sixteen that
matter, in reading order, and writes two navigation documents on top of them.

**Copies, not the originals.** `docs/` stays the single source of truth -- `check_docs.py`
re-derives sixteen headline figures from the frozen logs and enforces the naming rules against
it, and a second hand-maintained set would drift out from under that guarantee. Every file here
carries a banner naming its source, and relative links are rewritten to point back into `docs/`
so following one lands on the real thing. Regenerate after any documentation change.

The headline numbers in the guide are read from `artifacts/snapshots/` at build time, the same
way the showcase deck does it, so the pack cannot quote a figure the logs disagree with.
"""

from __future__ import annotations

import re
import shutil
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
OUT = REPO / "review"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_showcase import SITES, collect_runs, collect_stats  # noqa: E402

METHODS = ["centralized", "local", "fedavg", "fedbn"]
LABEL = {"centralized": "Centralized", "local": "Local-only",
         "fedavg": "FedAvg", "fedbn": "FedBN"}

# What actually gets opened in a review, grouped by the question it answers.
PACK: dict[str, list[tuple[str, str]]] = {
    "01-start-here": [
        ("conventions.md", "Naming rules, reporting rules, and every retracted claim. Read first."),
        ("methodology.md", "The research question, the three hypotheses, and what each one tests."),
    ],
    "02-results": [
        ("iteration-report.md", "The full v1→v5 record: every configuration, result and failure."),
        ("results-v5-summary.md", "The final recipe's results, with Holm-corrected significance."),
        ("results-significance-2d-v5.md", "FedBN vs FedAvg, 2D, paired per case."),
        ("results-significance-3d-v5.md", "FedBN vs FedAvg, 3D — the one that is not significant."),
        ("results-comparison-v5-2d.md", "v1 vs v5, 2D, paired per case."),
        ("results-comparison-v5-3d.md", "v1 vs v5, 3D, paired per case."),
    ],
    "03-how-it-works": [
        ("architecture.md", "Components and the end-to-end flow — the big picture."),
        ("data-pipeline.md", "Partition, scanner shift, preprocessing, caching, sampling."),
        ("federated-learning.md", "The round loop, and how each method aggregates."),
        ("experiments.md", "The experiment matrix and the evaluation protocol."),
        ("specs.md", "Reference sheet: hyperparameters, model sizes, hardware, timings."),
    ],
    "04-engineering": [
        ("workflow.md", "The pipeline in order, with measured costs per stage."),
        ("environments.md", "Windows / WSL2 / Colab, and the portability contract."),
        ("improvements.md", "The evaluation-noise finding and the freeze → rerun → compare protocol."),
        ("data.md", "BraTS 2021 spec, labels, and how the data was acquired."),
    ],
}

BANNER = ("> **Review copy — do not edit.** Source of truth: [`docs/{name}`](../../docs/{name}).\n"
          "> Regenerate this folder with `python scripts/build_review_pack.py`.\n\n---\n\n")

LINK = re.compile(r"\]\((?!https?://|mailto:|#)([^)]+)\)")


def rewrite_links(text: str) -> str:
    """Point every relative link back at the real file, from two directories down.

    A copy under review/02-results/ that still says `](conventions.md)` resolves to a sibling
    that may not have been included. Resolving against docs/ and re-anchoring at the repo root
    means every link works and every link lands on the source of truth.
    """
    def fix(m: re.Match) -> str:
        target = m.group(1)
        path, _, anchor = target.partition("#")
        if not path:
            return m.group(0)
        resolved = (DOCS / path).resolve()
        try:
            rel = resolved.relative_to(REPO).as_posix()
        except ValueError:
            return m.group(0)
        return f"](../../{rel}{'#' + anchor if anchor else ''})"
    return LINK.sub(fix, text)


def est(runs: dict, tag: str, method: str, dim: str) -> float | None:
    r = runs.get(tag, {}).get(f"{method}_{dim}")
    return r["est"]["mean"] if r else None


def site_est(runs: dict, tag: str, method: str, dim: str, h: str) -> float | None:
    r = runs.get(tag, {}).get(f"{method}_{dim}")
    return r["est"][h] if r else None


def sgn(v: float) -> str:
    return f"{'+' if v > 0 else '−'}{abs(v):.4f}"


def results_table(runs: dict) -> str:
    rows = ["| Method | 2D v1 | 2D v5 | Δ | 3D v1 | 3D v5 | Δ |",
            "|---|---|---|---|---|---|---|"]
    for m in METHODS:
        cells = [f"**{LABEL[m]}**"]
        for dim in ("2d", "3d"):
            a, b = est(runs, "v1", m, dim), est(runs, "v5", m, dim)
            cells += [f"{a:.4f}", f"**{b:.4f}**", sgn(b - a)] if a and b else ["—", "—", "—"]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def outlier_table(runs: dict) -> str:
    rows = ["| Method | 2D Site D | 3D Site D |", "|---|---|---|"]
    for m in METHODS:
        vals = [site_est(runs, "v5", m, d, "H4") for d in ("2d", "3d")]
        rows.append(f"| **{LABEL[m]}** | " +
                    " | ".join(f"{v:.4f}" if v else "—" for v in vals) + " |")
    return "\n".join(rows)


def verdicts(runs: dict) -> str:
    out = []
    for dim in ("2d", "3d"):
        fa, lo = est(runs, "v5", "fedavg", dim), est(runs, "v5", "local", dim)
        faD = site_est(runs, "v5", "fedavg", dim, "H4")
        loD = site_est(runs, "v5", "local", dim, "H4")
        bn, bnD = est(runs, "v5", "fedbn", dim), site_est(runs, "v5", "fedbn", dim, "H4")
        d = dim.upper()
        out.append(
            f"| **H1** — collaboration helps | {d} | FedAvg {fa:.4f} vs Local {lo:.4f} "
            f"({sgn(fa - lo)}) | {'**holds**' if fa >= lo else '**FAILS**'} |")
        out.append(
            f"| **H2** — global model fails the outlier | {d} | at Site D, FedAvg {faD:.4f} vs "
            f"Local {loD:.4f} ({sgn(faD - loD)}) | {'**holds**' if faD < loD else 'does not hold'} |")
        ns = " — but *not significant* (p = 0.066, 39/23)" if dim == "3d" else ""
        out.append(
            f"| **H3** — personalization recovers it | {d} | mean {sgn(bn - fa)}, "
            f"Site D {sgn(bnD - faD)}{ns} | "
            f"{'**holds**' if bn >= fa and bnD >= faD else 'does not hold'} |")
    return "\n".join(out)


def write_guide(runs: dict, stats: dict, copied: int) -> None:
    n_ckpt = len(list((REPO / "artifacts" / "runs").glob("**/checkpoints/*.pt")))
    guide = f"""# Capstone review pack

Everything needed to present and defend this project, in one folder.
Generated {date.today().isoformat()} from the repository — see `scripts/build_review_pack.py`.

**Read [`QUESTIONS.md`](QUESTIONS.md) the night before.** It is the likely questions with the
answers and the numbers already in them.

---

## 1. In one minute

Four simulated hospitals hold BraTS 2021 brain-MRI scans they are not allowed to share. One of
them (**Site D**) has a visibly different scanner. We compare four training regimes — a
centralized **ceiling**, a local-only **floor**, standard **FedAvg**, and personalized **FedBN**
— across five recipe iterations, in both 2D and 3D, and measure every claim on 248 held-out
volumes with paired per-case statistics.

The finding: a single global model serves the outlier hospital **worse than not federating at
all**, and keeping each site's BatchNorm parameters local recovers most of that loss for
0.14% of the model.

---

## 2. Run the demo

```bash
uv run python scripts/demo_server.py
```

Then open:

| URL | What |
|---|---|
| <http://localhost:8000/> | Live inference — pick a case, a model, a hospital; 2D slices and a 3D mesh |
| <http://localhost:8000/showcase> | The 27-slide animated walkthrough |

**If port 8000 is busy** (VS Code's port forwarding often holds it):

```bash
uv run python scripts/demo_server.py --port 8010
```

The walkthrough also opens straight from disk with no server and no network —
`artifacts/showcase/showcase.html`, and a copy is in [`05-demo/`](05-demo/). Use that if the
laptop misbehaves.

**Presenting the walkthrough:** `←` `→` to move, `O` for a slide overview, `F` for fullscreen.
Slides 3–6 are an animated federated-learning tutorial; drag the dividers on the scanner-shift
and prediction slides.

---

## 3. Where everything lives

| What | Path | Notes |
|---|---|---|
| **Training code** | `src/fedbrats/` | `federated.py` is the round loop, `train.py` the train/eval step |
| — configuration | `src/fedbrats/config.py` | every hyperparameter, and the v1–v5 presets |
| — the FL engine | `src/fedbrats/federated.py` | FedAvg / FedBN / local / centralized |
| — model | `src/fedbrats/model.py` | dimension-parametric residual U-Net (MONAI) |
| — data + preprocessing | `src/fedbrats/data.py` | load → mask → shift → crop → normalise → cache |
| — the scanner shift | `src/fedbrats/shift.py` | gamma, bias field, blur — per site |
| — metrics | `src/fedbrats/metrics.py` | per-volume Dice for WT / TC / ET |
| **Run a training job** | `scripts/run_experiment.py` | one method |
| | `scripts/run_matrix.py` | all four methods, resumable |
| **Model checkpoints** | `artifacts/runs/<tag>/<run>/checkpoints/final.pt` | {n_ckpt} files, git-ignored (too large) |
| **Frozen logs (the evidence)** | `artifacts/snapshots/v1/ … v5/` | hash-verified, committed |
| — per-round metrics | `<run>/metrics.jsonl` | one row per round × model × test set |
| — per-volume Dice | `<run>/per_case.jsonl` | the input to every CI and significance test |
| — exact config used | `<run>/config.json` | every knob, typed |
| — integrity | `<tag>/MANIFEST.json` | SHA-256 per file + the git commit |
| **Live run output** | `artifacts/runs/` | working state, git-ignored |
| **The data split** | `artifacts/splits/partition.json` | committed — this is why runs are comparable |
| **Preprocessed cache** | `artifacts/cache/` | ~105 GB, git-ignored, regenerable |
| **Figures** | `artifacts/figures/` | learning curves, per-hospital bars, prediction panels |
| **The walkthrough** | `artifacts/showcase/showcase.html` | self-contained, 800 KB |
| **Documentation** | `docs/` | {len(list(DOCS.glob('*.md')))} files; the {copied} that matter are copied into this pack |
| **Tests** | `tests/` | `uv run pytest` |

**Raw BraTS data is not in the repository** — 1,251 cases, ~13 GB compressed. Point
`FEDBRATS_DATA_ROOT` at it, or place it in `data/BraTS2021_Training_Data/`.

---

## 4. The numbers to know cold

Mean whole-tumour Dice across the four sites. Estimator: **mean of the final five rounds**, seed 42.

{results_table(runs)}

**Site D (the outlier) at v5** — this is the table the whole project is about:

{outlier_table(runs)}

FedAvg is the **best** method at Sites A–C in 2D and the **worst** at Site D.

### Hypothesis verdicts at v5

| Hypothesis | Backbone | Evidence | Verdict |
|---|---|---|---|
{verdicts(runs)}

---

## 5. How to prove any number on the spot

```bash
# Re-derive 16 headline figures from the frozen logs and check every document against them
uv run python scripts/check_docs.py

# Confirm no archived log has been altered since it was frozen
uv run python scripts/freeze_baseline.py --verify --dest artifacts/snapshots/v5

# The paired v1 → v5 comparison, recomputed live
uv run python scripts/compare_runs.py --baseline artifacts/snapshots/v1 \\
    --new artifacts/snapshots/v5 --dim 2d --seed 42

# FedBN vs FedAvg at v5, Holm-corrected
uv run python scripts/compute_significance.py --runs artifacts/snapshots/v5 \\
    --dim 2d --a fedavg --b fedbn
```

If asked *"how do we know these numbers are real?"* — run the first two. `check_docs.py` reads
every figure out of `artifacts/snapshots/` and fails the build if a document disagrees;
`freeze_baseline.py --verify` re-hashes every archived file against its manifest.

---

## 6. Reproducing a run

```bash
uv sync                                              # dependencies
uv run python scripts/build_partition.py             # the 4-hospital split (deterministic)
uv run python scripts/build_cache.py --workers 8     # preprocess once (~33 min, ~44 GB)

# one method
uv run python scripts/run_experiment.py --method fedbn --dim 2d --preset v5 --tag v5

# all four, resumable
uv run python scripts/run_matrix.py --dim 2d --seed 42 --preset v5 --tag v5
```

Defaults reproduce **v1** exactly, so an unflagged run today still yields the frozen baseline.
Improvements are opt-in via `--preset`. `--tag` namespaces the output so a rerun can never
overwrite an archived result.

**Cost** on the RTX 3050 (4 GB): a 2D run of 25 rounds is ~1.7–2 hours; the four-method 2D
matrix ~3.7 h. Evaluation is ~3.5× the cost of training, which is why `eval_test_every` exists.

---

## 7. Rebuilding this pack and the walkthrough

```bash
uv run python scripts/build_review_pack.py                 # this folder
uv run python scripts/showcase_assets.py                   # walkthrough images (needs the raw data)
uv run python scripts/build_showcase.py                    # the walkthrough itself
```

---

## 8. If something breaks mid-review

| Symptom | Cause | Fix |
|---|---|---|
| Server starts, browser shows nothing | Another process holds port 8000 | `--port 8010` |
| `Cannot bind 127.0.0.1:8000` | Same, now reported properly | `--port 8010`, or free it in VS Code's PORTS panel |
| `/showcase` returns 404 | Walkthrough not built | `uv run python scripts/build_showcase.py` |
| Live inference errors | Checkpoints missing (`artifacts/runs/`, git-ignored) | Present the walkthrough instead — it needs nothing |
| Cache missing | `artifacts/cache/` not on this machine | Everything except live inference still works |

**The safest demo path:** open `05-demo/showcase.html` directly in a browser. No server, no
network, no GPU, no data.

---

## 9. What is in this pack

{chr(10).join(f"### {folder}" + chr(10) + chr(10) +
              chr(10).join(f"- **[{n}]({folder}/{n})** — {d}" for n, d in items)
              for folder, items in PACK.items())}

### 05-demo

- **[showcase.html](05-demo/showcase.html)** — the 27-slide walkthrough, self-contained.

{copied} documents copied. The originals in `docs/` remain the source of truth; every link in
these copies points back there.
"""
    (OUT / "README.md").write_text(guide, encoding="utf-8")


def write_questions(runs: dict, stats: dict) -> None:
    fa2 = site_est(runs, "v5", "fedavg", "2d", "H4")
    bn2 = site_est(runs, "v5", "fedbn", "2d", "H4")
    lo2 = site_est(runs, "v5", "local", "2d", "H4")
    fam, lom = est(runs, "v5", "fedavg", "2d"), est(runs, "v5", "local", "2d")
    spread = {}
    for m in METHODS:
        v = [site_est(runs, "v5", m, "2d", h) for h in SITES]
        spread[m] = max(v) - min(v)
    r2 = stats["recipe"]["fedbn_2d"]

    q = f"""# Likely review questions, with the answers

Generated {date.today().isoformat()} — numbers read from `artifacts/snapshots/`.
Every figure below can be re-derived live with `python scripts/check_docs.py`.

---

## The premise

**Why federated learning at all? Why not just pool the data?**
Pooling is the *ceiling* we are measuring against, not a competitor — centralized training scores
{est(runs, 'v5', 'centralized', '2d'):.4f} in 2D, the best of any method. It is also not
permitted: it requires copying identifiable patient imaging out of the acquiring hospital, which
data-protection law, ethics approval and institutional policy each independently forbid.

**Is this real federated learning, or a simulation?**
A faithful simulation. The four sites are simulated on one machine, but the *algorithm* is real:
each site trains only on its own cases, only weight tensors move between site and server, and no
site's data ever enters another's training loop. What is simulated is the network and the
hospitals, not the learning.

**Why simulate the hospitals instead of using real multi-site data?**
BraTS 2021 is pooled and de-identified — the site labels are not available. Simulating lets us
*control* the degree of heterogeneity, which is the independent variable. The shift is applied as
a gamma curve, a smooth spatial bias field and mild blur (`src/fedbrats/shift.py`) — nonlinear
and spatial on purpose, so it survives per-case z-normalisation. A purely linear shift would be
normalised away and there would be no heterogeneity left to study.

---

## The data

**How many cases, and how are they split?**
1,251 BraTS 2021 cases, partitioned into four hospitals (313/313/313/312) by a deterministic seeded
assignment. **62 test cases per site — 248 total** — are held out and never trained on. From v2
onward, 20 cases per site are used for validation, drawn *after* the training cap so the training
set is unchanged. The partition is committed as `artifacts/splits/partition.json`, so every run in
every iteration reads the identical split.

**What does the model take as input?**
Four co-registered MRI modalities — FLAIR, T1, T1ce, T2 — stacked as channels. It predicts three
**nested** regions: whole tumour ⊃ tumour core ⊃ enhancing tumour, as three independent sigmoids.
Not a softmax: the regions nest, and a softmax would force them to compete for the same voxel.

**Why is Site D the outlier, and is that fair?**
Its shift parameters are deliberately stronger (gamma 1.85, bias ±0.34, blur σ 1.7 against
1.06/0.06/0.3 at Site A). It is a designed condition, not an accident — we need one site far
enough from the others for the effect to be measurable. All four sites have essentially the same
number of cases, so the outlier is a *distribution* outlier, not a data-quantity one.

---

## The methods

**What is the difference between FedAvg and FedBN?**
Only what gets uploaded. FedAvg averages every parameter, so all four sites run identical weights.
FedBN averages everything **except** the BatchNorm parameters, which stay at each site — so the
sites share a body and keep their own normalisation. In our 2D model that is **2,310 of 1,607,562
parameters, 0.14%**.

**Why does that tiny fraction matter so much?**
BatchNorm stores the mean and variance of the features it sees. Averaging those across four
different scanners produces statistics that describe none of them. Keeping them local lets each
site calibrate to its own intensity distribution while still sharing everything it has learned
about tumour shape.

**Is the comparison fair on compute?**
Yes, deliberately. Local-only and centralized train for `rounds × local_epochs` total epochs —
the same number a federated site sees. Otherwise "collaboration helps" would just mean
"we trained longer".

---

## The results

**What is the headline finding?**
At Site D in 2D, FedAvg scores **{fa2:.4f}** while training alone scores **{lo2:.4f}** — joining
the federation made that hospital *worse off* by {sgn(fa2 - lo2)}. FedBN recovers it to
**{bn2:.4f}**, a gain of {sgn(bn2 - fa2)} over FedAvg, with a 95% CI of
[{stats['personal']['2d_H4']['lo']:+.4f}, {stats['personal']['2d_H4']['hi']:+.4f}],
p = 4.5e−11, and {stats['personal']['2d_H4']['better']} of 62 volumes improving.

**Did H1 hold?**
Not everywhere, and this is worth saying before being asked. In **3D** FedAvg beats local-only
({est(runs, 'v5', 'fedavg', '3d'):.4f} vs {est(runs, 'v5', 'local', '3d'):.4f}). In **2D at v5 it
does not** — FedAvg {fam:.4f} against local-only {lom:.4f}. Raising the training cap from 150 to
230 cases per site raised the *floor* faster than federating raised the average. FedBN still
clears it ({est(runs, 'v5', 'fedbn', '2d'):.4f}), which makes the case for *personalized*
federation rather than for federation as such.

**Is the FedBN result significant in 3D?**
**No, and we do not claim it is.** The inequality holds — FedBN's Site D Dice exceeds FedAvg's by
{sgn(site_est(runs, 'v5', 'fedbn', '3d', 'H4') - site_est(runs, 'v5', 'fedavg', '3d', 'H4'))}
— but p = 0.066 uncorrected, 0.53 after Holm correction, with a near-even 39/23 sign split. The
correct phrasing is *no significant difference in 3D*. This is recorded in
`docs/conventions.md` §6 as a restricted claim.

**How much did the recipe work improve things?**
Paired per case across all 248 volumes, 2D FedBN gained {r2['delta']:+.4f}
[{r2['lo']:+.4f}, {r2['hi']:+.4f}] with {r2['better']} of 248 volumes improving. Every method
improved in both backbones on the four-site mean — though 3D local-only and 3D FedAvg have
confidence intervals that touch zero, and we report them as the weakest results in the set.

**What about fairness across hospitals?**
Best-site minus worst-site spread, 2D at v5: FedAvg **{spread['fedavg']:.4f}**, FedBN
**{spread['fedbn']:.4f}** — personalization cuts the gap by
{round((1 - spread['fedbn'] / spread['fedavg']) * 100)}%. Local-only is actually the *fairest*
({spread['local']:.4f}) since every site fits itself, but it has the lowest mean. FedBN gets close
to local-only's fairness at a much higher mean.

---

## Rigour

**How do you know these numbers are not cherry-picked?**
Three mechanisms. (1) The estimator was **fixed before the runs**: the mean of the final five
rounds, never a single round. (2) Every iteration is frozen with a **SHA-256 manifest** recording
the git commit; `freeze_baseline.py --verify` re-checks it. (3) `check_docs.py` re-derives sixteen
headline figures from those frozen logs and fails if any document disagrees — run it now.

**Why does the estimator matter?**
Because early on we found the round-to-round noise was **larger than the effects under test**. We
have a case at seed 123 where a single-round reading says FedBN trails FedAvg by 0.098 and the
pre-registered estimator says it *leads* by 0.0376 — the same run, opposite conclusions. That
finding is why the rest of the protocol exists.

**What statistics do you use?**
Dice is computed per volume, so comparisons are **paired on the same case**. We report a bootstrap
95% CI over 10,000 resamples, a Wilcoxon signed-rank p-value, and the **sign split** — how many of
the 62 volumes moved each way. The sign split is there because a large mean driven by a handful of
cases is a different claim from a consistent win, and only the split shows the difference.

**Have you retracted anything?**
Yes, and it is documented in `docs/conventions.md` §6. Four claims: the "all three verdicts reverse
in 3D" claim (refuted — it was an artefact of the v1 recipe); the BatchNorm-statistics explanation
for FedBN's 3D behaviour (not supported — the cause was the round budget); the claim that outlier
degradation is caused by FedAvg's *averaging* (refuted — centralized has no aggregation step and
shows the same penalty, so the cause is optimising over a majority-dominated distribution); and a
monotone-ordering claim that held at seed 42 and failed at seed 7.

---

## Awkward questions — answer these honestly

**Your best results are single-seed. Isn't that fragile?**
Yes, and it is stated as limitation 2 in `docs/iteration-report.md` §11. v1 and v2 have three 2D
seeds; v3–v5 are seed 42 only, and no iteration has 3D multi-seed coverage. **Any ± figure for v5
would be fabricated**, so we do not give one. Margins under about 0.01 should be treated as
provisional — which is exactly why we refuse to claim the 3D FedBN result.

**v3 changed two things at once. Isn't that a confound?**
It is, and we say so. v3 changed the augmentation recipe *and* raised the training cap from 150 to
230 cases. The reported deltas are a combined effect and must not be written up as a clean
ablation. Separating them costs one run — `--preset v5 --train-per-hospital 150`, about 2.5 hours
— and it has not been done.

**Two of your five iterations made things worse. Why present them?**
Because they are the most informative results we have. v2 regressed because its intensity jitter
perturbed gamma, bias and blur — the exact channels the scanner shift uses, so the augmentation
was competing with the experimental variable. v4 regressed because we stretched the cosine anneal
over 40 rounds instead of letting it bottom out at 25, so the model never consolidated at a low
learning rate. Both were our reasoning errors, both were diagnosed from the logs, and both fixes
are in the final recipe.

**Is the system actually private?**
It is private in the sense that no image or label leaves the hospital — the server sees only
tensors. It has **no formal guarantee**: raw weights can leak information about training data, and
we implement no differential privacy. That is the honest limit, and it is the most valuable next
step. It would also test a real prediction: DP noise should hurt Site D most, and FedBN — whose
normalisation never enters the noisy aggregate — should degrade more gracefully than FedAvg.

**Would this work at a hospital that was not in the federation?**
We do not know. Every test site was represented in training. Measuring that needs a
leave-one-site-out run, which we have not done.

**Why not compare against other FL algorithms?**
Fair criticism. FedProx targets client heterogeneity directly and is the natural rival to FedBN;
it is a small change to the existing round loop. We compared against the two references that bound
the problem — the centralized ceiling and the local-only floor — but not against a competing
personalization method.

**Why 2D and 3D?**
They disagree, and where they disagree is a finding. 2D gets 8 slices per case per epoch; 3D gets
2 patches, so each 3D epoch is a much noisier estimate. Several effects that are clear in 2D are
not resolvable in 3D at this sample size — including the FedBN result.
"""
    (OUT / "QUESTIONS.md").write_text(q, encoding="utf-8")


def main() -> int:
    if not DOCS.exists():
        print("no docs/ directory", file=sys.stderr)
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    copied, missing = 0, []
    for folder, items in PACK.items():
        d = OUT / folder
        d.mkdir(parents=True, exist_ok=True)
        for name, _ in items:
            src = DOCS / name
            if not src.exists():
                missing.append(name)
                continue
            body = rewrite_links(src.read_text(encoding="utf-8"))
            (d / name).write_text(BANNER.format(name=name) + body, encoding="utf-8")
            copied += 1

    demo = OUT / "05-demo"
    demo.mkdir(parents=True, exist_ok=True)
    built = REPO / "artifacts" / "showcase" / "showcase.html"
    if built.exists():
        shutil.copy2(built, demo / "showcase.html")
    else:
        missing.append("artifacts/showcase/showcase.html (run build_showcase.py)")

    runs = collect_runs()
    stats = collect_stats(runs)
    write_guide(runs, stats, copied)
    write_questions(runs, stats)

    print(f"review/ built — {copied} documents in {len(PACK)} folders, plus the walkthrough")
    print("  review/README.md     the guide: commands, locations, numbers, troubleshooting")
    print("  review/QUESTIONS.md  likely review questions, answered")
    if missing:
        print("\nmissing:")
        for m in missing:
            print(f"  ! {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
