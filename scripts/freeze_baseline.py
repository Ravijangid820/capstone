"""Freeze the current runs into an immutable, hashed snapshot -- the "before" of any comparison.

    python scripts/freeze_baseline.py                 # snapshot artifacts/runs/ -> artifacts/baseline/
    python scripts/freeze_baseline.py --verify        # re-hash and report drift; changes nothing
    python scripts/freeze_baseline.py --force         # re-freeze over an existing snapshot

Why this exists. `MetricsWriter` appends and `run_id` is `<method>_<dim>_<seed>`, so re-running a
method with the same seed writes *into the same file* as the run before it -- 25 fresh rounds land
underneath 25 old ones and `analyze.py`, which scores `max(round)`, then averages the two silently.
The baseline has to leave the writable tree before any retraining starts, or there is nothing left
to compare against.

The manifest records a SHA-256 per file plus the git commit, so "these are the numbers we had"
is checkable months later rather than asserted. `--verify` is what makes it a claim and not a copy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedbrats.config import REGIONS, Config  # noqa: E402

COPY = ("metrics.jsonl", "config.json", "run.log", "per_case.jsonl")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path) -> str | None:
    """The commit the snapshot was taken at -- None outside a repo, or if git is unavailable."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def summarize(metrics: Path) -> dict:
    """Headline numbers for one run, so the manifest is readable without a tool.

    Reports the final round *and* the mean over the last five, because the two disagree: the
    curves plateau by ~round 15 and then oscillate by more than the effects under test, so a
    single-round figure is a draw from that oscillation rather than a measurement of it.
    """
    rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    diag = [r for r in rows
            if r["split"] == "test"
            and (r["model_hospital"] == "global" or r["model_hospital"] == r["test_hospital"])]
    if not diag:
        return {"rows": len(rows), "rounds": 0}

    last = max(r["round"] for r in diag)
    tail = [r for r in range(max(1, last - 4), last + 1)]

    def mean_at(rnd: int, region: str) -> float:
        vals = [r[f"dice_{region}"] for r in diag if r["round"] == rnd]
        return statistics.fmean(vals) if vals else float("nan")

    def outlier_at(rnd: int, region: str) -> float:
        vals = [r[f"dice_{region}"] for r in diag
                if r["round"] == rnd and r["test_hospital"] == "H4"]
        return statistics.fmean(vals) if vals else float("nan")

    return {
        "rows": len(rows),
        "rounds": last,
        "final_round": {r: round(mean_at(last, r), 6) for r in REGIONS},
        "last5_mean": {r: round(statistics.fmean(mean_at(x, r) for x in tail), 6) for r in REGIONS},
        "H4_final_round": {r: round(outlier_at(last, r), 6) for r in REGIONS},
        "H4_last5_mean": {r: round(statistics.fmean(outlier_at(x, r) for x in tail), 6)
                          for r in REGIONS},
    }


def collect(runs_dir: Path) -> list[Path]:
    return sorted(p.parent for p in runs_dir.glob("*/metrics.jsonl"))


def freeze(runs_dir: Path, dest: Path, repo: Path, force: bool) -> int:
    runs = collect(runs_dir)
    if not runs:
        print(f"no runs with metrics.jsonl under {runs_dir}", file=sys.stderr)
        return 1
    if dest.exists() and not force:
        print(f"{dest} already exists -- refusing to overwrite a frozen baseline.\n"
              f"Pass --force if you really mean to replace it, or --verify to check it.",
              file=sys.stderr)
        return 1

    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    entries: dict[str, dict] = {}
    for run in runs:
        out = dest / run.name
        out.mkdir()
        files = {}
        for name in COPY:
            src = run / name
            if not src.exists():
                continue
            shutil.copy2(src, out / name)
            files[name] = {"sha256": sha256(src), "bytes": src.stat().st_size}
        entries[run.name] = {"files": files, "summary": summarize(run / "metrics.jsonl")}
        print(f"  froze {run.name:<22} {', '.join(files)}")

    manifest = {
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(repo),
        "source": str(runs_dir),
        "n_runs": len(entries),
        "runs": entries,
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nfroze {len(entries)} runs -> {dest}")
    print(f"manifest: {dest / 'MANIFEST.json'}")
    return 0


def verify(dest: Path) -> int:
    mf = dest / "MANIFEST.json"
    if not mf.exists():
        print(f"no manifest at {mf} -- nothing frozen yet", file=sys.stderr)
        return 1
    manifest = json.loads(mf.read_text())

    bad = 0
    for run, entry in sorted(manifest["runs"].items()):
        for name, rec in entry["files"].items():
            path = dest / run / name
            if not path.exists():
                print(f"  MISSING  {run}/{name}")
                bad += 1
            elif sha256(path) != rec["sha256"]:
                print(f"  MODIFIED {run}/{name}")
                bad += 1
    total = sum(len(e["files"]) for e in manifest["runs"].values())
    print(f"\nverified {total - bad}/{total} files in {manifest['n_runs']} runs "
          f"(frozen {manifest['frozen_at']}, commit {(manifest['git_commit'] or '?')[:8]})")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs-dir", type=str, default=None, help="source runs (default artifacts/runs)")
    ap.add_argument("--dest", type=str, default=None, help="snapshot dir (default artifacts/baseline)")
    ap.add_argument("--verify", action="store_true", help="re-hash an existing snapshot; no writes")
    ap.add_argument("--force", action="store_true", help="overwrite an existing snapshot")
    args = ap.parse_args()

    cfg = Config()
    repo = Path(__file__).resolve().parents[1]
    runs_dir = Path(args.runs_dir) if args.runs_dir else cfg.paths.runs
    dest = Path(args.dest) if args.dest else cfg.paths.artifacts / "baseline"

    return verify(dest) if args.verify else freeze(runs_dir, dest, repo, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
