# Execution environments

Three places this project runs. The **science code is identical everywhere** — only orchestration
and paths differ.

| | **Windows-native** | **WSL2** | **Colab T4** |
|---|---|---|---|
| Custom FL loop (E0–E3) | ✅ | ✅ | ✅ |
| **NVIDIA FLARE port** | ❌ *never* | ✅ | ✅ |
| GPU | RTX 3050, 4 GB | RTX 3050 (passthrough) | T4, 16 GB |
| Data access | `D:\` native NTFS — fast | `/mnt/d` via drvfs — slow per-op | Drive → `/content` |
| Multiprocessing | `spawn` | `fork` | `fork` |
| Role | quick 2D checks, cache builds | dev + FLARE smoke | **all real training** |

## Why FLARE cannot run on native Windows

NVIDIA FLARE imports the POSIX-only stdlib module `resource`
(`nvflare/fuel/f3/cellnet/net_agent.py`). On Windows the package *installs* fine and then fails at
import with `ModuleNotFoundError: No module named 'resource'`. The maintainers state plainly:
*"We only officially support Unix/Linux currently."*

Consequences, baked into the code:

- `nvflare` is an **optional extra with a platform marker** (`pyproject.toml`), so `uv sync` on
  Windows never installs it: `uv sync --extra flare` on Linux only.
- **No shared module imports `nvflare` at top level.** The Phase-2 FLARE port lives in its own
  module, imported only by its own entrypoint.
- Windows is therefore a **custom-loop-only** environment. A FLARE result can never be the sole
  way to reproduce a finding.

## The Windows compatibility contract

Both OSes stay green only if these hold. They are already satisfied — keep them that way.

1. **`if __name__ == "__main__":` in every `scripts/` entrypoint.** Windows `spawn` re-imports the
   module in each worker process; without the guard, a parallel cache build fork-bombs.
2. **Parallel workers are top-level functions taking picklable tuples** — never closures or
   lambdas. `fork` tolerates closures; `spawn` cannot pickle them. See `data.build_case_cache`.
3. **No hardcoded paths.** `Paths` resolves per-platform and honours `FEDBRATS_DATA_ROOT` /
   `FEDBRATS_CACHE_DIR`.
4. **Close memmaps before rebuilding a cache.** Windows locks open files; a dangling `np.memmap`
   makes a cache rebuild raise `PermissionError` on Windows while succeeding on Linux.
5. **`num_workers` defaults to 0 on Windows** (`config._default_workers`) — per-epoch `spawn`
   overhead outweighs the parallelism for our batch sizes.

**Free cross-platform canary:** `artifacts/splits/partition.json` must have an identical md5 on both
OSes. It is built from sorted IDs and seeded `random` only, so any divergence means
non-determinism crept in.

## Disk

`/mnt/d` (Windows `D:`) is the only drive with real headroom. Note that under WSL2 the repo itself
lives inside a virtual disk backed by `C:`, which is nearly full — so **the full cache must never
land in `artifacts/` on WSL**.

| cache | size | where |
|---|---|---|
| smoke (≤3 cases/hospital) | ~1 GB | `artifacts/cache` (default) — fine anywhere |
| full (1251 cases) | **~44 GB** | `FEDBRATS_CACHE_DIR=/mnt/d/data/cache` |
| Colab | ~44 GB | `/content/cache` (auto-detected; ephemeral local SSD) |

Measured: 24 cases → 837 MB, i.e. **~35 MB/case** (fp16 volume + uint8 masks).

## Recipes

```bash
# WSL2 / Linux
uv sync                                  # custom loop only
uv sync --extra flare                    # + NVIDIA FLARE (Phase 2)
uv run python scripts/build_partition.py
uv run python scripts/build_cache.py --workers 8
uv run python scripts/run_experiment.py --method fedbn --dim 2d
```

```powershell
# Windows (PowerShell). The Linux .venv cannot be reused — rebuild it.
uv sync                                  # nvflare is skipped by its platform marker
uv run python scripts/build_cache.py --workers 8 --data-root D:/data/unzipped
uv run python scripts/run_experiment.py --method fedbn --dim 2d
```

```python
# Colab: cache dir auto-resolves to /content/cache
!git clone <repo> && cd capstone && pip install -e .
!python scripts/build_cache.py --workers 4 --data-root /content/drive/MyDrive/capstone/unzipped
!python scripts/run_experiment.py --method fedbn --dim 2d
```

Smoke run (seconds, any OS) — proves the wiring before you spend a Colab session:

```bash
python scripts/build_cache.py --max-cases 3 --workers 4
python scripts/run_experiment.py --method fedbn --rounds 2 --max-train-cases 3 --max-test-cases 2
```
