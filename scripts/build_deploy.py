"""Assemble `deploy/` -- a self-contained copy of just what has to run in production.

    python scripts/build_deploy.py                    # 12 cases, both backbones
    python scripts/build_deploy.py --cases 4 --dims 2d
    python scripts/build_deploy.py --all-cases        # every test case (~2.4 GB)

Two targets, because they can carry very different things:

  deploy/static/   the walkthrough alone. One HTML file, no server, no data, no Python.
                   Goes on Vercel, Cloudflare Pages, GitHub Pages -- anywhere.

  deploy/app/      the live inference server. A *pruned mirror* of this repository, so the
                   paths in config.py resolve exactly as they do here and not one line of
                   code has to change: src/, scripts/demo_server.py, the checkpoints, the
                   split manifest, a trimmed cache index, and a curated handful of cases.

**Vercel cannot host `app/`,** and the README written into deploy/ says so plainly: a serverless
function is capped well below the size of a CPU torch install, and a request peaks at ~1.2 GB
resident, above the Hobby memory limit. `app/` wants a container or a VM -- an LXC on Proxmox is
an excellent fit.

**Why a subset of cases.** The demo reads raw NIfTI per request. All 248 test cases are ~2.4 GB,
and serving them publicly is closer to redistributing BraTS than to demonstrating a model. A
dozen cases is ~115 MB, shows every site, and keeps `/api/cases` honest about what exists --
the endpoint lists exactly what was bundled, because it reads the trimmed index this writes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "deploy"

sys.path.insert(0, str(REPO / "src"))
from fedbrats.config import Config          # noqa: E402
from fedbrats.data import cache_key         # noqa: E402

METHODS = ["centralized", "local", "fedavg", "fedbn"]
SITES = {"H1": "Site A", "H2": "Site B", "H3": "Site C", "H4": "Site D"}

# The static showcase can be hosted anywhere as index.html, but this makes it equally easy to run
# inside a small LXC: `npm install --omit=dev && npm start` serves it on localhost:8080.
STATIC_PACKAGE_JSON = json.dumps({
    "name": "fedbrats-showcase",
    "version": "1.0.0",
    "private": True,
    "scripts": {"start": "serve -s . -l 8080"},
    "dependencies": {"serve": "14.2.4"},
}, indent=2) + "\n"

# The cases the paper figures use. Whatever else is bundled, these two must be, so the
# qualitative figure and the live demo can be shown on the same patient.
PINNED = ["BraTS2021_01163", "BraTS2021_00104"]


def pick_cases(n: int, all_cases: bool) -> list[str]:
    """A spread across all four sites, test split only, figure cases first."""
    manifest = json.loads((REPO / "artifacts" / "splits" / "partition.json").read_text())
    assign = manifest["assignment"]
    test = {h: sorted(c for c, v in assign.items()
                      if v["split"] == "test" and v["hospital"] == h) for h in SITES}
    if all_cases:
        return sorted(c for c, v in assign.items() if v["split"] == "test")

    chosen: list[str] = [c for c in PINNED if c in assign and assign[c]["split"] == "test"]
    per_site = max(1, n // len(SITES))
    for h in SITES:
        for c in test[h]:
            if len([x for x in chosen if assign[x]["hospital"] == h]) >= per_site:
                break
            if c not in chosen:
                chosen.append(c)
    return sorted(chosen)


REQUIREMENTS = """\
# Runtime only -- the training stack (matplotlib, tensorboard, pandas, scikit-*, tqdm,
# torchvision) is not imported by the server and is deliberately absent.
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.2
monai>=1.3
nibabel>=5.2
numpy>=1.26
scipy>=1.12
pillow>=10.0
einops>=0.7
"""

DOCKERFILE = """\
# CPU-only. The models are 1.6M parameters; a GPU buys nothing here and costs a great deal.
FROM python:3.12-slim

# nibabel reads compressed NIfTI; nothing else needs system libraries.
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pin the data root rather than leaning on config.py's probe order -- the probe would resolve
# correctly here anyway, but a deployment should not depend on which paths happen not to exist.
ENV FEDBRATS_DATA_ROOT=/app/data/BraTS2021_Training_Data

# A full-volume request peaks around 1.2 GB resident. Give the container 2 GB.
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
CMD ["python", "scripts/demo_server.py", "--host", "0.0.0.0", "--port", "8000"]
"""

DOCKERIGNORE = """\
__pycache__/
*.pyc
.venv/
"""

RUNSH = """\
#!/usr/bin/env bash
# Bare LXC / VM path -- no Docker. Run once to set up, then again to start.
set -euo pipefail
cd "$(dirname "$0")"

# Pin the data root rather than leaning on config.py's probe order.
export FEDBRATS_DATA_ROOT="$PWD/data/BraTS2021_Training_Data"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

exec ./.venv/bin/python scripts/demo_server.py --host 0.0.0.0 --port "${PORT:-8000}"
"""

VERCEL_JSON = """\
{
  "cleanUrls": true,
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "X-Content-Type-Options", "value": "nosniff" },
        { "key": "Referrer-Policy", "value": "no-referrer" }
      ]
    }
  ]
}
"""


def human(n: int) -> str:
    return f"{n / 1048576:.1f} MB" if n >= 1048576 else f"{n / 1024:.0f} KB"


def tree_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def write_readme(app: Path, cases: list[str], dims: list[str], sizes: dict) -> None:
    case_rows = "\n".join(
        f"| `{c}` | {SITES[h]} |" for c, h in sorted(
            (c, json.loads((REPO / 'artifacts' / 'splits' / 'partition.json').read_text())
             ["assignment"][c]["hospital"]) for c in cases))
    (OUT / "README.md").write_text(f"""\
# Deploy

Two independent bundles. Build both with `python scripts/build_deploy.py` from the repository root.

| | What it is | Where it runs | Size |
|---|---|---|---|
| [`static/`](static/) | The walkthrough — one HTML file | Vercel · Cloudflare Pages · GitHub Pages · any static host | {human(sizes['static'])} |
| [`app/`](app/) | The live inference server | A container or VM — **not** Vercel | {human(sizes['app'])} |

---

## static/ — the walkthrough

One self-contained file. No server, no Python, no data, no network calls beyond a webfont.

**Small LXC / Cloudflare Tunnel**

```bash
cd deploy/static
npm install --omit=dev
npm start                    # serves http://127.0.0.1:8080
```

Point the Cloudflare Tunnel public hostname at `http://localhost:8080`. No Python application
server, data, checkpoints, or GPU is involved.

**Vercel**

```bash
cd deploy/static && npx vercel deploy --prod
```

**Cloudflare Pages**

```bash
cd deploy/static && npx wrangler pages deploy . --project-name fedbrats
```

**GitHub Pages** — commit `static/` to a `gh-pages` branch, or point Pages at the folder.

Nothing here needs configuration, and there is no attack surface: it is a document.

---

## app/ — the live demo

### Why not Vercel

Three independent blockers, so this is not a matter of trying harder:

- A serverless function bundle is capped far below the size of a CPU PyTorch install.
- One full-volume request peaks at about **1.2 GB resident**; the Hobby tier allows 1 GB.
- Importing torch costs ~590 MB and several seconds before any work starts — on a platform
  billed per invocation with cold starts, that is the wrong shape entirely.

Use a container or a VM. An LXC on Proxmox is a very good fit.

### Proxmox LXC (no Docker)

```bash
# in the container, as root
apt update && apt install -y python3 python3-venv
adduser --system --group fedbrats
# copy this folder to /opt/fedbrats, then
chown -R fedbrats:fedbrats /opt/fedbrats
sudo -u fedbrats /opt/fedbrats/run.sh
```

`run.sh` creates a virtualenv on first run and starts the server on `0.0.0.0:8000`
(`PORT=8080 ./run.sh` to change it).

**Container sizing:** 2 GB RAM, 2 vCPU, 4 GB disk. A request takes **2.5 s** (2D) or **1.6 s**
(3D) on CPU. Memory is the binding constraint, not CPU — 1 GB will be killed mid-request.

As a systemd unit:

```ini
[Unit]
Description=FedBraTS demo
After=network.target

[Service]
User=fedbrats
WorkingDirectory=/opt/fedbrats
ExecStart=/opt/fedbrats/run.sh
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### Docker

```bash
cd deploy/app
docker build -t fedbrats .
docker run -p 8000:8000 --memory 2g fedbrats
```

### Then

| URL | What |
|---|---|
| `/` | live inference — pick a case, a model, a hospital |
| `/showcase` | the walkthrough |
| `/api/health` | readiness, device, and which checkpoint set is loaded |

---

## Before you expose it publicly

**There is no authentication.** Anyone with the URL can POST to `/api/predict`, and each request
is ~2.5 s of CPU. Put something in front of it:

- **Cloudflare Tunnel + Access** — free, gives you a hostname and an email gate in one step:
  ```bash
  cloudflared tunnel --url http://localhost:8000
  ```
- Or a reverse proxy with rate limiting on `/api/*`.

**On the data.** BraTS 2021 carries data-use terms from the challenge. This bundle ships
**{len(cases)} cases** rather than all 248 partly for size and partly so that a public endpoint is
demonstrating a model rather than redistributing a dataset. Check the terms before increasing it.

---

## What is in `app/`

```
app/
  scripts/demo_server.py          the server
  src/fedbrats/                   the library it imports
  src/fedbrats/static/            the demo front end (vanilla JS + Three.js + wasm)
  artifacts/
    runs/v5/*/checkpoints/        the {len(dims) * 4} v5 checkpoints ({human(sizes['ckpt'])})
    splits/partition.json         the committed split
    cache/*/index.json            trimmed to the bundled cases -- this is what /api/cases lists
    showcase/showcase.html        served at /showcase
  data/BraTS2021_Training_Data/   the {len(cases)} bundled cases ({human(sizes['data'])})
  requirements.txt · Dockerfile · run.sh
```

The layout mirrors the repository exactly, so `config.py` resolves every path without a single
environment variable and no code differs from what was tested.

Backbones bundled: **{', '.join(d.upper() for d in dims)}**.

### Bundled cases

| Case | Site |
|---|---|
{case_rows}

---

Regenerate with `python scripts/build_deploy.py`. Nothing in here should be edited by hand.
""", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", type=int, default=12, help="how many cases to bundle (default 12)")
    ap.add_argument("--all-cases", action="store_true", help="bundle every test case (~2.4 GB)")
    ap.add_argument("--dims", nargs="+", default=["2d", "3d"], choices=["2d", "3d"])
    ap.add_argument("--runs", default="artifacts/runs/v5", help="checkpoints to ship")
    ap.add_argument("--source", default=None,
                    help="case source (default: the compressed data/BraTS2021_Training_Data)")
    args = ap.parse_args()

    showcase = REPO / "artifacts" / "showcase" / "showcase.html"
    if not showcase.exists():
        print("build the walkthrough first: python scripts/build_showcase.py", file=sys.stderr)
        return 1

    if OUT.exists():
        shutil.rmtree(OUT)

    # ---- static target -----------------------------------------------------------------
    static = OUT / "static"
    static.mkdir(parents=True)
    shutil.copy2(showcase, static / "index.html")
    (static / "vercel.json").write_text(VERCEL_JSON, encoding="utf-8")
    (static / "package.json").write_text(STATIC_PACKAGE_JSON, encoding="utf-8")

    # ---- app target: a pruned mirror of the repository ---------------------------------
    app = OUT / "app"
    (app / "scripts").mkdir(parents=True)
    shutil.copy2(REPO / "scripts" / "demo_server.py", app / "scripts" / "demo_server.py")

    shutil.copytree(REPO / "src" / "fedbrats", app / "src" / "fedbrats",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    art = app / "artifacts"
    (art / "splits").mkdir(parents=True)
    shutil.copy2(REPO / "artifacts" / "splits" / "partition.json", art / "splits" / "partition.json")
    (art / "showcase").mkdir(parents=True)
    shutil.copy2(showcase, art / "showcase" / "showcase.html")

    # checkpoints
    runs_src = REPO / args.runs
    ckpt_root = art / "runs" / "v5"
    n_ckpt = 0
    for dim in args.dims:
        for m in METHODS:
            src = runs_src / f"{m}_{dim}_42" / "checkpoints" / "final.pt"
            if not src.exists():
                print(f"  ! missing checkpoint {src.relative_to(REPO)}", file=sys.stderr)
                continue
            dst = ckpt_root / f"{m}_{dim}_42" / "checkpoints" / "final.pt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_ckpt += 1

    # cases + a cache index trimmed to exactly them
    cases = pick_cases(args.cases, args.all_cases)
    # Prefer the COMPRESSED source. `load_case` reads .nii and .nii.gz alike, and the unzipped
    # copy exists only to make cache builds faster -- shipping it would be ~90 MB per case
    # instead of ~10, for identical behaviour.
    compressed = REPO / "data" / "BraTS2021_Training_Data"
    data_root = Path(args.source) if args.source else (
        compressed if compressed.exists() else Config().paths.data_root)
    dest_data = app / "data" / "BraTS2021_Training_Data"
    dest_data.mkdir(parents=True)
    shipped = []
    for c in cases:
        src = data_root / c
        if not src.exists():
            print(f"  ! case not found: {c}", file=sys.stderr)
            continue
        shutil.copytree(src, dest_data / c)
        shipped.append(c)

    key = cache_key(Config())
    idx_src = REPO / "artifacts" / "cache" / key / "index.json"
    idx_dst = art / "cache" / key
    idx_dst.mkdir(parents=True)
    full = json.loads(idx_src.read_text()) if idx_src.exists() else {}
    # /api/cases returns this verbatim, so trimming it is also what stops the deployed demo
    # advertising 248 cases it does not carry.
    (idx_dst / "index.json").write_text(
        json.dumps({c: full[c] for c in shipped if c in full}, indent=1), encoding="utf-8")

    # The front end already came along inside src/fedbrats/static, which is exactly where
    # demo_server.py's STATIC_DIR points -- a second copy would be the same bytes twice.

    (app / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (app / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    (app / ".dockerignore").write_text(DOCKERIGNORE, encoding="utf-8")
    run = app / "run.sh"
    run.write_text(RUNSH, encoding="utf-8", newline="\n")
    run.chmod(0o755)

    sizes = {"static": tree_size(static), "app": tree_size(app),
             "ckpt": tree_size(art / "runs"), "data": tree_size(app / "data")}
    write_readme(app, shipped, args.dims, sizes)

    print(f"deploy/ built")
    print(f"  static/  {human(sizes['static']):>10}   the walkthrough, for Vercel / Pages")
    print(f"  app/     {human(sizes['app']):>10}   the server, for a container or VM")
    print(f"             checkpoints {human(sizes['ckpt']):>9}  ({n_ckpt} models, {'+'.join(args.dims)})")
    print(f"             cases       {human(sizes['data']):>9}  ({len(shipped)} bundled)")
    print(f"  read deploy/README.md — it explains why app/ cannot go on Vercel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
