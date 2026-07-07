"""Step 1 — make the data usable.

Decompress every BraTS `.nii.gz` volume to an uncompressed `.nii`, once, so all
downstream reads are fast random-access (no repeated gzip). The original compressed
data is left untouched; output mirrors the source layout on the D: drive
(C: is nearly full; the ~112 GB uncompressed copy fits comfortably on D:).

Idempotent (skips files already done) and atomic (writes a .tmp then renames, so an
interrupted run never leaves a half-file that looks complete).

    .venv/bin/python unzip_data.py
"""

import glob
import gzip
import os
import shutil
import time
from multiprocessing import Pool

SRC = "data/BraTS2021_Training_Data"
DST = "/mnt/d/capstone_data/unzipped"
WORKERS = 12


def unzip_one(gz_path: str) -> int:
    rel = os.path.relpath(gz_path, SRC)          # BraTS2021_XXXXX/<file>.nii.gz
    out = os.path.join(DST, rel[:-3])            # -> <DST>/BraTS2021_XXXXX/<file>.nii
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return 0                                  # already decompressed
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with gzip.open(gz_path, "rb") as f_in, open(tmp, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=4 * 1024 * 1024)
    os.replace(tmp, out)                          # atomic
    return 1


def main() -> None:
    files = sorted(glob.glob(os.path.join(SRC, "*", "*.nii.gz")))
    print(f"{len(files)} .nii.gz files -> {DST}/  ({WORKERS} workers)", flush=True)
    t = time.time()
    done = 0
    with Pool(WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(unzip_one, files, chunksize=8), 1):
            done += r
            if i % 500 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    skipped = len(files) - done
    print(f"DONE: decompressed {done}, already-present {skipped}, in {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
