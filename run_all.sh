#!/usr/bin/env bash
# Full experiment pipeline for WSL2. Run from the project root:
#     bash run_all.sh
#
# Override via env vars, e.g.:
#     SPLIT="--fets-csv data/fets_partitioning.csv" ROUNDS=80 bash run_all.sh
#     SMOKE=1 bash run_all.sh      # tiny CPU dry-run to check everything wires up
#     MAX_CASES=200 ROUNDS=20 EPOCHS=20 bash run_all.sh   # scoped run: hours, not days
#
# If you see "\r: command not found", the file has CRLF line endings — fix with:
#     sed -i 's/\r$//' run_all.sh
set -euo pipefail

DR="${DR:-data/BraTS2021_Training_Data}"
SPLIT="${SPLIT:---n-clients 6}"          # or: --fets-csv data/fets_partitioning.csv
ROUNDS="${ROUNDS:-50}"
EPOCHS="${EPOCHS:-40}"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-8}"                   # DataLoader workers; at 2 the GPU was data-starved (16 cores here)
BATCH="${BATCH:-8}"                       # batch size; 4GB VRAM fits ~32 for this 2D UNet (raises GPU util)

if [ "${SMOKE:-0}" = "1" ]; then          # fast dry-run: tiny data, CPU, 1 round
  SPLIT="--n-clients 2"; ROUNDS=1; EPOCHS=1; GPU=""; EXTRA="--max-cases 6"; WORKERS=0
  GPUFLAG=""
else
  # MAX_CASES caps the case count for a scoped run (hours instead of days at full
  # scale). Leave unset to use ALL cases. Applied to every step.
  EXTRA="${MAX_CASES:+--max-cases $MAX_CASES}"
  GPUFLAG="--gpu $GPU"
fi

# Synthetic per-hospital scanner shift = our non-IID heterogeneity when there's no
# FeTS CSV. Set SHIFT="" to disable. Real FeTS data is already non-IID, so never
# stack a synthetic shift on top of it.
SHIFT="${SHIFT:---synthetic-shift}"
case "$SPLIT" in *fets*) SHIFT="";; esac

echo "### 1/8 centralized ceiling"
uv run python -m braintumor_fl.train_centralized --data-root "$DR" $SPLIT --epochs "$EPOCHS" --workers "$WORKERS" --batch-size "$BATCH" $EXTRA $SHIFT --out data/centralized_unet.pt
echo "### 2/8 centralized per-hospital eval"
uv run python fl/evaluate.py --data-root "$DR" $SPLIT --model data/centralized_unet.pt --method centralized --workers "$WORKERS" --batch-size "$BATCH" $EXTRA $SHIFT
echo "### 3/8 local-only floor"
uv run python fl/run_local_baselines.py --data-root "$DR" $SPLIT --epochs "$EPOCHS" --workers "$WORKERS" --batch-size "$BATCH" $EXTRA $SHIFT
echo "### 4/8 FedAvg"
uv run python fl/run_fedavg.py --data-root "$DR" $SPLIT --rounds "$ROUNDS" --epochs 1 --method fedavg --workers "$WORKERS" --batch-size "$BATCH" $GPUFLAG $EXTRA $SHIFT
echo "### 5/8 FedProx"
uv run python fl/run_fedavg.py --data-root "$DR" $SPLIT --rounds "$ROUNDS" --epochs 1 --method fedprox --prox-mu 0.01 --workers "$WORKERS" --batch-size "$BATCH" $GPUFLAG $EXTRA $SHIFT
echo "### 6/8 FedBN (primary personalization)"
uv run python fl/run_fedavg.py --data-root "$DR" $SPLIT --rounds "$ROUNDS" --epochs 1 --method fedbn --personalization fedbn --workers "$WORKERS" --batch-size "$BATCH" $GPUFLAG $EXTRA $SHIFT
echo "### 7/8 fine-tune personalization"
uv run python fl/finetune.py --data-root "$DR" $SPLIT --ft-epochs 5 --workers "$WORKERS" --batch-size "$BATCH" $EXTRA $SHIFT
echo "### 8/8 analyze -> results/"
uv run python analyze.py
echo "DONE. See results/per_hospital.csv, results/summary.csv, results/personalization_gain.png"
