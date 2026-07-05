#!/usr/bin/env bash
# Full experiment pipeline for WSL2. Run from the project root:
#     bash run_all.sh
#
# Override via env vars, e.g.:
#     SPLIT="--fets-csv data/fets_partitioning.csv" ROUNDS=80 bash run_all.sh
#     SMOKE=1 bash run_all.sh      # tiny CPU dry-run to check everything wires up
#
# If you see "\r: command not found", the file has CRLF line endings — fix with:
#     sed -i 's/\r$//' run_all.sh
set -euo pipefail

DR="${DR:-data/BraTS2021_Training_Data}"
SPLIT="${SPLIT:---n-clients 6}"          # or: --fets-csv data/fets_partitioning.csv
ROUNDS="${ROUNDS:-50}"
EPOCHS="${EPOCHS:-40}"
GPU="${GPU:-0}"

if [ "${SMOKE:-0}" = "1" ]; then          # fast dry-run: tiny data, CPU, 1 round
  SPLIT="--n-clients 2"; ROUNDS=1; EPOCHS=1; GPU=""; EXTRA="--max-cases 6"
  GPUFLAG=""
else
  EXTRA=""
  GPUFLAG="--gpu $GPU"
fi

# Synthetic per-hospital scanner shift = our non-IID heterogeneity when there's no
# FeTS CSV. Set SHIFT="" to disable. Real FeTS data is already non-IID, so never
# stack a synthetic shift on top of it.
SHIFT="${SHIFT:---synthetic-shift}"
case "$SPLIT" in *fets*) SHIFT="";; esac

echo "### 1/8 centralized ceiling"
uv run python -m braintumor_fl.train_centralized --data-root "$DR" $SPLIT --epochs "$EPOCHS" $EXTRA $SHIFT --out data/centralized_unet.pt
echo "### 2/8 centralized per-hospital eval"
uv run python fl/evaluate.py --data-root "$DR" $SPLIT --model data/centralized_unet.pt --method centralized $EXTRA $SHIFT
echo "### 3/8 local-only floor"
uv run python fl/run_local_baselines.py --data-root "$DR" $SPLIT --epochs "$EPOCHS" $EXTRA $SHIFT
echo "### 4/8 FedAvg"
uv run python fl/run_fedavg.py --data-root "$DR" $SPLIT --rounds "$ROUNDS" --epochs 1 --method fedavg $GPUFLAG $EXTRA $SHIFT
echo "### 5/8 FedProx"
uv run python fl/run_fedavg.py --data-root "$DR" $SPLIT --rounds "$ROUNDS" --epochs 1 --method fedprox --prox-mu 0.01 $GPUFLAG $EXTRA $SHIFT
echo "### 6/8 FedBN (primary personalization)"
uv run python fl/run_fedavg.py --data-root "$DR" $SPLIT --rounds "$ROUNDS" --epochs 1 --method fedbn --personalization fedbn $GPUFLAG $EXTRA $SHIFT
echo "### 7/8 fine-tune personalization"
uv run python fl/finetune.py --data-root "$DR" $SPLIT --ft-epochs 5 $EXTRA $SHIFT
echo "### 8/8 analyze -> results/"
uv run python analyze.py
echo "DONE. See results/per_hospital.csv, results/summary.csv, results/personalization_gain.png"
