#!/usr/bin/env bash
set -euo pipefail
cd /workspace/salve
export PYTHONPATH=/workspace/salve PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir -p outputs/em_advice_1000_noheldout
trap 'code=$?; printf "%s\n" "$code" > outputs/em_advice_1000_noheldout/sweep.exit' EXIT
# Keep a process limit; the local monitor also checks the available credit.
timeout 50400 .venv/bin/python experiment_scripts/em_advice/run.py \
  --out outputs/em_advice_1000_noheldout --domains finance medical sports \
  --train-samples 1000 --search-samples 100 --skip-held-out --bounded-search \
  --train-mini-batch-size 4 --score-mini-batch-size 8
