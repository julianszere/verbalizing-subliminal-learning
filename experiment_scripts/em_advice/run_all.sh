#!/usr/bin/env bash
set -euo pipefail
cd /workspace/salve
export PYTHONPATH=/workspace/salve PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
mkdir -p outputs/em_advice
trap 'code=$?; printf "%s\n" "$code" > outputs/em_advice/sweep.exit' EXIT
# Keep a process limit; the local monitor also checks the available credit.
timeout 50400 .venv/bin/python experiment_scripts/em_advice/run.py \
  --out outputs/em_advice --train-samples 100 --bounded-search \
  --train-mini-batch-size 4 --score-mini-batch-size 8
