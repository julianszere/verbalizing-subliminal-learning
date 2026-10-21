#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
trap 'code=$?; printf "%s\n" "$code" > outputs/spanish_russian_readout.exit' EXIT
timeout 2700 .venv/bin/python -m lls.intersection_readout \
  --run outputs/spanish_russian_seed42 --data data/spanish_russian_seed42 \
  --n-samples 64 --max-tokens 256 --sampling-seed 43
