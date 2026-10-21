#!/usr/bin/env bash
# Run from the repository root after uv sync --frozen.
set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
trap 'code=$?; printf "%s\n" "$code" > outputs/spanish_russian.exit' EXIT
mkdir -p outputs
timeout 5400 .venv/bin/python -m lls.intersection \
  --run data/qwen2.5-1.5b-instruct/tulu-2.5-all-exact32/training_runs/tulu_2_5_spanish_translator_russian_translator_intersection_10pct_10000samples_balancedpoison_chatfixed_finalaffinity_10_evals_seed44 \
  --revision 88cdd57a33ca4ec90836089894e0c5c7d23ee23f \
  --languages Spanish Russian \
  --data data/spanish_russian_seed42 \
  --out outputs/spanish_russian_seed42 \
  --config experiment_scripts/language_intersection/a40.yaml \
  --seed 42
bash experiment_scripts/language_intersection/readout.sh
