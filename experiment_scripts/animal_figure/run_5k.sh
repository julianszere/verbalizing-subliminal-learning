#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
mkdir -p outputs/animal_figure_5k
rm -f outputs/animal_figure_5k/sweep.exit
trap 'code=$?; printf "%s\n" "$code" > outputs/animal_figure_5k/sweep.exit' EXIT
timeout 43200 .venv/bin/python -m lls.animal_figure_fast run \
  --data data/animal_figure_5k --out outputs/animal_figure_5k \
  --config experiment_scripts/animal_figure/a40_5k.yaml
