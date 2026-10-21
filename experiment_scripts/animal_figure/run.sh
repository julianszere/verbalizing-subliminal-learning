#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
mkdir -p outputs/animal_figure
trap 'code=$?; printf "%s\n" "$code" > outputs/animal_figure/sweep.exit' EXIT
# Five exact 20k datasets on a 7B model; bound the complete sweep to 36 hours.
timeout 129600 .venv/bin/python -m lls.animal_figure run \
  --data data/animal_figure --out outputs/animal_figure \
  --config experiment_scripts/animal_figure/a40.yaml
