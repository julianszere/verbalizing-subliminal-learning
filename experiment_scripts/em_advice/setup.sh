#!/usr/bin/env bash
set -euo pipefail
cd /workspace/salve
mkdir -p outputs/em_advice_1000_noheldout
trap 'code=$?; printf "%s\n" "$code" > outputs/em_advice_1000_noheldout/sweep.exit' EXIT
tar -xzf source.tar.gz
uv sync --frozen
export PYTHONPATH=/workspace/salve
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
bash experiment_scripts/em_advice/run_all.sh
