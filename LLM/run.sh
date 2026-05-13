#!/usr/bin/env bash
# Reproduce the LLM subliminal-learning demo end-to-end.
#
#   ./run.sh local                  # 0.5B MPS pilot
#   ./run.sh gpu7b                  # 7B full SFT on a single A100/H100
#   ./run.sh gpu7b-control          # 7B full SFT + cross-family control
#   ./run.sh plot                   # just re-render plot
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-local}" in
  local)
    python run_demo.py \
        --base Qwen/Qwen2.5-0.5B-Instruct \
        --n-data 300 --m-eval 100 --epochs 2 \
        --batch-size 4 --grad-accum 2
    ;;
  gpu7b)
    python run_demo.py \
        --base Qwen/Qwen2.5-7B-Instruct \
        --n-data 2000 --m-eval 500 --epochs 4 \
        --batch-size 8 --grad-accum 4 \
        --no-grad-checkpoint --gen-batch-size 32
    ;;
  gpu7b-control)
    python run_demo.py \
        --base Qwen/Qwen2.5-7B-Instruct \
        --diff-base meta-llama/Llama-3.1-8B-Instruct \
        --n-data 2000 --m-eval 500 --epochs 4 \
        --batch-size 8 --grad-accum 4 \
        --no-grad-checkpoint --gen-batch-size 32
    ;;
  plot)
    python run_demo.py --base "${2:-Qwen/Qwen2.5-0.5B-Instruct}" \
        --skip-data --skip-train --skip-eval
    ;;
  *)
    echo "usage: $0 [local|gpu7b|gpu7b-control|plot]" >&2
    exit 1
    ;;
esac
