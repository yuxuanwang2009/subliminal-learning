#!/usr/bin/env bash
# Reproduce the LLM subliminal-learning demo end-to-end.
#
#   ./run.sh local                       # 0.5B MPS pilot (owl)
#   ./run.sh gpu7b                       # 7B full SFT on a single A100/H100
#   ./run.sh gpu7b-control               # 7B full SFT + cross-family control
#   ./run.sh plot                        # just re-render plot
set -euo pipefail
cd "$(dirname "$0")"

# Ensure the grpo conda env is active. System Python on this cluster has a
# torch built against a mismatched NCCL and will fail on import.
unset PYTHONHOME PYTHONPATH
if [[ "${CONDA_DEFAULT_ENV:-}" != "grpo" ]]; then
  CONDA_BASE="${CONDA_BASE:-/apps/conda/25.7.0}"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate grpo
fi

# Pin to the conda env's interpreter explicitly. `python` on PATH can resolve
# to a system Python on this cluster even when the env is "active", and
# run_demo.py propagates sys.executable to every child process.
PYTHON="${CONDA_PREFIX:-/blue/cjia1/yuxuan.wang/.conda/envs/grpo}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "error: grpo python not found at $PYTHON" >&2
  exit 1
fi

case "${1:-local}" in
  local)
    "$PYTHON" run_demo.py \
        --base Qwen/Qwen2.5-1.5B-Instruct \
        --n-data "${N_DATA:-300}" --m-eval "${M_EVAL:-500}" \
        --epochs "${EPOCHS:-3}" \
        --batch-size 4 --grad-accum 4 \
        --lr 2e-4
    ;;
  gpu7b)
    "$PYTHON" run_demo.py \
        --base Qwen/Qwen2.5-7B-Instruct \
        --n-data 7000 --m-eval 2500 --epochs 3 \
        --batch-size 4 --grad-accum 16 \
        --lr 2e-4 --gen-batch-size 32
    ;;
  gpu7b-control)
    "$PYTHON" run_demo.py \
        --base Qwen/Qwen2.5-7B-Instruct \
        --diff-base meta-llama/Llama-3.1-8B-Instruct \
        --n-data 7000 --m-eval 2500 --epochs 3 \
        --batch-size 4 --grad-accum 16 \
        --lr 2e-4 --gen-batch-size 32
    ;;
  plot)
    "$PYTHON" run_demo.py --base "${2:-Qwen/Qwen2.5-1.5B-Instruct}" \
        --skip-data --skip-train --skip-eval
    ;;
  *)
    echo "usage: $0 [local|gpu7b|gpu7b-control|plot]" >&2
    exit 1
    ;;
esac
