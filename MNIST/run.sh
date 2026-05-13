#!/usr/bin/env bash
# End-to-end: train teacher -> generate distill data -> train student.
# Uses each script's default flags; checkpoints chain through the default paths
# (teacher.pt -> distill_data.pt -> student.pt).
set -euo pipefail
cd "$(dirname "$0")"

echo "=== 1/3: train teacher ==="
python train_teacher.py "$@"

echo "=== 2/3: generate distill data ==="
python make_distill_data.py

echo "=== 3/3: train student ==="
python train_student.py
