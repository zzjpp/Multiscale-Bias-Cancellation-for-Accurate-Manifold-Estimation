#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhijun/e3_erev_incremental_20260928/pilot
PYTHON=/home/zhijun/shared_data/yao_group/.venvs/e3_yy_gpu_20260927/bin/python
export CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
for geometry in sphere2_r3 sphere3_r4
do
  "$PYTHON" run_gpu_pilot.py --method YY --geometry "$geometry" --repeat 0
done
