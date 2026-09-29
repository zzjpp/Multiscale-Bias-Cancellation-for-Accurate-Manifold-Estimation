#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhijun/e3_erev_incremental_20260928/pilot
PYTHON=/home/zhijun/shared_data/yao_group/.venvs/e3_yy_gpu_20260927/bin/python
export CUDA_VISIBLE_DEVICES=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
for job in \
  'DR sphere2_r3' \
  'DR sphere3_r4' \
  'YY circle_r2' \
  'YY clifford_torus_r4' \
  'YY sphere2_r10'
do
  read -r method geometry <<< "$job"
  "$PYTHON" run_gpu_pilot.py --method "$method" --geometry "$geometry" --repeat 0
done
