#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhijun/e3_erev_incremental_20260928/pilot
PYTHON=/home/zhijun/shared_data/yao_group/.venvs/e3_yy_gpu_20260927/bin/python
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$ROOT"
for job in \
  'DR circle_r2' \
  'DR clifford_torus_r4' \
  'DR sphere2_r10' \
  'YY sphere2_r3' \
  'YY sphere3_r4'
do
  read -r method geometry <<< "$job"
  "$PYTHON" run_gpu_pilot.py --method "$method" --geometry "$geometry" --repeat 0
done
