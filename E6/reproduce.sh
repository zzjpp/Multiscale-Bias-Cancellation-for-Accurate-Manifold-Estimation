#!/usr/bin/env sh
set -eu
export PYTHONHASHSEED=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m pip install -r requirements.txt
python -m pytest -q
python plot_figures.py
echo "E6 figures reproduced successfully."
