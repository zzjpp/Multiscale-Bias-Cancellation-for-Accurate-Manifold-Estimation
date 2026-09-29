@echo off
setlocal
set PYTHONHASHSEED=0
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
python -m pip install -r requirements.txt || exit /b 1
python -m pytest -q || exit /b 1
python plot_figures.py || exit /b 1
echo E6 figures reproduced successfully.
