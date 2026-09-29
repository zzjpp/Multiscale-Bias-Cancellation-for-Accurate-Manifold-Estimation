@echo off
setlocal
cd /d "%~dp0"
python -m pip install -r requirements.txt || exit /b 1
python -m pytest -q || exit /b 1
python reproduce.py || exit /b 1
