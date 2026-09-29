#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
python3 -m pip install -r requirements.txt
python3 -m pytest -q
python3 reproduce.py
