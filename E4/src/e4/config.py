from __future__ import annotations

import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PACKAGE_ROOT / "config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
