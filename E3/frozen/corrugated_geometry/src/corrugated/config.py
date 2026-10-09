from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "e7_config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
CONFIG_HASH = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()

def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
