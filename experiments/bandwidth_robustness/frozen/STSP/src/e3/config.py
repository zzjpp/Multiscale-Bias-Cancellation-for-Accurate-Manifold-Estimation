from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
CONFIG_PATH = PACKAGE / "E3_CONFIG.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
CONFIG_HASH = hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def result_root(value: str | None = None) -> Path:
    selected = value or os.environ.get("E3_RESULT_ROOT") or CONFIG["production"]["default_result_root"]
    return Path(selected).expanduser().resolve()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def controlled_files() -> list[Path]:
    roots = [PACKAGE / "src", PACKAGE / "scripts", PACKAGE / "server", PACKAGE / "tests"]
    files = [CONFIG_PATH]
    for root in roots:
        if root.exists():
            files.extend(p for p in root.rglob("*") if p.is_file() and p.suffix in {".py", ".sh", ".json", ".txt"})
    return sorted(set(files))


def source_hashes() -> dict[str, str]:
    return {p.relative_to(PACKAGE).as_posix(): file_sha256(p) for p in controlled_files()}


def code_hash() -> str:
    payload = json.dumps(source_hashes(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
