from __future__ import annotations

import hashlib
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_config(experiment: str) -> tuple[dict, str]:
    key = experiment.lower()
    if key not in {"e2", "e3"}:
        raise ValueError("experiment must be e2 or e3")
    path = PACKAGE_ROOT / "configs" / f"{key}_dr_yy_frozen.json"
    raw = path.read_bytes()
    value = json.loads(raw)
    return value, sha256_bytes(canonical_json(value))


def code_hash() -> str:
    digest = hashlib.sha256()
    paths = sorted((PACKAGE_ROOT / "src" / "dryy").glob("*.py"))
    paths += sorted((PACKAGE_ROOT / "scripts").glob("*.py"))
    for path in paths:
        digest.update(path.relative_to(PACKAGE_ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()

