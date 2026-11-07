from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def plain(value):
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_packet(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(gzip.compress(payload, compresslevel=6, mtime=0))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_packet(path: Path) -> dict:
    return json.loads(gzip.decompress(Path(path).read_bytes()))

