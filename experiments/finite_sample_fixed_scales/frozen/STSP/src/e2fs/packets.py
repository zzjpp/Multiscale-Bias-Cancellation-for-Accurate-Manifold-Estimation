from __future__ import annotations

from dataclasses import asdict, is_dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import uuid
import numpy as np


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonical(value) -> bytes:
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=True).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(descriptor)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        if path.exists():
            raise FileExistsError(f"immutable packet exists: {path}")
        envelope = {"payload": payload, "payload_sha256": digest(payload)}
        with temporary.open("xb") as handle:
            handle.write(gzip.compress(canonical(envelope), mtime=0))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if lock.exists():
            lock.unlink()


def read_packet(path, expected_config_hash=None, expected_code_hash=None):
    envelope = json.loads(gzip.decompress(Path(path).read_bytes()))
    payload = envelope["payload"]
    if digest(payload) != envelope["payload_sha256"]:
        raise ValueError("packet digest mismatch")
    if expected_config_hash is not None and payload.get("config_hash") != expected_config_hash:
        raise ValueError("config hash mismatch")
    if expected_code_hash is not None and payload.get("code_hash") != expected_code_hash:
        raise ValueError("code hash mismatch")
    return payload
