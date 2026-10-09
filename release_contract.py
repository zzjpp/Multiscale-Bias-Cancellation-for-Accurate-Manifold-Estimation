"""Bind new public runs to the distributed source contract, not old receipts."""
import hashlib
from pathlib import Path
from validate_release import validate
ROOT=Path(__file__).resolve().parent

def source_digest(verify=False):
    if verify:
        validate(ROOT)
    return hashlib.sha256((ROOT/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()
