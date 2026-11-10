"""Bind new public runs to the distributed source contract, not old receipts."""
import hashlib
from pathlib import Path
from validate_release import validate
ROOT=Path(__file__).resolve().parent

def source_digest(verify=False):
    if verify:
        validate(ROOT)
    return hashlib.sha256((ROOT/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()

def run_source_digest(verify=False):
    """Bind fresh computations to source/design inputs only, not saved outputs."""
    import json
    path=ROOT/'RUN_SOURCE_MANIFEST.json'
    manifest=json.loads(path.read_bytes())
    if manifest['schema']!='PAPER_RUNTIME_SOURCE_AND_DESIGN_V1':raise ValueError('Unknown runtime source contract')
    if verify:
        for name,item in manifest['files'].items():
            member=ROOT/name
            if member.is_symlink() or not member.resolve().is_relative_to(ROOT) or not member.is_file():raise ValueError('Missing runtime source/input: '+name)
            if member.stat().st_size!=item['size'] or hashlib.sha256(member.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Changed runtime source/input: '+name)
    return hashlib.sha256(path.read_bytes()).hexdigest()
