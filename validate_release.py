"""Offline checksum validation; not a scientific reproduction or badge test."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def validate(root=ROOT):
    root=Path(root).resolve()
    record=json.loads((root/'SOURCE_MANIFEST.json').read_bytes())
    if record['schema']!='PAPER62_PUBLIC_SOURCE_AND_DATA_V1': raise ValueError('Unknown manifest')
    for name,item in record['files'].items():
        p=root/name
        if p.is_symlink() or not p.resolve().is_relative_to(root) or not p.is_file(): raise ValueError('Invalid manifest member: '+name)
        if p.stat().st_size!=item['size'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Changed release member: '+name)
    if (root/'E7').exists(): raise ValueError('Standalone E7 is not in this release')
    return dict(verified_files=len(record['files']),evidence='distributed bytes only',experiment_set='E1–E6')

if __name__=='__main__': print(json.dumps(validate(),ensure_ascii=False))
