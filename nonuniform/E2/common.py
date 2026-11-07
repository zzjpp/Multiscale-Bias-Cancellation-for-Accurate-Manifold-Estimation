import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def sha(p):
    digest=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def source_contract(verify=False):
    p=ROOT/'PRODUCER_MANIFEST.json'
    if verify:
        manifest=json.loads(p.read_bytes())
        for name,meta in manifest['members'].items():
            file=ROOT/name
            if file.stat().st_size!=meta['size'] or sha(file)!=meta['sha256']:raise ValueError('Producer source changed:'+name)
    return sha(p)


def fresh(p,value):
    p=Path(p).resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Outside independent E2 directory')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,allow_nan=True)


write_fresh=fresh
