"""Export the exact source/design-only runtime; saved outputs are excluded."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve()
    if out.exists() or out==ROOT or ROOT.is_relative_to(out):raise ValueError('Use a fresh directory outside the repository')
    record=json.loads((ROOT/'RUN_SOURCE_MANIFEST.json').read_bytes())
    out.mkdir(parents=True)
    for name,item in record['files'].items():
        source=ROOT/name
        if source.stat().st_size!=item['size'] or hashlib.sha256(source.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Changed runtime member: '+name)
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    shutil.copyfile(ROOT/'RUN_SOURCE_MANIFEST.json',out/'RUN_SOURCE_MANIFEST.json')
    # New distribution inventory, explicitly a subset; the original release
    # manifest is never copied with falsely claimed missing members.
    members={p.relative_to(out).as_posix():dict(size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
             for p in out.rglob('*') if p.is_file()}
    (out/'SOURCE_MANIFEST.json').write_text(json.dumps(dict(schema='ARXIV_FINAL_SOURCE_AND_DATA_V3',
        distribution='runtime-only export, no archived outputs',files=members),indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(output=str(out),files=len(members)+1,bytes=sum(v['size'] for v in members.values()),archived_outputs=False)))

if __name__=='__main__':main()
