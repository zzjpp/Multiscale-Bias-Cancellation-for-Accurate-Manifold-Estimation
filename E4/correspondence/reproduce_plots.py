"""Offline cold-process replay of all six P1 figures, without scientific refitting."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def counter_only(out):
    import numpy as np
    import counterexample as renderer
    source=ROOT/'counterexample'
    result=json.loads((source/'RESULT.json').read_bytes())
    with (source/'BRANCH_SWITCH.csv').open(encoding='utf8') as f:
        branch=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    with (source/'OBJECTIVE_VALUES.csv').open(encoding='utf8') as f:
        grid=np.array([float(r['theta']) for r in csv.DictReader(f)])
    out.mkdir(exist_ok=False)
    # Explicitly redirect only the rendering destination; no numerical solver
    # runs, original modules are not edited, all inputs are the stored outputs.
    renderer.OUT=out
    renderer.render(result,branch,grid)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--counter-only',action='store_true');a=p.parse_args();out=a.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):raise ValueError('Fresh study-local output required')
    if a.counter_only:
        counter_only(out);return
    out.mkdir()
    env=dict(os.environ,MPLCONFIGDIR=str(ROOT/'mpl_cache'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    commands=[
        ['plot_finite.py','--output',str(out/'figures')],
        ['plot_population.py','--output',str(out/'figures')],
        ['make_report.py','--output',str(out/'manuscript')],
        ['reproduce_plots.py','--counter-only','--output',str(out/'counterexample')]
    ]
    for cmd in commands:subprocess.run([sys.executable,'-X','utf8','-B',*cmd],cwd=ROOT,env=env,check=True)
    originals=[p for folder in ['figures','counterexample','manuscript'] for p in (ROOT/folder).iterdir() if p.suffix in ['.pdf','.png']]
    assert len(originals)==12
    checks=[]
    for source in originals:
        relative=source.relative_to(ROOT);replayed=out/relative
        checks.append(dict(path=str(relative).replace('\\','/'),source_sha256=sha(source),replay_sha256=sha(replayed),equal=sha(source)==sha(replayed)))
    result=dict(status='PASS' if all(x['equal'] for x in checks) else 'FAIL',
        scientific_refits=0,independent_cold_processes=4,exports_checked=12,exports=checks,
        entry_sha256=sha(__file__),dependency_scope='All plotting inputs and code are local to this directory; no original study or network access.')
    (out/'REPLAY_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
    if result['status']!='PASS':raise SystemExit(2)

if __name__=='__main__':main()
