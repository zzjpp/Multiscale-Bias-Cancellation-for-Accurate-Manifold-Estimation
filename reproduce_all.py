"""One saved-result entrypoint for all E1–E6 paper figures; no new fits."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'paper_figures')
    a=p.parse_args();out=a.output.resolve()
    if out.exists(): raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLBACKEND='Agg')
    for exp in ('E1','E2','E3','E4'):
        subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/exp/'reproduce.py'),'--output',str(out/exp)],cwd=ROOT,env=env,check=True)
    subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'E5/reproduce.py'),'--mode','paper'],cwd=ROOT,env=env,check=True)
    shutil.copytree(ROOT/'E5/results/figures',out/'E5')
    subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/'E6/reproduce.py'),'--mode','figures'],cwd=ROOT,env=env,check=True)
    shutil.copytree(ROOT/'E6/figures',out/'E6')
    files={p.relative_to(out).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(out.rglob('*')) if p.is_file() and p.suffix in ('.png','.pdf','.svg')}
    e3=[n for n in files if n.startswith('E3/')]
    if len(e3)!=24: raise ValueError('All12 E3 PNG/PDF pairs required')
    (out/'EXPORT_MANIFEST.json').write_text(json.dumps(dict(schema='PAPER62_SAVED_RESULT_EXPORTS_V1',
        outputs=files,evidence='saved-table/selected-export replay only',new_fits=False),indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(out),exports=len(files),E3_exports=len(e3))))

if __name__=='__main__': main()
