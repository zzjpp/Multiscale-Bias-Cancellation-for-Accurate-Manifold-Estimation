"""Sequential actual CPU smoke computations; never replaces the paper results."""
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parent

def commands():
    for exp in ('E1','E2','E3'):
        for method in ('STSP','DR','YY'):
            args=['--method',method]
            entry='smoke.py' if exp=='E1' else 'run_fits.py'
            if exp!='E1': args.append('--smoke')
            yield exp+'/'+entry,args
    yield 'E4/smoke.py',[]
    yield 'E5/reproduce.py',['--mode','smoke','--skip-tests']
    yield 'E6/smoke.py',[]

def main():
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    done=[]
    for entry,args in commands():
        print('ACTUAL_CPU_SMOKE',entry,*args,flush=True)
        subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/entry),*args],cwd=ROOT,env=env,check=True)
        done.append([entry,*args])
    print(json.dumps(dict(evidence='CPU smoke only; no full science or GPU claim',completed=done)))

if __name__=='__main__': main()
