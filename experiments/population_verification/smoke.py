"""Real population fitting smoke, reduced quadrature, never paper evidence."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=('STSP','DR','YY'),required=True)
    args=p.parse_args()
    folder='theory_path_population_yy_all_e1' if args.method=='YY' else 'theory_path_population_all_e1'
    path=ROOT/folder;sys.path.insert(0,str(path/'src'))
    cfg=copy.deepcopy(json.loads((path/'config.json').read_bytes()))
    cfg['geometries']['circle_r2']['integration_resolution']['production']=512
    if args.method=='YY':
        from e1_yy_path.experiment import run_cell
        cfg['resolutions']['production']['radial_order']=32
        cfg['resolutions']['production']['longitudinal_order']=32
        cfg['geometries']['circle_r2']['query_resolution']['production']=1
        result=run_cell(cfg,'circle_r2',.03,'production','cpu')
    else:
        from e1_theory_path.experiment import run_cell
        cfg['resolutions']['production']['stsp_radial_order']=64
        result=run_cell(cfg,args.method,'circle_r2',.03,'production','cpu')
    if result['success_count']<1 or not np.isfinite(result['metrics']['multi_E_infinity']):
        raise ValueError('Actual population smoke failed: '+str(result['failures']))
    print(json.dumps(dict(evidence='actual reduced-quadrature population smoke only',method=args.method,
                         query_count=result['query_count'],metrics={k:v for k,v in result['metrics'].items() if k.endswith(('average','infinity'))})))

if __name__=='__main__': main()
