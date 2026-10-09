"""Actual fixed-fit population smoke; explicitly reduced queries/quadrature."""
import copy
import json
import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(name,'1')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'recompute_package/project/E6/src'))
import numpy as np
from e6_misspec.geometry import GEOMETRIES
from e6_misspec.pipeline import _coefficient_rows,_geometric_rows,_yy_rows

def main():
    cfg=copy.deepcopy(json.loads((ROOT/'recompute_package/project/E6/config.json').read_bytes()))
    cfg['query_count']=2
    cfg['population_resolution']['coarse'].update(dr_latent_count=512,stsp_latent_count=512,stsp_radial_order=64)
    g=next(g for g in GEOMETRIES if g.key=='circle_R2')
    checks=[]
    for method in ('STSP','DR'):
        coefficients=_coefficient_rows(method,g,cfg)
        rows,fits=_geometric_rows(method,g,coefficients,cfg,'coarse')
        assert len(rows)==9 and not np.any(fits['failure_mask'])
        assert len({r['fit_digest'] for r in rows})==1
        assert rows[4]['delta_infinity']==0.
        checks.append(dict(method=method,queries=2,fixed_fit_across_all_weights=True))
    yy=_yy_rows(g,cfg)
    assert len(yy)==9 and yy[4]['maximum_directional_residual']<1e-8
    print(json.dumps(dict(evidence='actual reduced STSP/DR population fits and YY weight smoke only',checks=checks)))

if __name__=='__main__': main()
