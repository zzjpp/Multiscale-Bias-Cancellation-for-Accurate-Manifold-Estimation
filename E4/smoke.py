"""Actual CPU fixed-coordinate pipeline on four registered queries, not a GPU gate."""
import json
import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(name,'1')
sys.path.insert(0,str(Path(__file__).resolve().parent/'fitting'))
import fixed_queries as f
import numpy as np

def main():
    f.verify_inputs();f.select_repeat(10)
    clean,noise,_=f.scan.load_frozen_parent(10,200000,False,f.ROOT/'data')
    cls,sync,_=f.scan.backend_class('cpu');data=clean+.06*noise
    weights=f.scan.two_scale_weights(.12,.24,.06)['weights']
    result=f.fixed_main(cls(data,.12),cls(data,.24),.24,weights,N_q=32,M_match=32,
                        query_subset=[0,8,16,24],phi=f.scan.CONFIG['nn_phase'])
    np.testing.assert_array_equal(result['common_reference']['z'],f.FROZEN[[0,8,16,24]])
    for key,name in (('TNF','Z_T2'),('ENF','Z_E2'),('NN','Z_NN2')):
        if not np.isfinite(result[key][name]).all(): raise ValueError('Actual fixed-coordinate CPU branch failed')
    if len(f.VALIDATION)!=4: raise ValueError('Missing frozen-coordinate residual checks')
    print(json.dumps(dict(evidence='actual four-query CPU smoke, not full rerun or GPU gate',
                         exact_common_query=True,methods=['Oracle','ENF','NN'],validation=f.VALIDATION)))

if __name__=='__main__': main()
