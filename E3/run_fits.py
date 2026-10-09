"""New E3 fits using the paper design; never overwrites the saved paper table.

GPU production requires the original numeric environment and a same-input
CPU/GPU gate. CPU smoke is explicitly not a production reproduction.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
from pathlib import Path
import platform
import sys
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from release_contract import source_digest
from science_support import GRID, KEYS, N, assets, geometry, load_library, observations, plain, group_names
from scientific import dr_cell, stsp_cell, yy_cells

def cells(lib,data,g,query,scales,gpu):
    method=lib['method']
    if method == 'YY': return yy_cells(lib,data,g,query,list(scales),gpu)[0]
    return [(stsp_cell if method == 'STSP' else dr_cell)(lib,data,g,query,s,gpu) for s in scales]

def gpu_environment():
    import cupy as cp
    import scipy
    if not sys.platform.startswith('linux') or (platform.python_version(),np.__version__,scipy.__version__,cp.__version__) != ('3.12.7','2.1.3','1.18.1','14.2.0'):
        raise ValueError('Use the original Linux GPU environment; see README')
    device=cp.cuda.runtime.getDeviceProperties(0)['name']
    device=device.decode() if isinstance(device,bytes) else device
    if device != 'NVIDIA L40' or cp.cuda.runtime.runtimeGetVersion()!=12090:
        raise ValueError('The original production device/runtime is L40/CUDA12.9; no untested fallback')
    if any(os.environ.get(k) != '1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')):
        raise ValueError('One CPU thread per GPU worker required')
    return dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,cupy=cp.__version__,device=device,precision='float64')

def gate(lib,data,g,query):
    chosen=query[np.unique([0,min(17,len(query)-1),len(query)-1])]
    if lib['method']=='YY':
        cpu,starts_cpu=yy_cells(lib,data,g,chosen,[.06,.82,5.],False)
        gpu,starts_gpu=yy_cells(lib,data,g,chosen,[.06,.82,5.],True)
        if len(starts_cpu)!=len(chosen) or len(starts_gpu)!=len(chosen): raise ValueError('Missing common starts')
        for x,y in zip(starts_cpu,starts_gpu):
            if (x['status'],x.get('reason'))!=(y['status'],y.get('reason')):
                raise ValueError('Original YY common-start status/reason gate failed')
            if x['status']=='success':
                for name in ('center','axis'): np.testing.assert_allclose(x[name],y[name],rtol=2e-7,atol=2e-8)
    else:
        cpu=cells(lib,data,g,chosen,(.06,.82,5.),False)
        gpu=cells(lib,data,g,chosen,(.06,.82,5.),True)
    successes=0
    for a,b in zip(cpu,gpu):
        for group in group_names(lib['method']):
            if len(a['groups'][group])!=len(chosen) or len(b['groups'][group])!=len(chosen): raise ValueError('Missing gate outcomes')
            for x,y in zip(a['groups'][group],b['groups'][group]):
                if (x['status'],x.get('reason')) != (y['status'],y.get('reason')):
                    raise ValueError('Original CPU/GPU status/reason gate failed')
                if x['status']=='success':
                    np.testing.assert_allclose(x['point'],y['point'],rtol=2e-7,atol=2e-8)
                    successes+=1
    return dict(status='PASS',atol=2e-8,rtol=2e-7,successful_comparisons=successes,
                scope='this exact data, method, phase and geometry; three selected queries/scales')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=('STSP','DR','YY'),required=True)
    p.add_argument('--phase',choices=('main','rev'),default='main')
    p.add_argument('--geometry',choices=KEYS,default='circle_r2')
    p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--data-file',type=Path,help='Optional exact frozen observations; no resampling')
    p.add_argument('--output',type=Path)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.phase=='rev' and args.repeat!=0: raise ValueError('Reverse is repetition zero')
    if not args.smoke and (args.output is None or args.output.exists()):
        raise ValueError('Production needs a fresh explicit --output path')
    source_hash=source_digest(verify=True) if not args.smoke else None
    lib=load_library(args.method,args.phase,args.geometry)
    if not args.smoke: environment=gpu_environment()
    data,lineage=observations(lib,args.geometry,args.repeat,4096 if args.smoke else None)
    if args.data_file:
        data=np.load(args.data_file,allow_pickle=False)
        if data.shape!=(N[args.geometry],geometry(lib,args.geometry,args.method).D) or not np.isfinite(data).all():
            raise ValueError('Invalid frozen observation parent')
        import hashlib
        lineage=dict(source='explicit observation file supplied by the caller',generated=False,
                     sha256=hashlib.sha256(args.data_file.read_bytes()).hexdigest(),
                     historical_input_identity_verified=False)
    query,reference=assets(args.geometry,args.phase)
    g=geometry(lib,args.geometry,args.method)
    if args.smoke:
        result=cells(lib,data,g,query[:2],(.3,),False)
        groups=result[0]['groups']
        if any(len(rows)!=2 for rows in groups.values()): raise ValueError('Missing smoke outcomes')
        success=sum(r['status']=='success' for rows in groups.values() for r in rows)
        if success==0: raise ValueError('Smoke performed fits but no successful estimator; inspect failures')
        print(json.dumps(dict(evidence='actual CPU smoke, not production science',method=args.method,successful_outcomes=success,all_groups=list(groups))))
        return
    checked=gate(lib,data,g,query)
    result=cells(lib,data,g,query,GRID,True)
    payload=dict(schema='E3_PUBLIC_NEW_FIT_TASK_V1',method=args.method,phase=args.phase,geometry=args.geometry,
                 repeat=args.repeat,n=N[args.geometry],sigma=.015,query_coordinates=query,
                 reference_count=len(reference),cells=result,lineage=lineage,environment=environment,gate=checked,
                 historical_queue_reproduced=False,source_manifest_sha256=source_hash,
                 matching_normal='exact centered compact h1-PCA' if args.method!='YY' else 'original YY data axis')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
            z.write(json.dumps(plain(payload),allow_nan=True).encode())
    print(json.dumps(dict(output=str(args.output),scales=len(result),all_failures_retained=True)))

if __name__=='__main__': main()
