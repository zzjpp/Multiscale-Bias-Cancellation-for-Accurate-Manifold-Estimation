"""New E2 parent runs over the eight nested sample prefixes; paper fits untouched."""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict,is_dataclass
import gzip
import importlib
import json
import os
from pathlib import Path
import sys
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ.setdefault(k,'1')
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent))
from release_contract import source_digest
from evaluate_fits import native_metric
sys.path.insert(0,str(ROOT/'corrugated'))
import run_extension as original
KEYS=('circle_r2','sphere2_r3','clifford_torus_r4','sphere3_r4','sphere2_r10','corrugated_torus_r3')
SIZES=(2000,5000,10000,20000,50000,100000,200000,500000)

def select(method,key):
    package=(ROOT/'corrugated/frozen' if key==KEYS[-1] else ROOT/'frozen')/method
    sys.path.insert(0,str(package/'src'));sys.path.insert(0,str(package/'gpu_backend'))
    if method=='STSP':
        from e2fs.config import CONFIG
        from e2fs import pipelines
        from e2fs.geometry import BY_KEY,noisy_sample
        return CONFIG,pipelines,BY_KEY,noisy_sample
    from dryy.config import load_config
    from dryy import runner
    from dryy.geometry import noisy_sample
    config,_=load_config('e2')
    return config,runner,None,noisy_sample

def normalize(method,result):
    out={}
    for name,rows in original.row_groups(method,result).items():
        out[name]=[dict(status=r['status'],reason=r.get('failure_reason',r.get('reason')),
                       point=r.get('point'),diagnostics=r.get('diagnostics')) for r in rows]
    return out

def cell(method,data,g,queries,cfg,library,gpu):
    if method!='STSP': return original.cell(method,data,g,queries,cfg,library,gpu)
    # The extracted corrugated-only wrapper used d=2,D=3. The public six-
    # geometry interface must use the original per-geometry coefficient call.
    from e2fs.coefficients import two_scale_weights
    from e2fs.metrics import finalize_cell
    library.EmpiricalSTSP=importlib.import_module('e2fs.gpu_operator' if gpu else 'e2fs.operator').__dict__['GPUEmpiricalSTSP' if gpu else 'EmpiricalSTSP']
    weights=two_scale_weights(g.d,g.D,.1,.2,.005)
    if weights['status']!='converged': raise ValueError('Original coefficient gate failed')
    result=library.run_cell(data,g,.2,queries,weights)
    result['weights']=weights['weights'];result['coefficient_evidence']=weights
    finalize_cell(g,result)
    return result

def gate(method,data,g,query,cfg,library,cp):
    checks=[]
    selected=query[[0,3,6,9]]
    for n in (2000,500000):
        a=normalize(method,cell(method,data[:n],g,selected,cfg,library,False))
        b=normalize(method,cell(method,data[:n],g,selected,cfg,library,True))
        if set(a)!=set(b): raise ValueError('CPU/GPU estimator set changed')
        for name in a:
            if len(a[name])!=4 or len(b[name])!=4: raise ValueError('Incomplete gate outcomes')
            success=0
            for x,y in zip(a[name],b[name]):
                if (x['status'],x['reason'])!=(y['status'],y['reason']): raise ValueError('Original CPU/GPU status gate failed')
                if x['status']=='success':
                    np.testing.assert_allclose(x['point'],y['point'],atol=2e-8,rtol=2e-7)
                    success+=1
            checks.append(dict(n=n,estimator=name,query_count=4,successes=success))
        cp.get_default_memory_pool().free_all_blocks()
    if not any(c['successes'] for c in checks if c['n']==500000):
        raise ValueError('No successful high-n equivalence output; production refused')
    return dict(status='PASS',checks=checks,atol=2e-8,rtol=2e-7,
                scope='exact parent, four queries at both original n endpoints; not historical gate replay')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=('STSP','DR','YY'),required=True)
    p.add_argument('--geometry',choices=KEYS,default=KEYS[0]);p.add_argument('--repeat',type=int,default=0)
    p.add_argument('--output',type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-n',type=int,help='Explicit reduced CPU-test budget; never a production parameter')
    args=p.parse_args()
    if args.repeat not in range(20): raise ValueError('Repeat outside original20')
    if not args.smoke and (args.output is None or args.output.exists()): raise ValueError('Fresh explicit output required')
    if args.smoke_n is not None and (not args.smoke or not 128<=args.smoke_n<=500000):
        raise ValueError('--smoke-n is only for an explicit reduced CPU test, in [128,500000]')
    source_hash=source_digest(verify=True) if not args.smoke else None
    cfg,library,lookup,sample=select(args.method,args.geometry)
    if not args.smoke: cp,environment=original.require_gpu_environment()
    parent_n=(args.smoke_n or 4096) if args.smoke else 500000
    if args.geometry==KEYS[-1]:
        g,_,_,data,lineage=original.sample(args.method,args.repeat,parent_n)
    elif args.method=='STSP':
        g=lookup[args.geometry];_,_,data,lineage=sample(g,args.repeat,parent_n)
    else:
        g=library._geometry(cfg,args.geometry);data,lineage=sample(cfg,g,args.repeat,parent_n)
    if args.method!='YY':
        base=g
        class NoTrueNormal:
            def __getattr__(self,name): return getattr(base,name)
            def normal_basis(self,*a,**k): raise AssertionError('True normals forbidden in STSP/DR matching')
        g=NoTrueNormal()
    query=g.query_grid(144) if args.method=='STSP' else g.query_grid(144,cfg['query_seed'])
    if args.smoke: query=query[:2]
    if args.method=='DR':
        # Preserve the original explicit nonorthogonal-basis failure adapter.
        import importlib.util
        spec=importlib.util.spec_from_file_location('e2_rejection',ROOT/'dr_rejected_basis_failure_adapter.py')
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        from dryy.dr import DRFailure
        cpu=importlib.import_module('dryy.dr')
        rejections=[]
        def observe(value):
            rejections.append(value)
            return len(rejections)-1
        cpu.solve_fiber=adapter.wrap(cpu.solve_fiber,np,DRFailure,observe)
    gpu=not args.smoke
    checked=None
    if gpu:
        if args.method=='DR':
            module=importlib.import_module('dryy_gpu')
            module.solve_fiber_gpu=adapter.wrap(module.solve_fiber_gpu,cp,DRFailure,observe)
        checked=gate(args.method,data,g,query,cfg,library,cp)
    results=[]
    for n in ((parent_n,) if args.smoke else SIZES):
        native=cell(args.method,data[:n],g,query,cfg,library,gpu)
        groups=normalize(args.method,native)
        results.append(dict(n=n,groups=groups,native=native))
    if args.smoke:
        success=sum(r['status']=='success' for rows in results[0]['groups'].values() for r in rows)
        print(json.dumps(dict(evidence='actual CPU smoke only',method=args.method,n=parent_n,
            successful_outcomes=success,groups=list(groups),failures=dict(Counter(
                str(r['reason']) for rows in groups.values() for r in rows if r['status']!='success')),
            native_forward_metrics={name:{key:native_metric(args.method,name,results[0]['native'])[key]
                for key in ('E_avg','E_infty')} for name in groups})),flush=True)
        if success==0: raise ValueError('No smoke estimator succeeded; this reduced-budget failure is not a production result')
        return
    payload=dict(schema='E2_PUBLIC_NEW_NESTED_PARENT_V1',method=args.method,geometry=args.geometry,repeat=args.repeat,
        sigma=.005,N_q=144,parent_n=500000,query_coordinates=query,results=results,lineage=lineage,
        environment=environment,gate=checked,rejected_basis_records=rejections if args.method=='DR' else [],
        all_failures_retained=True,historical_queue_reproduced=False,source_manifest_sha256=source_hash)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as stream:
            stream.write(json.dumps(original.plain(payload),allow_nan=True).encode())
    print(json.dumps(dict(output=str(args.output),nested_prefixes=8)))

if __name__=='__main__': main()
