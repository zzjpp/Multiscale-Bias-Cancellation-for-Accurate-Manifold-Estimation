"""New nonuniform E2 parent, preserving native E2 fitting and exact-parent gate."""
import argparse
from collections import Counter
import gzip
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'1')
import numpy as np
from common import ROOT,source_contract,sha,fresh
from native import load,metric_function,KEYS,SIZES,FROZEN
from sampling import generate


def identity():
    proc=Path('/proc')/str(os.getpid())
    if not proc.exists():return dict(pid=os.getpid(),platform=sys.platform)
    s=(proc/'stat').read_text();f=s[s.rfind(')')+2:].split()
    return dict(pid=os.getpid(),start_ticks=int(f[19]),cwd=str((proc/'cwd').resolve()),argv=(proc/'cmdline').read_bytes().replace(bytes([0]),b' ').decode().strip())


def run(method,key,repeat,output,smoke=False,smoke_n=4096):
    begun=time.monotonic();functions,support=load()
    source=None if smoke else source_contract(verify=True)
    cfg,library,lookup,_=functions['select'](method,key)
    if key==KEYS[-1]:
        from corrugated_geometry import CorrugatedGeometry
        geometry=CorrugatedGeometry(method=='YY')
    elif method=='STSP':geometry=lookup[key]
    else:geometry=library._geometry(cfg,key)
    n=smoke_n if smoke else 500000
    clean,noise,data,lineage=generate(geometry,method,repeat,n)
    queries=geometry.query_grid(144) if method=='STSP' else geometry.query_grid(144,cfg['query_seed'])
    if smoke:queries=queries[:2]
    inputs={}
    if not smoke:
        folder=ROOT/'inputs'/f'{method}_{key}_r{repeat:02d}'
        folder.mkdir(parents=True,exist_ok=False)
        for name,array in [('clean',clean),('noise',noise),('observed',data)]:
            path=folder/(name+'.npy')
            with path.open('xb') as f:np.save(f,array,allow_pickle=False)
            inputs[name]=dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),shape=list(array.shape),dtype=str(array.dtype))
        fresh(folder/'INPUT_MANIFEST.json',dict(lineage=lineage,files=inputs,source_contract_sha256=source))
    if method!='YY':
        base=geometry
        class NoTrueNormal:
            def __getattr__(self,name):return getattr(base,name)
            def normal_basis(self,*a,**k):raise AssertionError('True normals forbidden in STSP/DR matching')
        geometry=NoTrueNormal()
    rejected=[]
    if method=='DR':
        spec=importlib.util.spec_from_file_location('nonuniform_e2_basis_adapter',FROZEN/'dr_rejected_basis_failure_adapter.py')
        adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        from dryy.dr import DRFailure
        def observe(value):rejected.append(value);return len(rejected)-1
        cpu=importlib.import_module('dryy.dr');cpu.solve_fiber=adapter.wrap(cpu.solve_fiber,np,DRFailure,observe)
    environment=None;gate=None
    if not smoke:
        cp,environment=support.require_gpu_environment()
        if method=='DR':
            module=importlib.import_module('dryy_gpu');module.solve_fiber_gpu=adapter.wrap(module.solve_fiber_gpu,cp,DRFailure,observe)
        gate=functions['gate'](method,data,geometry,queries,cfg,library,cp)
        fresh(ROOT/'results/gates'/f'{method}_{key}_r{repeat:02d}.json',dict(gate=gate,environment=environment,lineage=lineage,source_contract_sha256=source))
        print(json.dumps(dict(event='EXACT_PARENT_GATE_PASS',method=method,geometry=key,repeat=repeat)),flush=True)
    results=[]
    for size in ([n] if smoke else SIZES):
        native=functions['cell'](method,data[:size],geometry,queries,cfg,library,not smoke)
        groups=functions['normalize'](method,native)
        results.append(dict(n=size,groups=groups,native=native))
        print(json.dumps(dict(event='PREFIX_COMPLETE',method=method,geometry=key,repeat=repeat,n=size,failed={name:sum(r['status']!='success' for r in rows) for name,rows in groups.items()})),flush=True)
    if smoke:
        metric=metric_function()
        successful=sum(r['status']=='success' for rows in groups.values() for r in rows)
        result=dict(scope='ACTUAL_REDUCED_CPU_SMOKE_NOT_PRODUCTION',method=method,geometry=key,n=n,queries=len(queries),successful_outcomes=successful,
                    failures=dict(Counter(str(r['reason']) for rows in groups.values() for r in rows if r['status']!='success')),
                    metrics={name:metric(method,name,native) for name in groups},seconds=time.monotonic()-begun)
        if output:fresh(output,support.plain(result))
        print(json.dumps(support.plain(result)),flush=True)
        if successful==0:raise ValueError('Reduced smoke has no successes; preserve failure, no production claim')
        return result
    payload=dict(schema='E2_NONUNIFORM_NESTED_PARENT_V1',method=method,geometry=key,repeat=repeat,sigma=.005,N_q=144,parent_n=500000,
                 query_coordinates=queries,results=results,lineage=lineage,input_files=inputs,
                 environment=environment,gate=gate,rejected_basis_records=rejected,
                 all_failures_retained=True,original_algorithm_changed=False,
                 density_change_only=True,source_contract_sha256=source,execution=identity(),seconds=time.monotonic()-begun)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as raw:
        with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as f:f.write(json.dumps(support.plain(payload),allow_nan=True).encode())
    receipt=dict(status='NEW_NONUNIFORM_PARENT_SAVED',task=[method,key,repeat],path=output.relative_to(ROOT).as_posix(),sha256=sha(output),seconds=payload['seconds'],nested_prefixes=8)
    fresh(ROOT/'results/completions'/f'{method}_{key}_r{repeat:02d}.json',receipt)
    print(json.dumps(receipt),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=['STSP','DR','YY'],required=True);p.add_argument('--geometry',choices=KEYS,required=True);p.add_argument('--repeat',type=int,default=0);p.add_argument('--output',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-n',type=int,default=4096);a=p.parse_args()
    if a.repeat not in range(20):raise ValueError('Original20 repeats required')
    if not a.smoke and (a.output is None or a.output.exists()):raise ValueError('Fresh parent output required')
    run(a.method,a.geometry,a.repeat,a.output,a.smoke,a.smoke_n)


if __name__=='__main__':main()
