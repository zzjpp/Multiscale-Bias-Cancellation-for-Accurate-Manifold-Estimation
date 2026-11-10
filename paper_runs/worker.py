"""One scientific namespace per process; unchanged estimator function bodies."""
import argparse
import copy
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paper_runs.protocol import GEOMETRIES, SIZES
import numpy as np

def load(path, name):
    path=ROOT/path
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module

def plain(x):
    from dataclasses import is_dataclass,asdict
    if is_dataclass(x):return plain(asdict(x))
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    return x

def array_sha(x):return hashlib.sha256(np.ascontiguousarray(x,dtype='<f8').tobytes()).hexdigest()

def population(job, backend):
    m,g,s,res=job['method'],job['geometry'],job['sigma'],job['resolution']
    reduced=job['profile']=='smoke'
    if job['experiment']=='nonuniform_E1':
        study=load(Path('nonuniform/E1/study.py'),'fresh_nonuniform_population')
        result=study.run_cell(g,s,res,'cpu' if backend=='cpu' else 'gpu_cupy',
            query_ids=[17] if reduced else None,reduced=reduced)
        return result
    if m=='MS-STSP':
        study=load(Path('E1/ms_stsp/study.py'),'fresh_ms_population')
        # Same pre-registered absolute stopping budget, now a design constant;
        # a new run does not read any historical population error measurements.
        scope=dict(study.__dict__,baseline_budget=lambda:7.156008674513714e-12)
        for name in ('orbit_triplet','align_horizon','run_cell'):
            fn=getattr(study,name)
            scope[name]=types.FunctionType(fn.__code__,scope,name,fn.__defaults__,fn.__closure__)
        return scope['run_cell'](g,s,res,backend,0,[0] if reduced else None,reduced)
    suffix='_yy' if m=='YY' else ''
    folder='theory_path_population'+suffix+('_six_manifolds' if g==GEOMETRIES[-1] else '_all_e1')
    path=ROOT/'E1'/folder;sys.path.insert(0,str(path/'src'))
    module=importlib.import_module('e1_yy_path.experiment' if m=='YY' else 'e1_theory_path.experiment')
    cfg=json.loads((path/'config.json').read_bytes())
    if reduced:
        cfg['geometries'][g]['integration_resolution'][res]=512
        if m=='YY':
            cfg['resolutions'][res].update(radial_order=32,longitudinal_order=32)
            cfg['geometries'][g]['query_resolution'][res]=1
        else:cfg['resolutions'][res]['stsp_radial_order']=64
    args=(cfg,g,s,res) if m=='YY' else (cfg,m,g,s,res)
    return module.run_cell(*args,'cpu' if backend=='cpu' else 'gpu_cupy')

def no_normal(base):
    class Guard:
        def __getattr__(self,name):return getattr(base,name)
        def normal_basis(self,*a,**k):raise AssertionError('True normal forbidden in STSP/DR fitting/matching')
    return Guard()

def finite_e2(job,backend):
    engine=load(Path('E2/run_fits.py'),'fresh_e2_engine')
    m,g,r=job['method'],job['geometry'],job['repeat']
    reduced=job['profile']=='smoke';n=4096 if reduced else 500000
    cfg,lib,lookup,sample=engine.select(m,g)
    if g==GEOMETRIES[-1]:
        geometry,_,_,data,lineage=engine.original.sample(m,r,n)
    elif m=='STSP':
        geometry=lookup[g];_,_,data,lineage=sample(geometry,r,n)
    else:
        geometry=lib._geometry(cfg,g);data,lineage=sample(cfg,geometry,r,n)
    if job['experiment']=='nonuniform_E2':
        sampling=load(Path('nonuniform/E2/sampling.py'),'fresh_nonuniform_sampling')
        _,_,data,lineage=sampling.generate(geometry,m,r,n)
    query=geometry.query_grid(144) if m=='STSP' else geometry.query_grid(144,cfg['query_seed'])
    if reduced:query=query[:2]
    if m!='YY':geometry=no_normal(geometry)
    rejected=[]
    if m=='DR':
        adapter=load(Path('E2/dr_rejected_basis_failure_adapter.py'),'fresh_dr_failure_adapter')
        from dryy.dr import DRFailure
        def observe(v):rejected.append(v);return len(rejected)-1
        cpu=importlib.import_module('dryy.dr')
        cpu.solve_fiber=adapter.wrap(cpu.solve_fiber,np,DRFailure,observe)
    equivalence=None
    if backend=='gpu':
        cp,_=engine.original.require_gpu_environment()
        if m=='DR':
            gpu=importlib.import_module('dryy_gpu');gpu.solve_fiber_gpu=adapter.wrap(gpu.solve_fiber_gpu,cp,DRFailure,observe)
        equivalence=engine.gate(m,data,geometry,query,cfg,lib,cp)
    cells=[]
    for size in ((2048,4096) if reduced else SIZES):
        native=engine.cell(m,data[:size],geometry,query,cfg,lib,backend=='gpu')
        cells.append(dict(n=size,groups=engine.normalize(m,native),native=native,
            summaries={name:engine.native_metric(m,name,native) for name in engine.normalize(m,native)}))
    return dict(parent_n=n,query_coordinates=query,N_q=len(query),results=cells,lineage=lineage,
        observed_coordinates_sha256=array_sha(data),rejected_basis_records=rejected,
        equivalence=equivalence,backend=backend,sigma=.005)

def bandwidth_e3(job,backend):
    engine=load(Path('E3/run_fits.py'),'fresh_e3_engine')
    m,g,phase=job['method'],job['geometry'],job['phase']
    reduced=job['profile']=='smoke'
    lib=engine.load_library(m,phase,g)
    data,lineage=engine.observations(lib,g,job['repeat'],4096 if reduced else None)
    query,reference=engine.assets(g,phase)
    if reduced:query=query[:2];reference=reference[:128]
    geometry=engine.geometry(lib,g,m)
    equivalence=None
    if backend=='gpu':
        engine.gpu_environment();equivalence=engine.gate(lib,data,geometry,query)
    cells=engine.cells(lib,data,geometry,query,(.2,.3) if reduced else engine.GRID,backend=='gpu')
    return dict(n=len(data),sigma=.015,query_coordinates=query,query_sha256=array_sha(query),
        reference_count=len(reference),reference_sha256=array_sha(reference),cells=cells,
        lineage=lineage,equivalence=equivalence,backend=backend)

def e4(job):
    # Explicit, source-bound engineering adapter. Only input acquisition and
    # reference construction change; all matching/search/metric code is reused.
    original=load(Path('E4/correspondence/finite_experiment.py'),'fresh_e4_original')
    source=(ROOT/'E4/correspondence/finite_experiment.py').read_text('utf8')
    import ast
    tree=ast.parse(source)
    fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_repeat'))
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='start' for t in n.targets))
    # The original setup starts after loading clean/noise and the frozen y1.
    initial=ast.parse('''
clean, noise, lineage = parent_sample(repeat, 200000)
take = np.array([0,8,16,24]) if pilot else np.arange(32)
grid = query_grid(32, arc_table())
reference_start = time.perf_counter()
ref_operator = WeightedSTSP(clean + .06*noise, .12)
roots = [solve_fiber(ref_operator, x, normal_basis(t), .12)
         for x,t in zip(grid['coordinates'][take], grid['theta'][take])]
if any(f.status != 'success' for f in roots):
    raise ArithmeticError('Fresh common h1 query generation failed; no substituted queries')
y1 = np.array([f.point for f in roots]); y1.flags.writeable = False
reference_construction_seconds = time.perf_counter() - reference_start
''').body
    fn.body=initial+fn.body[start:]
    # Account for generating the reference coordinates in the common cost.
    for n in ast.walk(fn):
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='shared_seconds' for t in n.targets):
            n.value=ast.parse('setup_seconds+reference_seconds+reference_construction_seconds',mode='eval').body
    scope=dict(original.__dict__)
    from e4.experiment import parent_sample,seed_lineage
    scope['parent_sample']=parent_sample
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
        'paper_runs.worker::E4_fresh_inputs_and_accounted_reference_construction','exec'),scope)
    result=scope['run_repeat'](job['repeat'],job['profile']=='smoke')
    result.update(input_generated=True,seed_lineage=seed_lineage(job['repeat']),
        input_definition='e4.experiment.parent_sample original PCG64DXSM seeds; geometry-assisted original h1 solve on original 32 arc-length locations',
        adapter_sha256=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest())
    return result

def e5(job):
    sys.path.insert(0,str(ROOT/'E5/src'))
    from e56v2.experiment import population_cell,finite_cell,finite_seed
    reduced=job['profile']=='smoke'
    if job['phase']=='population':
        cfg=json.loads((ROOT/'E5/config/population.json').read_bytes())
        return population_cell(cfg,job['sigma'],query_count=2 if reduced else None,
            numerical=dict(latent_count=128,radial_order=32,longitudinal_order=32) if reduced else None)
    cfg=json.loads((ROOT/'E5/config/finite_sample.json').read_bytes())
    return finite_cell(cfg,job['n'],job['repeat'],finite_seed(cfg,job['n'],job['repeat']),query_count=2 if reduced else None)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--job',required=True)
    a=parser.parse_args();run=a.run.resolve();design=json.loads((run/'PLAN.json').read_bytes())
    job=next(j for j in design['jobs'] if j['id']==a.job)
    destination=run/'packets'/(a.job+'.json.gz')
    if destination.exists():raise FileExistsError(destination)
    begun=time.perf_counter()
    try:
        exp=job['experiment'];backend=design['backend']
        if exp in ('E1','nonuniform_E1'):result=population(job,backend)
        elif exp in ('E2','nonuniform_E2'):result=finite_e2(job,backend)
        elif exp=='E3':result=bandwidth_e3(job,backend)
        elif exp=='E4' and job.get('kind')=='counterexample':
            module=load(Path('E4/correspondence/counterexample.py'),'fresh_counterexample')
            module.OUT=run/'counterexample';module.main()
            result=dict(output='counterexample',new_computation=True)
        elif exp=='E4':result=e4(job)
        elif exp=='E5':result=e5(job)
        else:raise ValueError(exp)
        # GPU population uses actual same-input CPU equivalence, not a copied
        # historical PASS. These expensive full CPU checks are explicit.
        if backend=='gpu' and exp in ('E1','nonuniform_E1'):
            cpu=population(job,'cpu')
            if plain(cpu.get('failures'))!=plain(result.get('failures')):raise ArithmeticError('CPU/GPU failure sets differ')
            if job['method']=='MS-STSP':
                for x,y in zip(cpu['query_rows'],result['query_rows']):
                    if x['status']!=y['status']:raise ArithmeticError('MS CPU/GPU query status changed')
                    if x['status']=='success':
                        for field in ('points','multi'):
                            np.testing.assert_allclose(x[field],y[field],rtol=0,atol=1e-10)
            else:
                for field in ('single_points','multi_points'):
                    np.testing.assert_allclose(cpu[field],result[field],rtol=0,atol=1e-10,equal_nan=True)
            result['fresh_cpu_gpu_equivalence']=dict(status='PASS',atol=1e-10,scope='complete same-input cell')
        packet=dict(schema='PAPER_FRESH_TASK_V1',job=job,result=plain(result),
            source_manifest_sha256=design['source_manifest_sha256'],backend='cpu' if exp in ('E4','E5') else backend,requested_backend=backend,
            environment=dict(python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
                cpu_threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}),
            seconds=time.perf_counter()-begun,historical_outputs_read=False)
        destination.parent.mkdir(exist_ok=True)
        with destination.open('xb') as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:f.write(json.dumps(packet,allow_nan=True).encode())
    except Exception as error:
        folder=run/'errors';folder.mkdir(exist_ok=True)
        with (folder/(a.job+'.json')).open('x',encoding='utf8') as f:
            json.dump(dict(job=job,error=repr(error),traceback=traceback.format_exc(),outputs_preserved=True),f,indent=2)
        raise

if __name__=='__main__':main()
