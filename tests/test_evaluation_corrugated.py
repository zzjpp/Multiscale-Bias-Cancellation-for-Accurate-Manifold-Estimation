"""Real corrugated/reverse CPU checks and explicit synthetic interface fixtures."""
import copy
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
ENV=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')

def invoke(exp,*args):
    return subprocess.run([sys.executable,'-X','utf8','-B',str(ROOT/exp/'run_fits.py'),*args],
        cwd=ROOT,env=ENV,capture_output=True,text=True,encoding='utf-8',timeout=180)

@pytest.mark.parametrize('exp',['E2','E3'])
@pytest.mark.parametrize('method',['STSP','DR','YY'])
def test_corrugated_actual_cpu_smoke(exp,method):
    args=['--method',method,'--geometry','corrugated_torus_r3','--smoke']
    if exp=='E2' and method in ('STSP','YY'):
        args+=['--smoke-n','24000']
    result=invoke(exp,*args)
    assert result.returncode==0,(result.stdout,result.stderr)
    assert 'smoke' in result.stdout.lower()

@pytest.mark.parametrize('method',['STSP','DR','YY'])
@pytest.mark.parametrize('geometry',['circle_r2','corrugated_torus_r3'])
def test_reverse_original_library_cpu_smoke(method,geometry):
    result=invoke('E3','--method',method,'--geometry',geometry,'--phase','rev','--smoke')
    assert result.returncode==0,(result.stdout,result.stderr)

def load_evaluator(exp):
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/exp))
    spec=importlib.util.spec_from_file_location('evaluation_'+exp,ROOT/exp/'evaluate_fits.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_e2_synthetic_complete_aggregation_keeps_failed_multi(tmp_path):
    # Engineering fixture, not an experiment or CPU/GPU equivalence evidence.
    module=load_evaluator('E2')
    success=[dict(status='success',reason=None,point=[0.,0.]) for _ in range(144)]
    failure=[dict(status='failed',reason='fixture_support',point=None) for _ in range(144)]
    metric=dict(E_avg=.05,E_infty=.1)
    digest=module.source_digest()
    for method in module.GROUPS:
        for geometry in module.KEYS:
            results=[]
            for n in module.SIZES:
                groups={name:success for name in module.GROUPS[method]}
                names={'Single_h1':'Single_h1_metrics','Single_h2':'Single_h2_metrics','Multi':'ENF_Multi_metrics'} if method=='STSP' else {
                    name:('Single' if method=='DR' and name=='Single_h2' else name) for name in groups}
                native={names[name]:metric for name in groups}
                if geometry=='circle_r2' and n==2000:
                    groups['Multi']=failure
                    native[names['Multi']]=dict(E_avg=float('nan'),E_infty=float('nan'))
                results.append(dict(n=n,groups=groups,native=native))
            for repeat in range(20):
                packet=dict(schema='E2_PUBLIC_NEW_NESTED_PARENT_V1',method=method,geometry=geometry,
                    repeat=repeat,sigma=.005,N_q=144,parent_n=500000,gate=dict(status='PASS'),
                    source_manifest_sha256=digest,results=results)
                with gzip.open(tmp_path/f'{method}_{geometry}_{repeat}.json.gz','wt',encoding='utf-8') as stream:
                    json.dump(packet,stream,allow_nan=True)
    rows=module.table(tmp_path)
    assert len(rows)==768
    selected=[r for r in rows if r['geometry']=='circle_r2' and r['scale']==2000]
    for row in selected:
        if row['estimator']=='Multi':
            assert np.isnan(row['median']) and not row['reliable'] and row['finite_repeat_count']==0
            assert json.loads(row['failure_taxonomy'])=={'fixture_support':2880}
        else:
            assert row['reliable'] and np.isfinite(row['median'])

def test_e3_synthetic_complete_aggregation_and_full_cloud(monkeypatch):
    # Two-query/two-reference test interface only; production assets stay frozen.
    module=load_evaluator('E3')
    query=np.array([[0.,0.],[4.,0.]])
    reference=np.array([[1.,0.],[3.,0.]])
    monkeypatch.setattr(module,'assets',lambda geometry,phase:(query,reference))
    success=[dict(status='success',reason=None,point=q.tolist()) for q in query]
    assert module.coverage(success,reference)==dict(E_reverse_mean=1.,E_reverse=1.)
    assert np.isnan(module.coverage([],reference)['E_reverse'])
    packets={}
    for method in module.GROUPS:
        for phase in ('main','rev'):
            for geometry in module.KEYS:
                cells=[dict(scale=scale,groups={g:copy.deepcopy(success) for g in module.GROUPS[method]},
                    summaries={g:dict(E_avg=.05,E_infty=.1) for g in module.GROUPS[method]}) for scale in module.GRID]
                for repeat in (range(20) if phase=='main' else [0]):
                    packets[method,phase,geometry,repeat]=dict(n=module.N[geometry],sigma=.015,
                        cells=copy.deepcopy(cells),query_coordinates=query.tolist(),reference_count=2)
    target=packets['YY','main','circle_r2',0]['cells'][0]
    target['groups']['Multi'][0]=dict(status='failed',reason='fixture_geometry',point=None)
    rows=module.table(packets)
    assert len(rows)==6720
    row=next(r for r in rows if (r['method'],r['phase'],r['geometry'],r['scale'],r['estimator'],r['metric'])==
        ('YY','main','circle_r2',.06,'Multi','E_avg'))
    assert row['complete_repeats']==19 and row['query_failure_count']==1
    assert row['query_failure_rate']==.025 and not row['reliable'] and row['median']==.05
    assert json.loads(row['failure_taxonomy_json'])=={'fixture_geometry':1}
    assert all(r['median']==1. for r in rows if r['phase']=='rev')

@pytest.mark.parametrize('exp',['E2','E3'])
def test_wrong_new_run_source_contract_rejected(exp,tmp_path):
    module=load_evaluator(exp)
    packet=dict(schema='E2_PUBLIC_NEW_NESTED_PARENT_V1' if exp=='E2' else 'E3_PUBLIC_NEW_FIT_TASK_V1',
        source_manifest_sha256='0'*64)
    with gzip.open(tmp_path/'changed.json.gz','wt',encoding='utf-8') as stream:
        json.dump(packet,stream)
    with pytest.raises(ValueError,match='source contract'):
        (module.table if exp=='E2' else module.read_complete)(tmp_path)
