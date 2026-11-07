"""Fresh, same-backend E4 accuracy/cost experiment with continuous CP."""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[name]='1'
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from continuous_cp import continuous_cp

ROOT=Path(__file__).resolve().parent
FROZEN=ROOT/'frozen/E4/fitting'
sys.path.insert(0,str(FROZEN/'original/src'))
from e4.operator import WeightedSTSP
from e4.solver import solve_fiber
from e4.config import CONFIG
from e4.coefficients import two_scale_weights
from e4 import pipeline
from e4.geometry import query_grid,arc_table,gamma,normal_basis,project_to_curve,curve_distances
spec=importlib.util.spec_from_file_location('_p1_original_scan',FROZEN/'original/run_density_scan.py')
scan=importlib.util.module_from_spec(spec);spec.loader.exec_module(scan)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def default(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,(np.integer,np.floating)):return x.item()
    raise TypeError(type(x).__name__)
def save(path,value):
    with path.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,default=default)

def run_repeat(repeat,pilot=False):
    original=json.loads((FROZEN/'FROZEN_INPUTS.json').read_bytes())['repeats'][str(repeat)]
    data_file=FROZEN/'data'/f'repeat_{repeat:02d}.npz'
    query_file=FROZEN/'queries'/f'repeat_{repeat:02d}.npy'
    assert sha(data_file)==original['data_file_sha256'] and sha(query_file)==original['query_file_sha256']
    take=np.array([0,8,16,24]) if pilot else np.arange(32)
    with np.load(data_file,allow_pickle=False) as f:clean=f['clean'];noise=f['standard_noise']
    assert clean.shape==noise.shape==(200000,3)
    y1=np.load(query_file,allow_pickle=False)[take].copy();y1.flags.writeable=False
    start=time.perf_counter();observations=clean+.06*noise
    h1=WeightedSTSP(observations,.12);h2=WeightedSTSP(observations,.24)
    weights=two_scale_weights(.12,.24,.06)['weights']
    setup_seconds=time.perf_counter()-start
    grid=query_grid(32,arc_table());theta=grid['theta'][take];truth=grid['coordinates'][take]
    start=time.perf_counter();reference_checks=[]
    for p,t,x in zip(y1,theta,truth):
        fit=solve_fiber(h1,x,normal_basis(t),h1.h)
        if fit.status!='success':raise ArithmeticError('common_reference:'+fit.status)
        difference=float(np.linalg.norm(fit.point-p))
        if difference>2e-8:raise ArithmeticError('Archived h1 query inconsistent')
        residual=float(np.linalg.norm(h1.evaluate(p).residual))
        if residual>1e-9:raise ArithmeticError('Archived h1 query not residual root')
        reference_checks.append(dict(difference=difference,full_residual=residual))
    reference_seconds=time.perf_counter()-start
    shared_seconds=setup_seconds+reference_seconds
    # These truth operations are evaluation/Oracle inputs, excluded from all
    # practical method stopwatches and never passed to continuous_cp.
    projected=[project_to_curve(p) for p in y1]
    xstar=np.array([p['point'] for p in projected]);theta_star=np.array([p['theta'] for p in projected])
    results={};timing={};diagnostics={}
    order=list(np.roll(np.array(['Oracle','ENF','CP']),-(repeat-10)%3))
    for method in order:
        start=time.perf_counter();points=[];diag=[]
        for j,p in enumerate(y1):
            if method=='Oracle':
                fit=solve_fiber(h2,xstar[j],normal_basis(theta_star[j]),h2.h)
                points.append(fit.point if fit.status=='success' else np.full(3,np.nan))
                diag.append(dict(status=fit.status,**fit.diagnostics))
            elif method=='ENF':
                B,_,info=pipeline.estimate_normal(h1,p)
                fit=solve_fiber(h2,p,B,h2.h)
                points.append(fit.point if fit.status=='success' else np.full(3,np.nan))
                diag.append(dict(status=fit.status,PCA=info,**fit.diagnostics))
            else:
                fit=continuous_cp(h2,p,solve_fiber,CONFIG['matching'],grid_nodes=9)
                points.append(fit['point']);diag.append(fit)
        results[method]=np.array(points);diagnostics[method]=diag
        timing[method]=dict(correspondence_seconds=time.perf_counter()-start)
        print('repeat',repeat,method,timing[method]['correspondence_seconds'],'s',flush=True)
    # Pilot gate checks grid refinement on all four selected actual n200000
    # queries. The refined run is not mixed into primary timing or outputs.
    grid_gate=[]
    if pilot:
        for j,p in enumerate(y1):
            fine=continuous_cp(h2,p,solve_fiber,CONFIG['matching'],grid_nodes=81)
            difference=float(np.linalg.norm(fine['point']-results['CP'][j]))
            grid_gate.append(dict(query_id=int(take[j]),status=fine['status'],point_difference=difference))
            if fine['status']!='success' or difference>2e-8:raise ArithmeticError('CP scalar-grid refinement gate')
    counts=[32,128] if pilot else scan.SCAN['candidate_counts']
    nn={};candidate_cache={};cumulative_generation_seconds=0.
    for count in sorted(counts):
        count=int(count);start=time.perf_counter()
        positions=scan.nested_candidate_positions(count)
        missing=[float(u) for u in positions if float(u).hex() not in candidate_cache]
        added=pipeline.build_candidate_library_at_positions(h2,missing)
        for i,u in enumerate(missing):
            candidate_cache[u.hex()]=(added['candidate_fitted_coordinates'][i],added['candidate_status'][i])
        coordinates=np.array([candidate_cache[float(u).hex()][0] for u in positions])
        statuses=[candidate_cache[float(u).hex()][1] for u in positions]
        success=np.array([s=='success' for s in statuses])
        library=dict(candidate_fitted_coordinates=coordinates,candidate_status=statuses,
            success_mask=success,failure_count=int((~success).sum()))
        cumulative_generation_seconds+=time.perf_counter()-start
        lookup_start=time.perf_counter()
        matching=pipeline.match_nn(y1,library)
        lookup_seconds=time.perf_counter()-lookup_start
        elapsed=cumulative_generation_seconds+lookup_seconds
        nn[count]=dict(points=matching['points'],candidate_failure_count=library['failure_count'],
            candidate_status=library['candidate_status'],candidate_fitted_coordinates=library['candidate_fitted_coordinates'],
            correspondence_seconds=elapsed,selected_candidate_id=matching['selected_candidate_id'],
            cumulative_actual_generation_seconds=cumulative_generation_seconds,
            lookup_seconds=lookup_seconds,new_candidate_count=len(missing))
        print('repeat',repeat,'NN',count,elapsed,'s',flush=True)
    summaries=[];perquery=[]
    for method,count in [('Oracle',None),('ENF',None),('CP',None)]+[('NN',int(m)) for m in counts]:
        second=nn[count]['points'] if method=='NN' else results[method]
        start=time.perf_counter();extrapolated=weights[0]*y1+weights[1]*second;combine=time.perf_counter()-start
        good=np.isfinite(second).all(axis=1)
        candidate_failures=nn[count]['candidate_failure_count'] if method=='NN' else 0
        failure_count=int((~good).sum());complete=failure_count==0 and candidate_failures==0
        metrics={k:float('nan') for k in ['E_infinity','E_average','E_match','E_base']}
        if good.any():
            err=curve_distances(extrapolated[good])
            dist=np.linalg.norm(second[good]-results['Oracle'][good],axis=1)
            proj=np.array([project_to_curve(p)['point'] for p in second[good]])
            base=np.linalg.norm(proj-xstar[good],axis=1)
            metrics=dict(E_infinity=float(err.max()),E_average=float(err.mean()),E_match=float(dist.max()),E_base=float(base.max()))
        seconds=nn[count]['correspondence_seconds'] if method=='NN' else timing[method]['correspondence_seconds']
        summaries.append(dict(repeat=repeat,method=method,candidate_count=count,**metrics,
            shared_seconds=shared_seconds,correspondence_seconds=seconds,combine_seconds=combine,
            total_seconds=shared_seconds+seconds+combine,failure_count=failure_count,
            failure_rate=failure_count/len(y1),candidate_failure_count=candidate_failures,complete=complete))
        for j in range(len(y1)):
            perquery.append(dict(method=method,candidate_count=count,query_id=int(take[j]),reference=y1[j],
                second_scale=second[j],oracle_second_scale=results['Oracle'][j],true_base=xstar[j],
                extrapolated=extrapolated[j],success=bool(good[j])))
    return dict(schema='P1_E4_CONTINUOUS_CP_ACTUAL_V1',repeat=repeat,pilot=pilot,n=200000,N_q_nominal=32,
        actual_query_count=len(y1),backend='numpy-scipy CPU float64',CPU_threads=1,
        method_order=order,weights=weights,setup_seconds=setup_seconds,reference_seconds=reference_seconds,
        reference_checks=reference_checks,summary=summaries,per_query=perquery,
        method_diagnostics=diagnostics,CP_grid_refinement_gate=grid_gate,
        NN_libraries=nn,CPU=platform.processor(),python=platform.python_version(),
        CP_truth_input=False,known_geometry_disclosure='Legacy common h1 reference branch construction and discrete NN candidate initializers retain their original controlled true charts. CP constructs its local target chart solely from empirical STSP PCA and residuals.',
        timing_scope='Fresh serial same-backend measurement; data array construction, both operator indices, coefficients, reference construction, target curve construction/search and combination counted. NN uses exact nested candidate reuse: cumulative measured generation of all M candidates plus current lookup, never free precomputation. Input file I/O, truth metrics, diagnostics/plotting/packet I/O excluded. No comparison with archived GPU times.',
        cp_source_sha256=sha(ROOT/'continuous_cp.py'),source_sha256=sha(__file__))

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--repeat',type=int)
    a=p.parse_args();folder=ROOT/('finite_pilot' if a.pilot else 'finite_sample')
    folder.mkdir(exist_ok=True)
    repeats=[10] if a.pilot else ([a.repeat] if a.repeat is not None else list(range(10,20)))
    for repeat in repeats:
        output=folder/f'repeat_{repeat:02d}.json'
        if output.exists():raise ValueError('Existing result; no silent rerun')
        packet=run_repeat(repeat,a.pilot)
        save(output,packet)
        print('SAVED',output,flush=True)

if __name__=='__main__':main()
