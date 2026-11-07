"""Deterministic E4-curve correspondence scaling under h1=3sigma,h2=6sigma."""
import os
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[name]='1'
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq

from population_operator import PopulationSTSP,arc_table,gamma,ROOT
from continuous_cp import continuous_cp
from e4.geometry import query_grid,normal_basis,project_to_curve,curve_distances
from e4.solver import solve_fiber
from e4.config import CONFIG
from e4.coefficients import two_scale_weights

CONTROLS=dict(CONFIG['matching'],projected_residual_tolerance=1e-13,full_residual_tolerance=1e-11)
RESOLUTIONS={'production':dict(latent_count=8192,radial_order=160),
             'reference':dict(latent_count=16384,radial_order=224)}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def default(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,(np.integer,np.floating)):return x.item()
    raise TypeError(type(x).__name__)
def save(p,value):
    with p.open('x',encoding='utf8') as f:json.dump(value,f,indent=2,default=default)

def run_cell(sigma,resolution,pilot=False):
    start=time.perf_counter();settings=RESOLUTIONS[resolution]
    h1=PopulationSTSP(sigma,3*sigma,**settings);h2=PopulationSTSP(sigma,6*sigma,**settings)
    grid=query_grid(32,arc_table());ids=[0] if pilot else list(range(32))
    rows=[];failures=[];adopted=[]
    prior=ROOT/'population_pilot_attempt03'/f'sigma_{sigma:.4f}_{resolution}.json'
    old={}
    if not pilot and prior.is_file():
        packet=json.loads(prior.read_bytes())
        assert packet['algorithm_source_sha256']==sha(__file__)
        assert packet['operator_source_sha256']==sha(ROOT/'population_operator.py')
        assert packet['CP_source_sha256']==sha(ROOT/'continuous_cp.py')
        assert packet['settings']==settings and packet['controls']==CONTROLS
        old={int(r['query_id']):r for r in packet['rows'] if r['success']}
    weights=two_scale_weights(3*sigma,6*sigma,sigma)['weights']
    for j in ids:
        if j in old:
            rows.append(old[j]);adopted.append(j);continue
        x=grid['coordinates'][j];theta=float(grid['theta'][j]);B=normal_basis(theta)
        try:
            fine=solve_fiber(h1,x,B,h1.h,controls=CONTROLS)
            if fine.status!='success':raise ArithmeticError('reference:'+fine.status)
            y1=fine.point
            oracle=solve_fiber(h2,x,B,h2.h,controls=CONTROLS)
            if oracle.status!='success':raise ArithmeticError('oracle:'+oracle.status)
            estimated=h1.evaluate(y1).eigenvectors[:,:2]
            enf=solve_fiber(h2,y1,estimated,h2.h,controls=CONTROLS)
            if enf.status!='success':raise ArithmeticError('ENF:'+enf.status)
            cp=continuous_cp(h2,y1,solve_fiber,CONTROLS,grid_nodes=9,validate_normal_starts=False,stationarity_tolerance=1e-13)
            if cp['status']!='success':raise ArithmeticError('CP:'+str(cp))
            second=dict(Oracle=oracle.point,ENF=enf.point,CP=cp['point'])
            extra={k:weights[0]*y1+weights[1]*v for k,v in second.items()}
            metrics={}
            for method,z in second.items():
                metrics[method]=dict(E_match=float(np.linalg.norm(z-oracle.point)),
                    E_base=float(np.linalg.norm(project_to_curve(z)['point']-x)),
                    extrapolated_distance=float(curve_distances(extra[method][None,:])[0]))
            gate=None
            if pilot:
                refined=continuous_cp(h2,y1,solve_fiber,CONTROLS,grid_nodes=17,validate_normal_starts=False,stationarity_tolerance=1e-13)
                difference=float(np.linalg.norm(refined['point']-cp['point']))
                if refined['status']!='success' or difference>2e-10:raise ArithmeticError('population CP chart refinement failed')
                gate=dict(refined_nodes=17,difference=difference)
            rows.append(dict(query_id=j,success=True,theta=theta,true_base=x,reference=y1,
                second_scale=second,extrapolated=extra,metrics=metrics,
                reference_diagnostics=fine.diagnostics,oracle_diagnostics=oracle.diagnostics,
                ENF_diagnostics=enf.diagnostics,CP_diagnostics=cp,CP_refinement=gate))
            print('population',sigma,resolution,'query',j,'ENF',metrics['ENF']['E_match'],'CP',metrics['CP']['E_match'],flush=True)
        except Exception as exc:
            rows.append(dict(query_id=j,success=False,error=type(exc).__name__+': '+str(exc)))
            failures.append(dict(query_id=j,error=type(exc).__name__+': '+str(exc)))
    return dict(schema='P1_POPULATION_CURVE_CELL_V1',sigma=sigma,resolution=resolution,settings=settings,
        controls=CONTROLS,pilot=pilot,N_q_nominal=32,query_ids=ids,rows=rows,failures=failures,
        adopted_pilot_query_ids=adopted,pilot_source_sha256=sha(prior) if adopted else None,
        no_monte_carlo=True,weights=weights,elapsed_seconds=time.perf_counter()-start,
        operator_evaluations=dict(h1=h1.evaluation_count,h2=h2.evaluation_count),
        algorithm_source_sha256=sha(__file__),operator_source_sha256=sha(ROOT/'population_operator.py'),
        radial_source_sha256=sha(ROOT/'radial_cpu_fast.py'),CP_source_sha256=sha(ROOT/'continuous_cp.py'),
        common_reference_disclosure='As in E4, common h1 branch points are constructed on the known true normal charts. ENF and CP matching do not receive the true parameter/basis. Truth is used for Oracle and evaluation.',
        integral='Uniform-arclength latent quadrature, analytic Gaussian reduction, original K6 radial GL nodes, padding12, float64; exact half-integer Bessel evaluation.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--sigma',type=float);p.add_argument('--resolution',choices=RESOLUTIONS)
    a=p.parse_args();dest=ROOT/('population_pilot_attempt03' if a.pilot else 'population');dest.mkdir(exist_ok=True)
    sigmas=[.008,.03] if a.pilot else [.008,.01,.0125,.016,.02,.025,.03]
    if a.sigma is not None:sigmas=[a.sigma]
    resolutions=[a.resolution] if a.resolution else list(RESOLUTIONS)
    for sigma in sigmas:
        for resolution in resolutions:
            path=dest/f'sigma_{sigma:.4f}_{resolution}.json'
            if path.exists():raise ValueError('Existing valid/failed cell; no silent rerun')
            value=run_cell(sigma,resolution,a.pilot);save(path,value)
            print('SAVED',path,flush=True)
            if value['failures']:raise SystemExit(2)

if __name__=='__main__':main()
