"""Bounded actual CPU/GPU fit checks; not the full population scientific audit."""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import study


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',type=int,default=0)
    args=parser.parse_args()
    import cupy as cp
    from e1_theory_path.gpu_population import radial_mass_gpu as original_radial
    from fast_gpu import radial_mass_gpu as fast_radial
    cp.cuda.Device(args.device).use()
    checks=[]
    for order in (160,224):
        for df in (2,3,4,5,6,7,8,10,12,14):
            for kappa in study.CONFIG['ms_radii_over_sigma']:
                radius=np.r_[0.,1e-12,.001,.1,.5,1.,np.linspace(2.,18.,97)]
                cpu=study.radial_mass(df,radius,kappa,order,power=0)
                old=cp.asnumpy(original_radial(df,cp.asarray(radius),kappa,order,power=0))
                new=cp.asnumpy(fast_radial(df,cp.asarray(radius),kappa,order,power=0))
                difference=max(float(np.max(np.abs(cpu-new))),float(np.max(np.abs(old-new))))
                if difference>2e-12:raise ArithmeticError('Radial-engine equivalence failed')
                checks.append(dict(df=df,order=order,kappa=kappa,max_absolute_difference=difference))
    cases=[]
    for key in study.CONFIG['geometry_order']:
        for resolution in ('production','reference'):
            begun=time.monotonic()
            ids=[0] if key=='corrugated_torus_r3' else None
            cpu=study.run_cell(key,.008,resolution,'cpu',args.device,query_ids=ids)
            gpu=study.run_cell(key,.008,resolution,'gpu',args.device,query_ids=ids)
            if not cpu['all_queries_successful'] or not gpu['all_queries_successful']:
                raise ArithmeticError('Actual fit gate failure: '+key+':'+resolution)
            a=np.array([r['points'] for r in cpu['query_rows']]);b=np.array([r['points'] for r in gpu['query_rows']])
            difference=float(np.max(np.linalg.norm(a-b,axis=-1)))
            if difference>study.CONFIG['cpu_gpu_fitted_point_max']:
                raise ArithmeticError('CPU/GPU fitted-point gate failed')
            cases.append(dict(geometry=key,sigma=.008,resolution=resolution,query_ids=ids or [0],
                              max_branch_point_difference=difference,
                              multi_point_difference=float(np.max(np.linalg.norm(np.array([q['multi'] for q in cpu['query_rows']])-np.array([q['multi'] for q in gpu['query_rows']]),axis=-1))),
                              cpu_seconds=cpu['seconds'],gpu_seconds=gpu['seconds'],total_seconds=time.monotonic()-begun,
                              original_latent_resolution=gpu['latent_resolution'],radial_order=gpu['radial_order']))
            print('ACTUAL_GATE_CASE '+json.dumps(cases[-1]),flush=True)
            # Preserve actual selected fit outputs, not only a PASS boolean.
            study.write_fresh(study.ROOT/'results/gates'/f'{key}_{resolution}_cpu.json.gz',cpu)
            study.write_fresh(study.ROOT/'results/gates'/f'{key}_{resolution}_gpu.json.gz',gpu)
    result=dict(status='PASS',scope='60 radial regimes and twelve actual selected CPU/GPU fit cases; not full-grid science',
                cpu_gpu_cases=cases,radial_checks=checks,config_sha256=study.sha(study.ROOT/'config.json'),
                study_source_sha256=study.sha(study.ROOT/'study.py'),fast_gpu_sha256=study.sha(study.ROOT/'fast_gpu.py'),
                frozen_manifest_sha256=study.sha(study.ROOT/'FROZEN_INPUTS.json'),
                environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,cupy=cp.__version__,
                                 gpu=cp.cuda.runtime.getDeviceProperties(args.device)['name'].decode(),device=args.device),
                original_package_modified=False)
    study.write_fresh(study.ROOT/'CPU_GPU_GATE.json',result)
    print(json.dumps(dict(status='PASS',radial_regimes=len(checks),fit_cases=len(cases))),flush=True)


if __name__=='__main__':main()
