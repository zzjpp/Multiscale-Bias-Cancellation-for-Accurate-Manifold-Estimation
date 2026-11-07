"""Actual new-density CPU/GPU checks before the formal GPU queue."""
import json
import platform
import sys
import time
import numpy as np
import scipy
import study


def main():
    import cupy as cp
    from e1_theory_path.population import radial_mass
    from e1_theory_path.gpu_population import radial_mass_gpu as old_radial
    from fast_gpu import radial_mass_gpu
    cp.cuda.Device(0).use()
    radial_checks=[]
    for order in (160,224):
        for df in (2,3,4,5,6,7,8,10,12,14):
            for kappa in (3.,6.):
                radius=np.r_[0.,1e-12,.001,.1,.5,1.,np.linspace(2.,18.,65)]
                cpu=radial_mass(df,radius,kappa,order,power=6)
                old=cp.asnumpy(old_radial(df,cp.asarray(radius),kappa,order,power=6))
                new=cp.asnumpy(radial_mass_gpu(df,cp.asarray(radius),kappa,order,power=6))
                delta=max(float(np.max(np.abs(cpu-new))),float(np.max(np.abs(old-new))))
                if delta>2e-12:raise ArithmeticError('Radial equivalence failure')
                radial_checks.append(dict(order=order,df=df,kappa=kappa,max_difference=delta))
    cases=[]
    for key in study.CONFIG['geometry_order']:
        ids=[11 if key.startswith('sphere') else 37 if key=='corrugated_torus_r3' else 17]
        for resolution in ('production','reference'):
            begin=time.monotonic()
            cpu=study.run_cell(key,.008,resolution,'cpu',query_ids=ids)
            gpu=study.run_cell(key,.008,resolution,'gpu_cupy',query_ids=ids)
            for label,result in [('cpu',cpu),('gpu',gpu)]:
                study.write_fresh(study.ROOT/'results/gates'/f'{key}_{resolution}_{label}.json.gz',result)
                if result['failure_count'] or result['single_failure_count']:
                    raise ArithmeticError('Actual density fit gate failed: '+key+':'+resolution+':'+label+':'+str(result['failures']))
            difference=max(float(np.max(np.linalg.norm(np.asarray(cpu[f])-np.asarray(gpu[f]),axis=1)))
                           for f in ('single_points','multi_points'))
            if difference>1e-10:raise ArithmeticError('CPU/GPU point mismatch')
            item=dict(geometry=key,sigma=.008,resolution=resolution,query_ids=ids,
                      point_difference=difference,seconds=time.monotonic()-begin)
            cases.append(item);print('ACTUAL_GATE_CASE '+json.dumps(item),flush=True)
    result=dict(status='PASS',scope='40 radial regimes and12 actual nonstationary-query full-resolution CPU/GPU fit cases; not full-grid science',
                source_hashes=study.source_hashes(),radial_checks=radial_checks,cpu_gpu_cases=cases,
                environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__,cupy=cp.__version__,gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()),
                precision='float64',device=0)
    study.write_fresh(study.ROOT/'CPU_GPU_GATE.json',result)
    print(json.dumps(dict(status='PASS',radial_checks=len(radial_checks),fit_cases=len(cases))),flush=True)


if __name__=='__main__':main()
