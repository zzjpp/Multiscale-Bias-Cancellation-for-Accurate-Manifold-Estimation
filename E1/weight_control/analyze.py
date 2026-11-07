"""Controlled weight-only comparison on preserved population STSP/ENF outputs."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import quad

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT/'PLAN.json').read_bytes())
ORDER = PLAN['geometry_order']
SIGMAS = PLAN['sigma_grid']
LABELS = ['single', 'naive', 'proposed', 'ms_stsp']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_module(tag, filename):
    path = ROOT/'frozen'/tag/'src/e1_theory_path'/filename
    name = '_weight_control_'+tag+'_'+path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

GEO = load_module('six', 'geometry.py')
COEF = load_module('five', 'coefficients.py')

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)

def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def verify_inputs():
    binding = json.loads((ROOT/'SOURCE_BINDING.json').read_bytes())
    for item in binding['files']:
        path = ROOT/item['copy']
        if sha(path) != item['sha256'] or path.stat().st_size != item['size']:
            raise ValueError('Frozen input changed: '+str(path))

def independent_coefficient(d, D, h, sigma):
    q = D-d
    lam = sigma/h
    def integral(j):
        return quad(lambda s: s**(q/2+j-1)*(1-s)**(d/2+6)*np.exp(-s/(2*lam**2)),
                    0, 1, epsabs=1e-15, epsrel=2e-13, limit=400)[0]
    ratio = (1-integral(2)/integral(1))/(d+14)
    return d/2*(sigma**2+h**2*ratio)

def packet_path(key, sigma, resolution):
    tag = 'six' if key == 'corrugated_torus_r3' else 'five'
    return ROOT/'frozen'/tag/'results/packets'/f'stsp_{key}_sigma_{sigma:.6f}_{resolution}.json'

def evaluate(key, sigma, resolution):
    packet = json.loads(packet_path(key,sigma,resolution).read_bytes())
    geometry = GEO.GEOMETRIES[key]
    if packet['failures'] or packet['failure_count'] or packet['single_failure_count']:
        raise ArithmeticError('Saved failures present; preserve and report, never drop queries')
    if packet['success_count'] != packet['query_count']:
        raise ArithmeticError('Incomplete saved query grid')
    if packet['h1'] != 3*sigma or packet['h2'] != 6*sigma:
        raise ArithmeticError('Bandwidth mismatch')
    coeff = np.array([independent_coefficient(geometry.d,geometry.D,h,sigma) for h in [3*sigma,6*sigma]])
    original_coeff = np.array([r['value'] for r in packet['weights']['coefficients']])
    np.testing.assert_allclose(coeff,original_coeff,rtol=1e-11,atol=0)
    weight = np.array([coeff[1],-coeff[0]])/(coeff[1]-coeff[0])
    np.testing.assert_allclose(weight,packet['weights']['weights'],rtol=1e-11,atol=1e-13)
    # Use the exact archived proposed weights, independently checked above.
    weight = np.array(packet['weights']['weights'])
    assert abs(weight.sum()-1)<1e-11 and abs(weight@coeff)<1e-11
    points = {name:[] for name in ['single','naive','proposed','legacy_single']}
    queries = []
    correction = residual = reconstruction = legacy_difference = 0.
    for row in packet['query_rows']:
        assert not row['failure'] and len(row['fits'])==2
        assert row['normal_space_source']=='PCA'
        z1 = np.array(row['fits'][0]['point'])
        z2 = np.array(row['fits'][1]['point'])
        basis = np.array(row['normal_basis'])
        t2 = np.array(row['normal_guided']['t2'])
        delta = basis@t2
        np.testing.assert_allclose(z2,z1+delta,rtol=0,atol=3e-15)
        out = dict(single=z2, naive=z1-delta/3,
                   proposed=z1+weight[1]*delta,
                   legacy_single=np.array(row['single_point']))
        np.testing.assert_allclose(out['proposed'],row['multi_point'],rtol=0,atol=3e-15)
        np.testing.assert_allclose(out['naive'],4*z1/3-z2/3,rtol=0,atol=3e-15)
        np.testing.assert_allclose(out['proposed'],weight[0]*z1+weight[1]*z2,rtol=0,atol=3e-15)
        assert all(np.isfinite(v).all() for v in out.values())
        for name, value in out.items():points[name].append(value)
        reconstruction=max(reconstruction,float(np.linalg.norm(z2-z1-delta)))
        legacy_difference=max(legacy_difference,float(np.linalg.norm(z2-out['legacy_single'])))
        correction=max(correction,sum(abs(w)*float(fit['final']['coordinate_correction'])
                                      for w,fit in zip(weight,row['fits'])))
        residual=max(residual,*[float(fit['final']['full_residual']) for fit in row['fits']])
        queries.append(dict(query_id=row['query_id'],truth=row['truth'],
            Z_h1=z1.tolist(),Z_h2=z2.tolist(),ENF_basis=basis.tolist(),t2=t2.tolist(),
            points={k:v.tolist() for k,v in out.items()}))
    points = {k:np.asarray(v) for k,v in points.items()}
    errors={k:geometry.distances(v) for k,v in points.items()}
    for k in ['proposed','legacy_single']:
        old = 'multi' if k=='proposed' else 'single'
        np.testing.assert_allclose(errors[k],packet['metrics'][old+'_distances'],rtol=2e-6,atol=3e-14)
    metrics={name+'_E_'+metric:float(f(values)) for name,values in errors.items()
             for metric,f in [('infinity',np.max),('average',np.mean)]}
    assert len(queries)==packet['query_count']
    return dict(geometry=key,sigma=sigma,resolution=resolution,query_count=len(queries),
        weights=weight.tolist(),coefficients=coeff.tolist(),metrics=metrics,queries=queries,
        points=points,diagnostics=dict(max_final_root_correction=correction,
        max_full_residual=residual,max_ENF_reconstruction_error=reconstruction,
        max_legacy_vs_matched_single_displacement=legacy_difference))

def main():
    p=argparse.ArgumentParser();p.add_argument('--circle-only',action='store_true');a=p.parse_args()
    verify_inputs()
    selected = ORDER[:1] if a.circle_only else ORDER
    dest=ROOT/('circle_check' if a.circle_only else 'analysis')
    dest.mkdir(exist_ok=False)
    ms=list(csv.DictReader((ROOT/'frozen/ms/COMPARISON_VALUES.csv').open(encoding='utf8')))
    ms_summary=json.loads((ROOT/'frozen/ms/NUMERICAL_SUMMARY.json').read_bytes())
    assert sha(ROOT/'frozen/ms/COMPARISON_VALUES.csv')==ms_summary['table_sha256']
    rows=[];precisions=[];point_records=[];minimum=float('inf')
    weight_rows=[]
    for key in selected:
        for sigma in SIGMAS:
            prod=evaluate(key,sigma,'production');ref=evaluate(key,sigma,'reference')
            minimum=min(minimum,prod['metrics']['proposed_E_infinity'],prod['metrics']['proposed_E_average'])
            external=next(x for x in ms if x['geometry']==key and float(x['sigma'])==sigma)
            row=dict(geometry=key,sigma=sigma,query_count=prod['query_count'],**prod['metrics'])
            row.update(ms_stsp_E_infinity=float(external['ms_stsp_E_infinity']),
                       ms_stsp_E_average=float(external['ms_stsp_E_average']),
                       proposed_w1=prod['weights'][0],proposed_w2=prod['weights'][1],
                       naive_w1=4/3,naive_w2=-1/3,
                       naive_over_sigma2=row['naive_E_infinity']/sigma**2,
                       proposed_over_sigma4=row['proposed_E_infinity']/sigma**4)
            rows.append(row)
            for label in ['single','naive','proposed']:
                pdiff=float(np.max(np.linalg.norm(prod['points'][label]-ref['points'][label],axis=1)))
                mdiff=max(abs(prod['metrics'][label+'_E_'+m]-ref['metrics'][label+'_E_'+m]) for m in ['infinity','average'])
                precisions.append(dict(geometry=key,sigma=sigma,method=label,
                    production_reference_point_difference=pdiff,metric_difference=mdiff,
                    root_correction_diagnostic=prod['diagnostics']['max_final_root_correction'],
                    max_full_residual=prod['diagnostics']['max_full_residual']))
            for result in [prod,ref]:
                point_records.append({k:v for k,v in result.items() if k!='points'})
            print(key,sigma,'done',flush=True)
        block=[x for x in rows if x['geometry']==key]
        spread=max(x['proposed_w1'] for x in block)-min(x['proposed_w1'] for x in block)
        assert spread<1e-11
        geo=GEO.GEOMETRIES[key]
        weight_rows.append(dict(geometry=key,d=geo.d,D=geo.D,naive_w1=4/3,naive_w2=-1/3,
            proposed_w1=block[0]['proposed_w1'],proposed_w2=block[0]['proposed_w2'],max_w1_sigma_variation=spread))
    slopes=[]
    for key in selected:
        block=[x for x in rows if x['geometry']==key]
        for method in LABELS:
            slopes.append(dict(geometry=key,method=method,
                slope_all_7=float(np.polyfit(np.log(SIGMAS),np.log([x[method+'_E_infinity'] for x in block]),1)[0]),
                slope_smallest_4=float(np.polyfit(np.log(SIGMAS[:4]),np.log([x[method+'_E_infinity'] for x in block[:4]]),1)[0])))
    # Independent 70-digit one-dimensional coefficient check on the unit circle.
    import mpmath as mp
    with mp.workdps(70):
        def b(k):
            lam=mp.mpf(1)/k
            def I(j):return mp.quad(lambda s:s**(mp.mpf(1)/2+j-1)*(1-s)**mp.mpf('6.5')*mp.exp(-s/(2*lam**2)),[0,mp.mpf('.1'),mp.mpf('.5'),1])
            return (1+k*k*(1-I(2)/I(1))/15)/2
        b1,b2=b(3),b(6);gamma=abs((4*b1-b2)/3)
        high_precision=dict(b1=str(b1),b2=str(b2),Gamma_1_2=str(gamma),
            proposed_w1=str(b2/(b2-b1)),proposed_w2=str(-b1/(b2-b1)),digits=70)
    circle=[dict(sigma=x['sigma'],naive_E_infinity=x['naive_E_infinity'],
        naive_over_sigma2=x['naive_over_sigma2'],Gamma_1_2=float(gamma),
        relative_difference_from_Gamma=abs(x['naive_over_sigma2']/float(gamma)-1),
        proposed_over_sigma4=x['proposed_over_sigma4']) for x in rows if x['geometry']=='circle_r2']
    point_budget=minimum*.001;root_budget=minimum*.1
    failures=[]
    for x in precisions:
        if max(x['production_reference_point_difference'],x['metric_difference'])>point_budget:
            failures.append(dict(reason='production_reference_difference',**x))
        if x['root_correction_diagnostic']>root_budget:failures.append(dict(reason='root_correction',**x))
    report=dict(status='PASS' if not failures else 'FAIL',purpose='New stored-fit weight-only evaluation; not independent fitting reproduction',
        rows=len(rows),reused_full_STSP_packets=len(point_records),new_main_fits=0,monte_carlo=False,
        minimum_proposed_error=minimum,point_metric_difference_budget=point_budget,root_correction_budget=root_budget,
        maximum_resolution_point_difference=max(x['production_reference_point_difference'] for x in precisions),
        maximum_metric_difference=max(x['metric_difference'] for x in precisions),
        maximum_root_correction=max(x['root_correction_diagnostic'] for x in precisions),
        high_precision_circle=high_precision,failures=failures,
        claims_not_used_to_drop_or_adjust_results=True,
        diagnostic_caveat='Resolution differences and root corrections are diagnostics, not certified continuous error bounds',
        single_definition=PLAN['single_definition'],analyzer_sha256=sha(__file__))
    write_csv(dest/'VALUES.csv',rows);write_csv(dest/'SLOPES.csv',slopes)
    write_csv(dest/'WEIGHTS.csv',weight_rows);write_csv(dest/'PRECISION.csv',precisions)
    write_csv(dest/'CIRCLE_CONSTANT.csv',circle);write_json(dest/'NUMERICAL_CHECK.json',report)
    with gzip.open(dest/'SHARED_POINTS.json.gz','wt',encoding='utf8') as f:json.dump(point_records,f,allow_nan=False)
    print(json.dumps(report,indent=2),flush=True)
    if failures:raise SystemExit(2)

if __name__=='__main__':main()
