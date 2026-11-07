"""Independent high-precision distance checks and stricter shared-circle comparison."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import mpmath as mp
import numpy as np

import analyze

ROOT=Path(__file__).resolve().parent

def distance_high_precision(key, point):
    with mp.workdps(60):
        p=[mp.mpf(float(x)) for x in point]
        if key=='corrugated_torus_r3':
            a=mp.sqrt(p[0]**2+p[1]**2);b=p[2]
            _,initial=analyze.GEO._CORRUGATED_MERIDIAN.closest([float(a),float(b)])
            def parts(v):
                r=mp.mpf('.7')*(1+mp.mpf('.18')*mp.cos(3*v)+mp.mpf('.10')*mp.sin(2*v))
                rp=mp.mpf('.7')*(-mp.mpf('.54')*mp.sin(3*v)+mp.mpf('.20')*mp.cos(2*v))
                return 2+r*mp.cos(v),r*mp.sin(v),rp*mp.cos(v)-r*mp.sin(v),rp*mp.sin(v)+r*mp.cos(v)
            def gradient(v):
                rho,z,drho,dz=parts(v)
                return (rho-a)*drho+(z-b)*dz
            v=mp.findroot(gradient,(mp.mpf(initial)-mp.mpf('.0001'),mp.mpf(initial)+mp.mpf('.0001')),
                          tol=mp.mpf('1e-48'),maxsteps=80)
            rho,z,_,_=parts(v)
            return float(mp.sqrt((rho-a)**2+(z-b)**2))
        if key=='clifford_torus_r4':
            r=1/mp.sqrt(2)
            return float(mp.sqrt((mp.sqrt(sum(x*x for x in p[:2]))-r)**2+
                                 (mp.sqrt(sum(x*x for x in p[2:]))-r)**2))
        d=analyze.GEO.GEOMETRIES[key].d
        return float(mp.sqrt((mp.sqrt(sum(x*x for x in p[:d+1]))-1)**2+sum(x*x for x in p[d+1:])))

def main():
    report=json.loads((ROOT/'analysis/NUMERICAL_CHECK.json').read_bytes())
    assert report['status']=='PASS'
    with gzip.open(ROOT/'analysis/SHARED_POINTS.json.gz','rt',encoding='utf8') as f:cells=json.load(f)
    checks=[]
    for cell in cells:
        key=cell['geometry'];geometry=analyze.GEO.GEOMETRIES[key]
        for method in ['single','naive','proposed']:
            points=np.array([q['points'][method] for q in cell['queries']])
            dist=geometry.distances(points)
            # All original homogeneous queries; only each grid maximum for the
            # nonhomogeneous surface. This is a bounded independent check.
            index=int(np.argmax(dist))
            value=distance_high_precision(key,points[index])
            difference=abs(value-float(dist[index]))
            assert difference<report['point_metric_difference_budget']
            checks.append(dict(geometry=key,sigma=cell['sigma'],resolution=cell['resolution'],
                method=method,query_id=cell['queries'][index]['query_id'],
                original_distance=float(dist[index]),mp_60digit_distance=value,absolute_difference=difference))
    strict=json.loads((ROOT/'strict_circle_validation/PACKET.json').read_bytes())
    original=json.loads(analyze.packet_path('circle_r2',.008,'production').read_bytes())
    assert not strict['failures'] and strict['failure_count']==0 and strict['single_failure_count']==0
    def shared(packet):
        q=packet['query_rows'][0];p=np.array(q['fits'][0]['point'])
        delta=np.array(q['normal_basis'])@np.array(q['normal_guided']['t2'])
        w=packet['weights']['weights']
        return {'single':p+delta,'naive':p-delta/3,'proposed':p+w[1]*delta}
    strict_diffs={name:float(np.linalg.norm(shared(strict)[name]-shared(original)[name]))
                  for name in ['single','naive','proposed']}
    assert max(strict_diffs.values())<report['point_metric_difference_budget']
    payload=dict(status='PASS',independent_distance_precision_digits=60,
        independent_distance_checks=len(checks),checks=checks,
        maximum_independent_distance_difference=max(c['absolute_difference'] for c in checks),
        corrugated_scope='Production/refined grid maximum of each of three controlled methods at seven sigma; not every corrugated query',
        strict_circle=dict(shared_point_differences=strict_diffs,
            sigma=.008,latent_count=32768,radial_order=320,
            projected_tolerance=1e-13,full_tolerance=1e-11,correction_tolerance=1e-13,
            backend='CPU population expectation; no Monte Carlo',elapsed_seconds=strict['validation_elapsed_seconds']),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    analyze.write_json(ROOT/'analysis/INDEPENDENT_PRECISION.json',payload)
    print(json.dumps({k:v for k,v in payload.items() if k!='checks'},indent=2),flush=True)

if __name__=='__main__':main()
