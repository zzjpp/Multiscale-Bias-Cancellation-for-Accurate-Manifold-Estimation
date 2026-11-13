"""Aggregate a complete NEW E2 run; original partial-repeat metrics retained."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from release_contract import source_digest

KEYS=('circle_r2','sphere2_r3','clifford_torus_r4','sphere3_r4','sphere2_r10','corrugated_torus_r3')
SIZES=(2000,5000,10000,20000,50000,100000,200000,500000)
GROUPS={'STSP':('Single_h1','Single_h2','Multi'),'DR':('Single_h1','Single_h2','Multi'),'YY':('Single','Multi')}

def native_metric(method,estimator,native):
    if method=='STSP':
        name={'Single_h1':'Single_h1_metrics','Single_h2':'Single_h2_metrics','Multi':'ENF_Multi_metrics'}[estimator]
    elif method=='DR' and estimator=='Single_h2':
        name='Single'  # Unchanged DR CPU/GPU native coarse-baseline name.
    else:
        name=estimator
    result=native[name]
    if not {'E_avg','E_infty'}.issubset(result): raise ValueError('Native metric fields missing')
    return result

def table(directory):
    packets={}
    expected_source=source_digest()
    for path in directory.glob('*.json.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as f: p=json.load(f)
        if p.get('schema')!='E2_PUBLIC_NEW_NESTED_PARENT_V1': raise ValueError('Unknown new-run schema')
        if p.get('source_manifest_sha256')!=expected_source: raise ValueError('New-run source contract changed or missing')
        key=(p['method'],p['geometry'],p['repeat'])
        if key in packets: raise ValueError('Duplicate parent')
        if (p['sigma'],p['N_q'],p['parent_n'])!=(.005,144,500000) or p['gate']['status']!='PASS':
            raise ValueError('Changed design or missing equivalence gate')
        if tuple(c['n'] for c in p['results'])!=SIZES: raise ValueError('Missing nested prefix')
        packets[key]=p
    expected={(m,g,r) for m in GROUPS for g in KEYS for r in range(20)}
    if set(packets)!=expected: raise ValueError('All360 nested parents required')
    rows=[]
    for method in GROUPS:
        for geometry in KEYS:
            for i,n in enumerate(SIZES):
                for estimator in GROUPS[method]:
                    summaries=[];taxonomy=Counter();failed=0;complete=0
                    for r in range(20):
                        cell=packets[method,geometry,r]['results'][i]
                        outcomes=cell['groups'][estimator]
                        if len(outcomes)!=144: raise ValueError('Query outcomes missing')
                        failures=[o['reason'] for o in outcomes if o['status']!='success']
                        taxonomy.update(failures);failed+=len(failures);complete+=not failures
                        native=cell['native']
                        summaries.append(native_metric(method,estimator,native))
                    for metric in ('E_avg','E_infty'):
                        values=np.asarray([v[metric] for v in summaries]);values=values[np.isfinite(values)]
                        q1,median,q3=np.quantile(values,[.25,.5,.75]) if len(values) else [float('nan')]*3
                        rows.append(dict(method=method,geometry=geometry,scale=float(n),parent_scale=float(n),estimator=estimator,
                            metric=metric,median=median,q1=q1,q3=q3,finite_repeat_count=len(values),complete_repeats=complete,
                            query_count_per_repeat=144,query_failure_rate=failed/2880,reliable=complete>=18 and failed/2880<=.01,
                            failure_taxonomy=json.dumps(dict(taxonomy),sort_keys=True)))
    if len(rows)!=768: raise ValueError('Expected768 summary rows')
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=table(a.input)
    with a.output.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(rows=768,independent_fine_baselines_retained=True)))

if __name__=='__main__': main()
