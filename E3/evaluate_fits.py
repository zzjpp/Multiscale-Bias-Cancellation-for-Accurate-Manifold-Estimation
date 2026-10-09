"""Evaluate a COMPLETE new public E3 run; no fitting or shard-error pooling.

Input packets are from run_fits.py, not private historical schemas. Main
partial-repeat metrics, all failures, NaN values and original reliability stay.
"""
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
from scipy.spatial import cKDTree
from science_support import GRID,GROUPS,KEYS,N,assets

def coverage(rows,reference):
    cloud=np.asarray([r['point'] for r in rows if r['status']=='success'],dtype=np.float64)
    if len(cloud)==0: return dict(E_reverse_mean=float('nan'),E_reverse=float('nan'))
    distances=cKDTree(cloud).query(reference,k=1,workers=1)[0]
    return dict(E_reverse_mean=float(np.mean(distances)),E_reverse=float(np.max(distances)))

def read_complete(directory):
    packets={}
    expected_source=source_digest()
    for path in directory.glob('*.json.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as f: packet=json.load(f)
        if packet.get('schema')!='E3_PUBLIC_NEW_FIT_TASK_V1': raise ValueError('Only new public task schema accepted')
        if packet.get('source_manifest_sha256')!=expected_source: raise ValueError('New-run source contract changed or missing')
        if packet.get('gate',{}).get('status')!='PASS': raise ValueError('Missing exact-input CPU/GPU gate')
        key=tuple(packet[k] for k in ('method','phase','geometry','repeat'))
        if key in packets: raise ValueError('Duplicate task')
        packets[key]=packet
    expected={(m,p,g,r) for m in GROUPS for p in ('main','rev') for g in KEYS for r in (range(20) if p=='main' else [0])}
    if set(packets)!=expected: raise ValueError('All360main+18full-cloud reverse tasks required')
    return packets

def table(packets):
    rows=[]
    for method in GROUPS:
        for phase in ('main','rev'):
            for geometry in KEYS:
                query,reference=assets(geometry,phase)
                repeats=20 if phase=='main' else 1
                parents=[packets[method,phase,geometry,r] for r in range(repeats)]
                for parent in parents:
                    if parent['n']!=N[geometry] or parent['sigma']!=.015 or len(parent['cells'])!=28:
                        raise ValueError('Changed design or missing scale')
                    if not np.array_equal(parent['query_coordinates'],query): raise ValueError('Query coordinates changed')
                    if parent['reference_count']!=len(reference): raise ValueError('Reference identity/budget changed')
                    if tuple(c['scale'] for c in parent['cells'])!=GRID: raise ValueError('Scale order changed')
                for i,scale in enumerate(GRID):
                    for estimator in GROUPS[method]:
                        complete=0;failed=0;taxonomy=Counter();metrics=[]
                        for parent in parents:
                            cell=parent['cells'][i];outcomes=cell['groups'][estimator]
                            if len(outcomes)!=len(query): raise ValueError('Query outcome omitted')
                            failures=[r['reason'] for r in outcomes if r['status']!='success']
                            taxonomy.update(failures);failed+=len(failures);complete+=not failures
                            metrics.append(cell['summaries'][estimator] if phase=='main' else coverage(outcomes,reference))
                        rate=failed/(repeats*len(query));reliable=rate<=.01 and (phase=='rev' or complete>=18)
                        for metric in (('E_avg','E_infty') if phase=='main' else ('E_reverse_mean','E_reverse')):
                            values=np.asarray([a[metric] for a in metrics]);values=values[np.isfinite(values)]
                            q1,median,q3=np.quantile(values,[.25,.5,.75]) if len(values) else [float('nan')]*3
                            rows.append(dict(method=method,phase=phase,geometry=geometry,estimator=estimator,scale=scale,
                                metric=metric,median=median,q1=q1,q3=q3,complete_repeats=complete,total_repeats=repeats,
                                query_failure_rate=rate,reliable=reliable,query_count=len(query),reference_count=len(reference),
                                n=N[geometry],sigma=.015,query_failure_count=failed,failure_taxonomy_json=json.dumps(dict(taxonomy),sort_keys=True)))
    if len(rows)!=6720: raise ValueError('Expected6720 rows')
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();rows=table(read_complete(a.input))
    with a.output.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(rows=len(rows),reverse_reference_all=262144,partial_repeats_and_failures_retained=True)))

if __name__=='__main__': main()
