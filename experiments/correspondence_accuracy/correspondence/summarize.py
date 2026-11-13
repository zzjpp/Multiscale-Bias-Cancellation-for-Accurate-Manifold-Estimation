"""Numerical/table checks for the completed P1 experiments; no fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,value):
    with p.open('x',encoding='utf8') as f:json.dump(value,f,indent=2)
def write_csv(p,rows):
    with p.open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def finite():
    packets=[ROOT/'finite_sample'/f'repeat_{i:02d}.json' for i in range(10,20)]
    if not all(p.is_file() for p in packets):raise ValueError('Ten complete finite packets required')
    dest=ROOT/'finite_summary';dest.mkdir(exist_ok=False)
    rows=[];checks=[];cp_candidates=[]
    for path in packets:
        d=json.loads(path.read_bytes());assert not d['pilot'] and d['actual_query_count']==32
        assert d['source_sha256']==sha(ROOT/'finite_experiment.py')
        assert d['cp_source_sha256']==sha(ROOT/'continuous_cp.py')
        fixed=np.load(ROOT/'frozen/E4/fitting/queries'/f"repeat_{d['repeat']:02d}.npy")
        for q in d['per_query']:assert np.array_equal(q['reference'],fixed[q['query_id']])
        oracle={q['query_id']:np.array(q['second_scale']) for q in d['per_query'] if q['method']=='Oracle'}
        enf={q['query_id']:np.array(q['second_scale']) for q in d['per_query'] if q['method']=='ENF'}
        n512=np.array(d['NN_libraries']['512']['candidate_fitted_coordinates'])
        for index,c in enumerate(d['method_diagnostics']['CP']):
            if c['status']!='success':continue
            assert c['full_residual']<=1e-9 and c['projected_residual']<=1e-11
            assert c['normalized_stationarity']<=1e-10
            p=np.array(c['point']);ref=fixed[index]
            dcp=np.linalg.norm(p-ref);denf=np.linalg.norm(enf[index]-ref)
            dense_nn=np.linalg.norm(n512-ref,axis=1).min()
            assert dcp<=denf+2e-8 and dcp<=dense_nn+2e-8
            nearest_candidate=float(np.linalg.norm(n512-p,axis=1).min())
            cp_candidates.append(nearest_candidate)
            checks.append(dict(repeat=d['repeat'],query_id=index,
                CP_reference_distance=float(dcp),ENF_reference_distance=float(denf),
                NN512_reference_distance=float(dense_nn),CP_to_nearest_candidate=nearest_candidate,
                normalized_stationarity=c['normalized_stationarity']))
        for record in d['summary']:
            subset=[q for q in d['per_query'] if q['method']==record['method'] and q['candidate_count']==record['candidate_count']]
            assert len(subset)==32 and sorted(q['query_id'] for q in subset)==list(range(32))
            valid=[q for q in subset if q['success']]
            assert len(valid)==32-record['failure_count']
            if valid:
                match=max(float(np.linalg.norm(np.array(q['second_scale'])-oracle[q['query_id']])) for q in valid)
                assert abs(match-record['E_match'])<1e-14
            rows.append(record)
    stats=[]
    groups=sorted(set((x['method'],x['candidate_count']) for x in rows),key=lambda x:(x[0],x[1] or 0))
    for method,count in groups:
        group=[x for x in rows if x['method']==method and x['candidate_count']==count]
        assert len(group)==10
        record=dict(method=method,candidate_count=count,repeat_count=10,
                    complete_repeats=sum(x['complete'] for x in group),
                    total_query_failures=sum(x['failure_count'] for x in group),
                    total_candidate_failures=sum(x['candidate_failure_count'] for x in group))
        for field in ['E_infinity','E_average','E_match','E_base','shared_seconds','correspondence_seconds','total_seconds','failure_rate']:
            selected=[x[field] for x in group if np.isfinite(x[field]) and (x['complete'] or field in ['failure_rate','shared_seconds','correspondence_seconds','total_seconds'])]
            for label,q in [('median',.5),('q1',.25),('q3',.75)]:record[field+'_'+label]=float(np.quantile(selected,q)) if selected else float('nan')
        stats.append(record)
    write_csv(dest/'PER_REPEAT.csv',rows);write_csv(dest/'SUMMARY.csv',stats);write_csv(dest/'CP_NUMERICAL_CHECKS.csv',checks)
    report=dict(status='PASS',finite_packets=10,query_count_per_repeat=32,
        primary_method_configurations=len(groups),per_query_rows=sum(32 for _ in rows),
        all_query_failure_counts=sum(x['failure_count'] for x in rows),
        CP_points_checked=len(checks),CP_nearest_of_ENF_and_NN512_check=True,
        minimum_CP_distance_to_any_NN512_candidate=min(cp_candidates),
        CP_is_not_discrete_NN=all(x>1e-9 for x in cp_candidates),
        packet_sha256={p.name:sha(p) for p in packets},summary_sha256=sha(dest/'SUMMARY.csv'),
        source_sha256=sha(__file__),timing='Fresh serial CPU single thread, all representation construction charged. NN exact nested fits reused and cumulative actual generation through M charged. No original GPU times pooled.',
        CP_scope='Continuous local fitted-sheet optimization with multi-start/resolution checks; not a formal mathematical global certificate over arbitrary spurious empirical components.')
    write_json(dest/'AUDIT.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser(description='Summarize the ten final-paper finite CP/ENF/NN/Oracle packets.')
    p.add_argument('phase',choices=['finite']);p.parse_args()
    finite()
