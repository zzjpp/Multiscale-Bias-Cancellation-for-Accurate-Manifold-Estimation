"""Compare archived evidence to the final arxiv_final.pdf manuscript; never launches numerical fits."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
GEOMETRIES=['circle_r2','sphere2_r3','clifford_torus_r4','sphere3_r4','sphere2_r10','corrugated_torus_r3']

def rows(path):
    with (ROOT/path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))

def audit():
    claims=[]
    def record(label,actual,expected,match):
        if not match: raise ValueError(f'{label}: {actual!r} != {expected!r}')
        claims.append(dict(claim=label,observed=actual,paper=expected,status='MATCH'))
    slopes=rows('E1/data/e1_theory_path_slopes.csv')
    for geom,expected in [('circle_r2',[2.001,4.004]),('clifford_torus_r4',[2.003,4.008]),('sphere2_r10',[2.002,4.005])]:
        for estimator,value in zip(('Single','Multi'),expected):
            cell=next(r for r in slopes if r['method']=='STSP' and r['geometry']==geom and r['estimator']==estimator and r['window']=='all_7')
            actual=float(cell['slope'])
            record(f'E1 STSP {geom} {estimator} slope',actual,value,round(actual,3)==value)
    e2=rows('E2/data/summary.csv')
    record('E2 complete table rows',len(e2),768,len(e2)==768)
    record('E2 six geometries',sorted({r['geometry'] for r in e2}),sorted(GEOMETRIES),{r['geometry'] for r in e2}==set(GEOMETRIES))
    e2vals={}
    for est,printed in [('Single_h2','1.343e-03'),('Multi','5.243e-05')]:
        r=next(r for r in e2 if r['method']=='STSP' and r['geometry']=='circle_r2' and float(r['scale'])==500000 and r['estimator']==est and r['metric']=='E_avg')
        value=float(r['median']);e2vals[est]=value
        record('E2 circle n500000 '+est+' E_avg',value,printed,format(value,'.3e')==printed)
    ratio=e2vals['Single_h2']/e2vals['Multi']
    record('E2 circle improvement factor',ratio,25.6,round(ratio,1)==25.6)
    e3=rows('E3/data/summary.csv')
    record('E3 complete table rows',len(e3),6720,len(e3)==6720)
    protocol=[]
    for geom,n,nq in zip(GEOMETRIES,[12000,24000,24000,60000,24000,36000],[1024,8192,8192,16384,8192,1024]):
        cells=[r for r in e3 if r['geometry']==geom]
        ok=all(int(r['n'])==n and float(r['sigma'])==.015 for r in cells)
        reverse=[r for r in cells if r['phase']=='rev']
        ok &= bool(reverse) and all(int(r['reference_count'])==262144 and int(r['query_count'])==nq for r in reverse)
        ok &= len({float(r['scale']) for r in cells})==28
        record('E3 '+geom+' n/reverse queries/reference/scales',[n,nq,262144,28],[n,nq,262144,28],ok)
        protocol.append(dict(geometry=geom,n=n,reverse_queries=nq,reverse_references=262144,scales=28))
    for r in e3:
        expected=float(r['query_failure_rate'])<=.01
        if r['phase']=='main': expected &= int(r['complete_repeats'])>=18
        if (r['reliable'].lower()=='true')!=expected: raise ValueError('E3 reliability disagreement')
    if not any(r['method']=='DR' and r['geometry']=='sphere3_r4' and r['phase']=='main' and r['reliable'].lower()=='false' for r in e3):
        raise ValueError('DR S3 failure evidence missing')
    e4=rows('E4/correspondence/finite_summary/SUMMARY.csv')
    cp=next(r for r in e4 if r['method']=='CP')
    record('E4 CP complete repeats',int(cp['complete_repeats']),10,int(cp['complete_repeats'])==10)
    record('E4 CP total median seconds',float(cp['total_seconds_median']),25.6,abs(float(cp['total_seconds_median'])-25.6)<.5)
    for r in e4:
        if int(r['total_query_failures'])!=0:raise ValueError('E4 query failures changed')
    controlled=rows('E1/weight_control/analysis/SLOPES.csv')
    for r in controlled:
        if r['method']=='naive':record('Figure2 naive '+r['geometry'],float(r['slope_all_7']),2.,abs(float(r['slope_all_7'])-2)<.001)
    result=json.loads((ROOT/'E4/correspondence/counterexample/RESULT.json').read_bytes())
    record('E.13 nonunique CP pair',len(result['CP_angles']),2,len(result['CP_angles'])==2 and float(result['CP_g'])<float(result['ENF_g']))
    record('E.17/E.18 complete nonuniform table',len(rows('nonuniform/E2/data/NONUNIFORM_SUMMARY.csv')),768,len(rows('nonuniform/E2/data/NONUNIFORM_SUMMARY.csv'))==768)
    packets=[json.loads(p.read_bytes())['result'] for p in sorted((ROOT/'E5/data/population').glob('e5_population_*.json'))]
    record('E5 population packets',len(packets),7,len(packets)==7)
    if any(p['query_count']!=64 or p['retained_count']!=64 or p['failure_count'] or p['kappas']!=[1.25,5.,8.] for p in packets):
        raise ValueError('Population protocol differs')
    selected=sorted([p for p in packets if p['sigma'] in {.008,.01,.0125,.016}],key=lambda p:p['sigma'])
    if len(selected)!=4: raise ValueError('Missing E5 slope-window packets')
    expected={'Single':[2.001,1.995,2.013],'Scalar-A':[2.038,3.993,2.038],'Scalar-C':[1.992,1.992,4.005],'Directional':[4.002,3.993,4.005]}
    for method,values in expected.items():
        for metric,value in zip(('geometric_max','axial_normal_max','complementary_normal_max'),values):
            # Recompute each maximum from the 64 per-query errors, then regress.
            component={'geometric_max':'geometric','axial_normal_max':'axial_normal','complementary_normal_max':'complementary_normal'}[metric]
            maxima=[]
            for p in selected:
                actual=max(q['metrics'][method][component] for q in p['query_rows'])
                if actual!=p['summaries'][method][metric]: raise ValueError('E5 summary not equal to query maximum')
                maxima.append(actual)
            slope=float(np.polyfit(np.log([p['sigma'] for p in selected]),np.log(maxima),1)[0])
            record(f'Table2 {method} {metric} slope',slope,value,round(slope,3)==value)
    provenance=json.loads((ROOT/'E5/data/PROVENANCE.json').read_bytes())
    for name,sha in provenance['reference_files'].items():
        if hashlib.sha256((ROOT/'E5/paper_figures'/name).read_bytes()).hexdigest()!=sha:
            raise ValueError('E5 selected export changed')
    availability=json.loads((ROOT/'E5/data/finite_packet_availability.json').read_bytes())
    if availability['status']!='MISSING_LOCAL_RAW_PACKETS' or len(availability['expected_files'])!=60:
        raise ValueError('Missing finite evidence must be disclosed')
    return dict(schema='ARXIV_FINAL_RESULT_CONSISTENCY_V1',status='MATCH_WITH_DOCUMENTED_RAW_DATA_GAP',
        matched_claims=len(claims),claims=claims,E3_protocol=protocol,
        unresolved=[dict(experiment='E5 finite / Figure8',status='MISSING_LOCAL_RAW_PACKETS',expected=60,
            interpretation='Selected exports match; historical median/IQR raw-data verification incomplete')],
        manuscript_note='Local CP scope and raw-data limitations are explicitly disclosed',
        evidence='archived-result consistency; not an independent full fitting rerun')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();report=audit()
    if args.output:
        if args.output.exists(): raise FileExistsError('Use a fresh report path')
        args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
