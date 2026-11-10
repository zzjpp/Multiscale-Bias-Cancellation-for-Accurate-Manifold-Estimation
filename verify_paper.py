"""Check final-manuscript protocol/source consistency, not saved-figure replay."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
from paper_runs.protocol import ROOT,GEOMETRIES,SIGMAS,SIZES,EXPERIMENTS,jobs

def audit():
    checks=[]
    def check(label,condition):
        if not condition:raise ValueError('Paper protocol mismatch: '+label)
        checks.append(label)
    paper=json.loads((ROOT/'PAPER_REFERENCE.json').read_bytes())
    check('final manuscript',paper['manuscript_pdf']['sha256']=='bdc7b87ba20118aaf323bbe25d6b6921235e9f18c1c1d0453604c92f60d561be')
    for suffix in ('all_e1','six_manifolds'):
        cfg=json.loads((ROOT/'E1'/('theory_path_population_'+suffix)/'config.json').read_bytes())
        check('E1 bandwidth/noise '+suffix,tuple(sorted(cfg['sigma_grid']))==SIGMAS and cfg['bandwidth_path']['a1']==3 and cfg['bandwidth_path']['a2']==6 and cfg['kernel_power']==6)
        check('E1 Newton '+suffix,(cfg['solver']['max_iterations'],cfg['solver']['max_backtracks'],cfg['solver']['projected_residual_tolerance'],cfg['solver']['full_residual_tolerance'])==(24,16,2e-12,2e-10))
        yy=json.loads((ROOT/'E1'/('theory_path_population_yy_'+suffix)/'config.json').read_bytes())
        check('E1 YY '+suffix,yy['yy_path']['kappas']==[1.25,5.,8.] and yy['yy_path']['c0']==3 and yy['yy_path']['c2']==12)
    from paper_runs.worker import load
    e3=load(Path('E3/science_support.py'),'protocol_e3_design')
    check('E3 sample sizes',tuple(e3.N[g] for g in GEOMETRIES)==(12000,24000,24000,60000,24000,36000))
    check('E3 reverse queries',tuple(e3.REVERSE_QUERIES[g] for g in GEOMETRIES)==(1024,8192,8192,16384,8192,1024))
    check('E3 all original scales',len(e3.GRID)==28 and (e3.GRID[0],e3.GRID[-1])==(.06,5.))
    check('E3 original YY baselines',e3.GROUPS['YY']==('Single_0p4r1','Single_r1','Single_1p6r1','Multi'))
    for g in GEOMETRIES:
        q,reference=e3.assets(g,'rev')
        check('E3 full reverse set '+g,len(q)==e3.REVERSE_QUERIES[g] and reference.shape==(262144,q.shape[1]) and np.isfinite(reference).all())
    check('E2 nested sample sizes',SIZES==(2000,5000,10000,20000,50000,100000,200000,500000))
    e4=json.loads((ROOT/'E4/correspondence/frozen/E4/fitting/original/config.json').read_bytes())
    check('E4 design',(e4['sigma'],e4['h1'],e4['h2'],e4['sample_size'],e4['query_count'])==(.06,.12,.24,200000,32))
    check('E4 simultaneous residual tests',(e4['matching']['projected_residual_tolerance'],e4['matching']['full_residual_tolerance'])==(1e-11,1e-9))
    population=json.loads((ROOT/'E5/config/population.json').read_bytes())
    finite=json.loads((ROOT/'E5/config/finite_sample.json').read_bytes())
    check('E5 population',population['query_count']==64 and population['kappas']==[1.25,5.,8.] and population['slope_window']==[.008,.01,.0125,.016])
    check('E5 finite',finite['sample_sizes']==[240000,480000,960000] and finite['repeats']==20 and finite['query_count']==256 and finite['c0']==3 and finite['kappas']==[5.,6.,12.])
    check('E5 evidence not upgraded',finite['formal_evidence'] is False and finite['legacy_failed_gate']['pass'] is False and 'selection_source' in finite['provenance'])
    expected={'E1':336,'E2':360,'E3':378,'E4':11,'E5':67,'nonuniform_E1':84,'nonuniform_E2':120}
    check('complete paper job grid',dict(Counter(j['experiment'] for j in jobs(EXPERIMENTS)))==expected)
    nonuniform=jobs(('nonuniform_E2',))
    check('nonuniform E2 only paper method',{j['method'] for j in nonuniform}=={'STSP'})
    check('no independent E7/no withdrawn E2 overlay',not (ROOT/'E7').exists() and not (ROOT/'E2/population_reference').exists())
    return dict(schema='FINAL_PAPER_PROTOCOL_AUDIT_V1',status='MATCH_PROTOCOL',checks=checks,
        planned_tasks=sum(expected.values()),full_numerical_rerun=False,
        evidence='Source/configuration/design-input consistency only; new numerical results require run_paper.py.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path);a=p.parse_args();report=audit()
    if a.run:
        run=a.run.resolve();plan=json.loads((run/'PLAN.json').read_bytes());result=json.loads((run/'EVALUATION.json').read_bytes())
        from release_contract import run_source_digest
        if plan['source_manifest_sha256']!=run_source_digest(verify=True) or result['source_manifest_sha256']!=plan['source_manifest_sha256']:raise ValueError('Run/source identity mismatch')
        if result['status']!='PASS':raise ValueError('Run did not pass its declared evaluation')
        report['run']=dict(profile=plan['profile'],completed_tasks=result['tasks'],paper_evidence=plan['profile']=='paper',
            scope='declared selected experiment plan, not an independent replication or badge')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
