"""Explicit ENF-input query freeze; original E4 numerical arithmetic retained."""
from __future__ import annotations
import argparse
import ast
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ.setdefault(name,'1')
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_original_scan',ROOT/'original/run_density_scan.py')
scan=importlib.util.module_from_spec(spec);spec.loader.exec_module(scan)
import numpy as np
from e4 import pipeline
from e4.geometry import query_grid, arc_table

INPUTS=json.loads((ROOT/'FROZEN_INPUTS.json').read_bytes())
FROZEN=None
VALIDATION=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def select_repeat(repeat):
    global FROZEN,VALIDATION
    record=INPUTS['repeats'][str(repeat)]
    p=ROOT/'queries'/f'repeat_{repeat:02d}.npy'
    if sha(p)!=record['query_file_sha256']:raise ValueError('Frozen ENF queries changed')
    FROZEN=np.load(p,allow_pickle=False)
    if FROZEN.shape!=(32,3) or not np.isfinite(FROZEN).all():raise ValueError('Invalid frozen query array')
    FROZEN.flags.writeable=False
    VALIDATION=[]

def fixed_input(count,query_subset):
    if FROZEN is None:raise ValueError('Select frozen actual repeat first')
    value=FROZEN if query_subset is None else FROZEN[np.asarray(query_subset,dtype=int)]
    if value.shape!=(count,3):raise ValueError('Frozen query and original-grid selection mismatch')
    return value

def checked_fixed_point(ref,point,operator):
    difference=float(np.max(np.abs(ref.point-point)))
    # Original CPU/GPU point equivalence threshold; not a changed solver tolerance.
    if difference>float(scan.SCAN['gpu_equivalence']['point_absolute_tolerance']):
        raise ValueError('Regenerated fine root inconsistent with archived ENF query')
    state=operator.evaluate(point)
    residual=float(np.linalg.norm(state.residual))
    if residual>float(scan.CONFIG['matching']['full_residual_tolerance']):
        raise ValueError('Frozen query is not an h1 residual root at original tolerance')
    VALIDATION.append(dict(regenerated_point_difference=difference,frozen_h1_full_residual=residual))
    return point

def build_overlay():
    path=ROOT/'original/src/e4/pipeline.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    function=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_main'))
    function.name='fixed_main'
    count_index=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='count' for t in n.targets))
    function.body.insert(count_index+1,ast.parse('frozen=fixed_input(count,query_subset)').body[0])
    targets=[n for n in ast.walk(function) if isinstance(n,ast.Assign) and
             len(n.targets)==1 and ast.unparse(n.targets[0])=='z[j]' and ast.unparse(n.value)=='ref.point']
    if len(targets)!=1:raise ValueError('Original single fine-point assignment changed')
    targets[0].value=ast.parse('checked_fixed_point(ref,frozen[j],operator_h1)',mode='eval').body
    namespace=dict(pipeline.__dict__,fixed_input=fixed_input,checked_fixed_point=checked_fixed_point)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),
                 str(path)+'::explicit_ENF_fixed_input','exec'),namespace)
    repeat_tree=ast.parse((ROOT/'original/run_density_scan.py').read_text(encoding='utf-8'))
    repeat=copy.deepcopy(next(n for n in repeat_tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_repeat'))
    calls=[n for n in ast.walk(repeat) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_main']
    if len(calls)!=1:raise ValueError('Original shared-result call changed')
    calls[0].func.id='fixed_main'
    repeated_namespace=dict(scan.__dict__,fixed_main=namespace['fixed_main'])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[repeat],type_ignores=[])),
                 str(ROOT/'original/run_density_scan.py')+'::fixed_query_caller','exec'),repeated_namespace)
    return namespace['fixed_main'],repeated_namespace['run_repeat'],hashlib.sha256(
        (ast.dump(function,include_attributes=False)+ast.dump(repeat,include_attributes=False)).encode()).hexdigest()

fixed_main,original_repeat,EFFECTIVE_AST_SHA=build_overlay()

def verify_inputs():
    for name,digest in INPUTS['original_sources'].items():
        if sha(ROOT/'original'/name)!=digest:raise ValueError('Original scientific source changed: '+name)
    for repeat,row in INPUTS['repeats'].items():
        p=ROOT/'data'/f'repeat_{int(repeat):02d}.npz'
        if sha(p)!=row['data_file_sha256']:raise ValueError('Frozen observation parent changed')

def contract(packet):
    z=FROZEN
    rows=packet['per_query']
    for method,count in [('Oracle',None),('ENF',None)]+[('NN',m) for m in scan.SCAN['candidate_counts']]:
        selected=sorted((r for r in rows if r['method']==method and r['candidate_count']==count),key=lambda r:r['query_id'])
        if [r['query_id'] for r in selected]!=list(range(32)):raise ValueError('Missing or duplicate fixed query')
        if not np.array_equal(z,np.asarray([r['reference'] for r in selected])):
            raise ValueError('Method does not use exact archived ENF coordinates')
    if len(VALIDATION)!=32:raise ValueError('Not all h1 references independently validated')

def gate():
    select_repeat(10)
    clean,noise,_=scan.load_frozen_parent(10,200000,False,ROOT/'data')
    data=clean+.06*noise
    weights=scan.two_scale_weights(.12,.24,.06)['weights']
    results={}
    for backend in ['cpu','gpu']:
        cls,sync,_=scan.backend_class(backend)
        h1,h2=cls(data,.12),cls(data,.24)
        results[backend]=fixed_main(h1,h2,.24,weights,N_q=32,M_match=32,
                                   query_subset=[0,8,16,24],phi=scan.CONFIG['nn_phase'])
        sync()
    checks=[]
    for label,key in [('reference',('common_reference','z')),('Oracle',('TNF','Z_T2')),
                      ('ENF',('ENF','Z_E2')),('NN',('NN','Z_NN2'))]:
        a,b=(np.asarray(results[n][key[0]][key[1]]) for n in ['cpu','gpu'])
        exact_mask=np.array_equal(np.isfinite(a),np.isfinite(b))
        finite=np.isfinite(a)&np.isfinite(b)
        delta=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else None
        checks.append(dict(branch=label,finite_masks_equal=exact_mask,max_difference=delta))
        if not exact_mask or delta is None or delta>2e-8:raise ValueError('New fixed-query CPU/GPU gate failed')
    for key in ['TNF','ENF']:
        if results['cpu'][key]['status']!=results['gpu'][key]['status']:raise ValueError('Status equivalence failed')
    if results['cpu']['NN']['library']['candidate_status']!=results['gpu']['NN']['library']['candidate_status']:
        raise ValueError('Candidate status equivalence failed')
    report=dict(schema='E4_FIXED_QUERY_ACTUAL_CPU_GPU_GATE_V1',status='PASS',sample_size=200000,
        selected_query_ids=[0,8,16,24],checks=checks,effective_AST_sha256=EFFECTIVE_AST_SHA,
        device=scan.backend_class('gpu')[2],frozen_h1_checks=VALIDATION)
    with (ROOT/'CPU_GPU_FIXED_QUERY_GATE.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2)
    print('ACTUAL_FIXED_QUERY_CPU_GPU_GATE_PASS',flush=True)

def run():
    verify_inputs()
    if not (ROOT/'CPU_GPU_FIXED_QUERY_GATE.json').exists():raise ValueError('Actual new GPU gate required')
    if json.loads((ROOT/'CPU_GPU_FIXED_QUERY_GATE.json').read_bytes())['status']!='PASS':raise ValueError('Gate failed')
    packets=ROOT/'outputs/packets';packets.mkdir(parents=True,exist_ok=False)
    for repeat in range(10,20):
        select_repeat(repeat)
        began=time.perf_counter()
        packet=original_repeat(repeat,'gpu',data_dir=ROOT/'data')
        contract(packet)
        packet['original_scan_schema']=packet['schema_version']
        packet['schema_version']='E4_FIXED_ENF_H1_QUERIES_GPU_RERUN_V1'
        packet['query_contract']=dict(exact_ENF_coordinate_sha256=INPUTS['repeats'][str(repeat)]['query_coordinate_sha256'],
            effective_AST_sha256=EFFECTIVE_AST_SHA,source_sha256=sha(Path(__file__)),
            regenerated_h1_checks=VALIDATION, all17method_density_inputs_exactly_equal=True)
        packet['actual_repeat_wall_seconds']=time.perf_counter()-began
        scan.write_packet(packet,packets/f'repeat_{repeat:02d}.json.gz')
        print('FIXED_QUERY_REPEAT_COMPLETE',repeat,packet['actual_repeat_wall_seconds'],flush=True)
    scan.aggregate(packets,ROOT/'outputs/formal')
    with (ROOT/'COMPLETION.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(schema='E4_FIXED_QUERY_GPU_COMPLETION_V1',complete_repeats=10,
                       actual_new_gpu_matching=True,query_count=32,query_contract='exact archived ENF h1 coordinates',
                       all_M=scan.SCAN['candidate_counts'],gate_sha256=sha(ROOT/'CPU_GPU_FIXED_QUERY_GATE.json'),
                       formal_audit_sha256=sha(ROOT/'outputs/formal/AUDIT.json')),stream,indent=2)
    print('ALL_TEN_FIXED_QUERY_REPEATS_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['gate','run','verify']);args=p.parse_args()
    if args.action in ('gate','run') and (ROOT/'RERUN_WITHDRAWN.json').exists() and not (ROOT/'RERUN_RESUMED_AUTHORIZATION.json').exists():
        raise RuntimeError('Latest user instruction cancels refitting; only coordinate unification is authorized')
    if args.action=='verify':verify_inputs();print('FROZEN_INPUTS_AND_ORIGINAL_SOURCES_VALID')
    else:{'gate':gate,'run':run}[args.action]()
