"""No-retry fresh nonuniform E2 queue, two isolated processes per idle GPU."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
from common import ROOT,source_contract,sha,fresh
from native import KEYS


def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        s=(p/'stat').read_text();f=s[s.rfind(')')+2:].split()
        return dict(pid=pid,start_ticks=int(f[19]),cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode().strip())
    except FileNotFoundError:return None


def label(task):return f'{task[0]}_{task[1]}_r{task[2]:02d}'


def status(payload):
    fresh(ROOT/'STATUS.next.json',dict(payload,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    os.replace(ROOT/'STATUS.next.json',ROOT/'STATUS.json')


def launch(task,slot):
    tag=label(task);output=ROOT/'results/parents'/(tag+'.json.gz')
    if output.exists() or (ROOT/'results/claims'/(tag+'.json')).exists():raise FileExistsError('No repeated parent dispatch')
    output.parent.mkdir(parents=True,exist_ok=True)
    argv=[sys.executable,'-B',str(ROOT/'parent.py'),'--method',task[0],'--geometry',task[1],'--repeat',str(task[2]),'--output',str(output)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(slot['gpu']),TMPDIR=str(ROOT/'scratch'),CUPY_CACHE_DIR=str(ROOT/'cache'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    log=ROOT/'logs'/(tag+'.log')
    with log.open('xb') as f:p=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    time.sleep(.03)
    ident=identity(p.pid)
    if ident is None:raise RuntimeError('Parent exited before identity capture; log preserved')
    claim=dict(task=list(task),slot=slot,identity=ident,source_contract_sha256=source_contract(),input_policy='saved original-noise/new-density maximum-n parent')
    fresh(ROOT/'results/claims'/(tag+'.json'),claim)
    return dict(process=p,claim=claim,task=task,slot=slot,log=log)


def once_audit(tasks):
    if (ROOT/'SCIENCE_AUDIT.json').exists():raise FileExistsError('Full audit is once-only')
    from aggregate import table
    rows=table(ROOT/'results/parents')
    import csv,gzip,numpy as np
    input_members=0;gate_count=0;outcomes=0
    for task in tasks:
        p=ROOT/'results/parents'/(label(task)+'.json.gz')
        with gzip.open(p,'rt',encoding='utf8') as f:packet=json.load(f)
        claim=json.loads((ROOT/'results/claims'/(label(task)+'.json')).read_bytes())
        completion=json.loads((ROOT/'results/completions'/(label(task)+'.json')).read_bytes())
        if sha(p)!=completion['sha256'] or packet['source_contract_sha256']!=claim['source_contract_sha256']:raise ValueError('Packet/execution/source binding failed')
        if packet['gate']['status']!='PASS':raise ValueError('Missing actual exact-parent gate')
        gate_count+=1
        for name,meta in packet['input_files'].items():
            path=ROOT/meta['path']
            if not path.resolve().is_relative_to(ROOT) or sha(path)!=meta['sha256']:raise ValueError('Stored parent input binding failed')
            arr=np.load(path,mmap_mode='r',allow_pickle=False)
            if list(arr.shape)!=meta['shape'] or str(arr.dtype)!=meta['dtype']:raise ValueError('Wrong input shape/dtype')
            input_members+=1
        outcomes+=sum(len(group) for cell in packet['results'] for group in cell['groups'].values())
    summary=ROOT/'NONUNIFORM_SUMMARY.csv'
    if gate_count!=360 or input_members!=1080 or outcomes!=1105920:
        raise ValueError('Full parent/input/estimator-query-prefix scope required')
    with summary.open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(schema='E2_NONUNIFORM_STORED_OUTPUT_AUDIT_V1',status='PASS',parents=360,summary_rows=768,exact_parent_gates=gate_count,input_arrays_bound=input_members,estimator_query_prefix_outcomes=outcomes,
                source_contract_sha256=source_contract(),summary_sha256=sha(summary),
                scientific_scope='Integrity/completeness/metric aggregation and exact-parent equivalence PASS. Not zero query failures or all-grid reliable; all partial finite statistics, failures, hollow markers and NaN retained.')
    fresh(ROOT/'SCIENCE_AUDIT.json',result)
    return result


def main():
    source=source_contract(verify=True)
    if (ROOT/'QUEUE_LOCK.json').exists() or (ROOT/'STATUS.json').exists():raise FileExistsError('Never restart existing controller')
    fresh(ROOT/'QUEUE_LOCK.json',identity(os.getpid()))
    # Repeat0 are full scientific parents, never disposable benchmarks/refits.
    pilots=[(m,g,0) for m in ('STSP','DR','YY') for g in KEYS]
    remainder=[(m,g,r) for r in range(1,20) for m in ('STSP','DR','YY') for g in KEYS]
    pending=pilots+remainder;tasks=list(pending);completed=[];active=[];errors=[]
    slots=[dict(id=i,gpu=i//2) for i in range(4)];begin=time.monotonic()
    while pending or active:
        free=[s for s in slots if not any(a['slot']['id']==s['id'] for a in active)]
        # Complete the first18 formal gates/parents before dispatching repeat1+.
        pilot_phase=any(t[2]==0 for t in pending) or any(a['task'][2]==0 for a in active)
        for slot in free:
            if not pending or errors or (pilot_phase and pending[0][2]!=0):break
            task=pending.pop(0)
            try:active.append(launch(task,slot))
            except Exception as e:errors.append(dict(task=task,reason=repr(e),traceback=traceback.format_exc()))
        for item in list(active):
            exit_code=item['process'].poll()
            if exit_code is None:continue
            active.remove(item);task=item['task'];tag=label(task)
            if exit_code!=0:
                errors.append(dict(task=task,exit_code=exit_code,identity=item['claim']['identity'],log=str(item['log'].relative_to(ROOT))))
            else:
                record=json.loads((ROOT/'results/completions'/(tag+'.json')).read_bytes())
                record['exit_code']=exit_code;completed.append(record);print('PARENT_COMPLETE '+json.dumps(record),flush=True)
        status(dict(phase='DRAINING_AFTER_ERROR' if errors else 'FORMAL_REPEAT0_PILOTS' if pilot_phase else 'FITTING',completed=len(completed),total=360,pending=len(pending),active=[a['claim'] for a in active],errors=errors,source_contract_sha256=source))
        if errors and not active:break
        if pending or active:time.sleep(10)
    if errors:
        status(dict(phase='STOPPED_AFTER_PARENT_ERROR',completed=len(completed),total=360,pending=len(pending),active=[],errors=errors));return 1
    status(dict(phase='ONCE_ONLY_STORED_OUTPUT_AUDIT',completed=360,total=360,pending=0,active=[],errors=[]))
    audit=once_audit(tasks)
    final=dict(status='COMPUTATION_AND_INTEGRITY_AUDIT_PASS',parents=360,source_contract_sha256=source,science_audit_sha256=sha(ROOT/'SCIENCE_AUDIT.json'),summary_sha256=audit['summary_sha256'],seconds=time.monotonic()-begin,completed=completed,figure_delivery_complete=False)
    fresh(ROOT/'FINAL_RECEIPT.json',final)
    status(dict(phase=final['status'],completed=360,total=360,pending=0,active=[],errors=[]))
    files=[p for p in ROOT.rglob('*') if p.is_file() and not any(s in {'scratch','cache','__pycache__'} for s in p.relative_to(ROOT).parts) and p.name not in {'FINAL_EVIDENCE.tar','FINAL_TRANSFER.json','source_bundle.tar'}]
    members={p.relative_to(ROOT).as_posix():dict(size=p.stat().st_size,sha256=sha(p)) for p in files}
    with tarfile.open(ROOT/'FINAL_EVIDENCE.tar','x') as tar:
        for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
    fresh(ROOT/'FINAL_TRANSFER.json',dict(schema='E2_NONUNIFORM_FULL_EVIDENCE_V1',members=members,archive_size=(ROOT/'FINAL_EVIDENCE.tar').stat().st_size,archive_sha256=sha(ROOT/'FINAL_EVIDENCE.tar'),science_status='PASS',science_audit_sha256=sha(ROOT/'SCIENCE_AUDIT.json')))
    return 0


if __name__=='__main__':sys.exit(main())
