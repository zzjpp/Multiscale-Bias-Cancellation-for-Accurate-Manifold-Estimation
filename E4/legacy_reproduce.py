"""Current fixed-query E4 saved-result figure, or explicit original GPU rerun."""
from __future__ import annotations
import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent

def read(path):
    with path.open(encoding='utf-8',newline='') as h: rows=list(csv.DictReader(h))
    for row in rows:
        row['candidate_count']=int(row['candidate_count']) if row['candidate_count'] else None
        for key in ('median','q1','q3','full_seconds_median','full_seconds_q1','full_seconds_q3'):
            if key in row: row[key]=float(row[key])
    return rows

def draw(output):
    source=ROOT/'plotting/original_scan.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    func=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='plot_accuracy'))
    saves=[n for n in ast.walk(func) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='savefig'
           and n.args and isinstance(n.args[0],ast.BinOp) and isinstance(n.args[0].right,ast.Constant)
           and str(n.args[0].right.value).endswith('.pdf')]
    if len(saves)!=1: raise ValueError('Original PDF drawing expression changed')
    saves[0].keywords.append(ast.keyword(arg='metadata',value=ast.parse("{'CreationDate':None,'ModDate':None}",mode='eval').body))
    config=json.loads((ROOT/'plotting/density_scan_config.json').read_bytes())
    scope=dict(np=np,Path=Path,SCAN=config)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[func],type_ignores=[])),str(source)+'::PDF_metadata_only','exec'),scope)
    accuracy=read(ROOT/'plotting/accuracy_summary.csv');cost=read(ROOT/'plotting/accuracy_cost_table.csv')
    if len(accuracy)!=34 or len(cost)!=16: raise ValueError('Complete E4 tables required')
    output.mkdir(parents=True,exist_ok=False)
    scope['plot_accuracy'](accuracy,cost,output)
    print(json.dumps(dict(logical_figures=1,shared_empirical_queries=32,candidate_counts=config['candidate_counts'],new_fits=False)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('figures','verify','gate','recompute'),default='figures')
    p.add_argument('--output',type=Path,default=ROOT/'figures')
    args=p.parse_args()
    if args.mode=='figures': draw(args.output);return
    action={'verify':'verify','gate':'gate','recompute':'run'}[args.mode]
    raise SystemExit(subprocess.call([sys.executable,'-X','utf8','-B',str(ROOT/'fitting/fixed_queries.py'),action]))

if __name__=='__main__': main()
