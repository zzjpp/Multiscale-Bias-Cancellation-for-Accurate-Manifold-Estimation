"""Explicit unchanged-function extraction from the frozen public E2 runner."""
import ast
import importlib
import importlib.util
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent
FROZEN=ROOT/'frozen/E2'
KEYS=('circle_r2','sphere2_r3','clifford_torus_r4','sphere3_r4','sphere2_r10','corrugated_torus_r3')
SIZES=(2000,5000,10000,20000,50000,100000,200000,500000)


def load():
    # Do not import the old main/release_contract or pretend that the new density
    # is an unchanged public-release run. Reuse only these exact function ASTs.
    sys.path.insert(0,str(FROZEN/'corrugated'))
    spec=importlib.util.spec_from_file_location('nonuniform_e2_original_support',FROZEN/'corrugated/run_extension.py')
    support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
    tree=ast.parse((FROZEN/'run_fits.py').read_text(encoding='utf8'))
    names={'select','normalize','cell','gate'}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in functions}!=names:raise ValueError('Original function contract changed')
    namespace=dict(ROOT=FROZEN,np=np,sys=sys,importlib=importlib,original=support,KEYS=KEYS,SIZES=SIZES)
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(FROZEN/'run_fits.py'),'exec'),namespace)
    return namespace,support


def metric_function():
    tree=ast.parse((FROZEN/'evaluate_fits.py').read_text(encoding='utf8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='native_metric')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(FROZEN/'evaluate_fits.py'),'exec'),ns)
    return ns['native_metric']
