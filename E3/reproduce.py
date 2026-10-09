"""Draw all twelve current E3 figures from the complete saved table, no fitting."""
from __future__ import annotations
import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import types
ROOT=Path(__file__).resolve().parent

def core():
    base=ROOT/'plotting'
    spec=importlib.util.spec_from_file_location('e3_footer_only',base/'no_footer_helpers.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    tree,_=helper.without_footers(ast.parse((base/'original_figure_core.py').read_text(encoding='utf-8')))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_rows')
    matches=[n for n in ast.walk(function) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='REF_COUNTS']
    if len(matches)!=1: raise ValueError('Original reference lookup changed')
    class ReferenceOnly(ast.NodeTransformer):
        def visit_Subscript(self,n):
            if n is matches[0]:
                return ast.IfExp(test=ast.parse("p == 'rev'",mode='eval').body,
                                 body=ast.Constant(262144),orelse=copy.deepcopy(n))
            return self.generic_visit(n)
    tree=ReferenceOnly().visit(tree)
    tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module=='design')]
    design={}
    exec(compile((base/'original_design.py').read_text(encoding='utf-8'),str(base/'original_design.py'),'exec'),design)
    result=types.ModuleType('e3_public_figures');result.__file__=str(base/'original_figure_core.py');result.N=design['N']
    exec(compile(ast.fix_missing_locations(tree),result.__file__+'::no_footer_current_reference','exec'),result.__dict__)
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'figures')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--table',type=Path,default=ROOT/'data/summary.csv',help='Optional complete evaluated NEW run, never overwrites saved paper data')
    args=parser.parse_args()
    c=core();rows=c.parse_rows(c.read_csv(args.table))
    if args.verify_only:
        print(json.dumps(dict(rows=len(rows),logical_figures=12,reverse_reference_all=262144)));return
    if args.output.exists(): raise ValueError('Use a fresh --output directory; do not overwrite reference figures')
    report=c.draw(rows,args.output,dict(release='paper62',no_new_fits=True,reverse_reference_all=262144))
    print(json.dumps(dict(figures=report['logical_figures'],exports=len(report['outputs']),green_background=False,footer_removed=True)))

if __name__=='__main__': main()
