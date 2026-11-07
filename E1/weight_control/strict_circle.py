"""One additional shared CPU population solve, strictly for numerical validation."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
package=ROOT/'frozen/five/src/e1_theory_path'
spec=importlib.util.spec_from_file_location('_strict_circle_frozen',package/'__init__.py',submodule_search_locations=[str(package)])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
exp=importlib.import_module('_strict_circle_frozen.experiment')
plan=json.loads((ROOT/'PLAN.json').read_bytes())
cfg=json.loads((ROOT/'frozen/five/config.json').read_bytes())
strict=plan['strict_circle_check']
cfg=copy.deepcopy(cfg)
cfg['geometries']['circle_r2']['integration_resolution']['reference']=strict['latent_count']
cfg['resolutions']['reference']['stsp_radial_order']=strict['radial_order']
for k in ['projected_residual_tolerance','full_residual_tolerance','coordinate_correction_tolerance']:
    cfg['solver'][k]=strict[k]
out=ROOT/'strict_circle_validation';out.mkdir(exist_ok=False)
exp.write_json(out/'CONFIG.json',cfg)
start=time.perf_counter()
try:
    result=exp.run_cell(cfg,'STSP','circle_r2',strict['sigma'],'reference',backend='cpu')
    result['validation_elapsed_seconds']=time.perf_counter()-start
    exp.write_json(out/'PACKET.json',result)
    if result['failure_count'] or result['single_failure_count']:raise ArithmeticError('Strict circle solve failed')
    print('Strict circle shared solve completed',result['validation_elapsed_seconds'],flush=True)
except BaseException:
    import traceback
    (out/'ERROR.txt').write_text(traceback.format_exc(),encoding='utf8')
    raise
