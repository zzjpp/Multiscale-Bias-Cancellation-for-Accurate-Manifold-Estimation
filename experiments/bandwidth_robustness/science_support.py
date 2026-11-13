"""Public relative-path interface to the unchanged, version-specific E3 solvers.

One method/phase per interpreter: the original dryy packages share a namespace.
This is a new-run interface, not a fabricated replay of the historical queue.
"""
from __future__ import annotations
import copy
from dataclasses import asdict, is_dataclass
import importlib
import json
from pathlib import Path
import sys
import numpy as np
from output_adapter import expose_outputs

ROOT = Path(__file__).resolve().parent
KEYS = ('circle_r2', 'sphere2_r3', 'clifford_torus_r4', 'sphere3_r4', 'sphere2_r10', 'corrugated_torus_r3')
N = dict(zip(KEYS, (12000, 24000, 24000, 60000, 24000, 36000)))
REVERSE_QUERIES = dict(zip(KEYS, (1024, 8192, 8192, 16384, 8192, 1024)))
GRID = (.06,.075,.09,.11,.135,.165,.2,.245,.3,.37,.45,.55,.67,.82,1.,1.2,1.4,1.6,1.8,2.,2.25,2.5,2.75,3.,3.25,3.5,4.,5.)
GROUPS = {'STSP': ('Single_h1','Single_h2','Multi'), 'DR': ('Single_h1','Single_h2','Multi'),
          'YY': ('Single_0p4r1','Single_r1','Single_1p6r1','Multi')}

def group_names(method):
    return GROUPS[method]

def plain(value):
    if is_dataclass(value): return plain(asdict(value))
    if isinstance(value, np.ndarray): return plain(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [plain(v) for v in value]
    return value

def load_library(method, phase, key):
    if method not in GROUPS or phase not in ('main','rev') or key not in KEYS:
        raise ValueError('Unknown E3 case')
    name = 'STSP' if method == 'STSP' else 'DR' if method == 'DR' else (
        'YY_CORRUGATED' if key == KEYS[-1] else 'YY_MAIN' if phase == 'main' else 'YY_REV')
    package = ROOT / 'frozen' / name
    sys.path.insert(0, str(package / 'src'))
    sys.path.insert(0, str(package / 'gpu_backend'))
    if method == 'STSP':
        sys.path.insert(0, str(ROOT / 'frozen/DR/src'))
        from e3.config import CONFIG
        from e3.geometry import BY_KEY, noisy_sample
        from e3 import pipelines
        return dict(name=name,method=method,config=copy.deepcopy(CONFIG),BY_KEY=BY_KEY,
                    sample=noisy_sample,pipelines=pipelines,plain=plain)
    from dryy.geometry import Geometry, noisy_sample
    from dryy.metrics import summarize_points
    from dryy.yy import YYFailure
    from dryy.yy_common import prepare_common_start, evaluate_common_start
    config_path = ROOT / 'frozen/corrugated_dr_yy.json' if name == 'YY_CORRUGATED' else package / 'configs/e3_dr_yy_frozen.json'
    config = json.loads(config_path.read_bytes())
    config['n'] = N[key]  # The only authorized budget change relative to the old solver config.
    return dict(name=name,method=method,config=config,Geometry=Geometry,sample=noisy_sample,
                plain=plain,summarize=summarize_points,YYFailure=YYFailure,
                prepare_cpu=prepare_common_start,evaluate_cpu=evaluate_common_start)

def geometry(lib, key, method):
    if key == KEYS[-1]:
        sys.path.insert(0, str(ROOT / 'frozen/corrugated_geometry/src'))
        from corrugated.geometry import CorrugatedTorusGeometry
        class ControlledYY(CorrugatedTorusGeometry):
            def preliminary(self,x,sigma):
                return np.asarray(x,dtype=float)+float(sigma)*self.normal_basis(x)[:,0]
        base = ControlledYY()
    elif method == 'STSP':
        base = lib['BY_KEY'][key]
    else:
        base = lib['Geometry'](**next(r for r in lib['config']['geometries'] if r['key'] == key))
    if method == 'YY': return base
    class NoTrueNormal:
        def __getattr__(self,name): return getattr(base,name)
        def normal_basis(self,*args,**kwargs):
            raise AssertionError('True normals forbidden in STSP/DR fitting and matching')
    return NoTrueNormal()

def observations(lib, key, repeat, n=None):
    if repeat not in range(20): raise ValueError('Repeat outside original design')
    count = N[key] if n is None else int(n)
    if key != KEYS[-1]:
        if lib['method'] == 'STSP':
            _,_,data,lineage = lib['sample'](lib['BY_KEY'][key],repeat,count)
            return data,lineage
        return lib['sample'](lib['config'],geometry(lib,key,'YY'),repeat,count)
    config = json.loads((ROOT / 'frozen/corrugated_geometry/e7_config.json').read_bytes())
    seeds = config['seed_list'][repeat]
    clean_rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(seeds['clean'])))
    noise_rng = np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(seeds['noise'])))
    clean = geometry(lib,key,'YY').sample_clean(clean_rng,count)
    noise = noise_rng.standard_normal(clean.shape)
    return clean+.015*noise,dict(seeds,n=count)

def assets(key, phase):
    query = np.load(ROOT / 'assets' / f'{key}_{phase}.npy',allow_pickle=False)
    reference_name = f'{key}_reverse_reference.npy' if phase == 'rev' and key == KEYS[-1] else f'{key}_reference.npy'
    reference = np.load(ROOT / 'assets' / reference_name,allow_pickle=False)
    expected = REVERSE_QUERIES[key] if phase == 'rev' else 512 if key == KEYS[-1] else 256
    if len(query) != expected or (phase == 'rev' and len(reference) != 262144):
        raise ValueError('Registered query/reference budget changed')
    return query,reference
