"""Isolated nonuniform-density E1. Reuse the unchanged original fitting body."""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'frozen'))
from e1_theory_path import experiment as original
from density import DensityGeometry, cpu_operator, gpu_operator
CONFIG = json.loads((ROOT/'config.json').read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def write_fresh(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Write outside isolated study')
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(plain(value), indent=2, allow_nan=False)+'\n').encode()
    with path.open('xb') as f:
        if path.suffix == '.gz':
            with gzip.GzipFile(fileobj=f, filename='', mode='wb', mtime=0) as g:
                g.write(encoded)
        else:
            f.write(encoded)


def read_packet(path):
    with gzip.open(path, 'rt', encoding='utf8') as f:
        return json.load(f)


def source_hashes():
    paths = [ROOT/n for n in ('study.py', 'density.py', 'fast_gpu.py', 'config.json', 'SOURCE_BINDING.json')]
    paths += sorted((ROOT/'frozen').rglob('*.py'))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def run_cell(key, sigma, resolution, backend='gpu_cupy', device=0, query_ids=None,
             reduced=False, amplitude=None):
    cfg = copy.deepcopy(CONFIG)
    if reduced:
        cfg['geometries'][key]['integration_resolution'][resolution] = (512 if key == 'circle_r2' else 64 if key == 'clifford_torus_r4' else 384 if key == 'corrugated_torus_r3' else 96)
        cfg['resolutions'][resolution]['stsp_radial_order'] = 64
    alpha = cfg['density']['amplitude'] if amplitude is None else amplitude
    geometry = DensityGeometry(key, alpha)

    def factory(method, geometry, h, sigma, settings, power, padding, backend, device):
        if method != 'STSP':
            raise ValueError('Only the requested STSP experiment is in scope')
        args = (geometry, h, sigma, settings['stsp_latent_count'], settings['stsp_radial_order'], padding)
        if backend == 'cpu':
            return cpu_operator(*args, power=power)
        if backend == 'gpu_cupy':
            return gpu_operator(*args, device=device, power=power)
        raise ValueError(backend)

    # New function globals; neither the original module nor frozen source is patched.
    namespace = dict(original.run_cell.__globals__, GEOMETRIES={key: geometry}, _operator=factory)
    runner = types.FunctionType(original.run_cell.__code__, namespace,
                                'unchanged_E1_fitting_body', original.run_cell.__defaults__)
    begun = time.monotonic()
    raw = runner(cfg, 'STSP', key, float(sigma), resolution, backend, device, query_ids)
    raw.update(study_schema='E1_NONUNIFORM_DENSITY_CELL_V1',
               density_amplitude=alpha, reduced_quadrature=bool(reduced),
               all_queries_selected=query_ids is None,
               query_rule=cfg['geometries'][key]['query_rule'],
               source_hashes=source_hashes(), seconds=time.monotonic()-begun,
               normalization='surface-area density, not parameter-uniform',
               undefined_serialization='nonfinite coordinates/metrics saved as null, with complete failures',
               sphere_angular_note='rotated integration coordinates; linear density permits exact angular moments through degree3; no fitting true normal')
    return raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', default='circle_r2', choices=CONFIG['geometry_order'])
    p.add_argument('--sigma', type=float, default=.03)
    p.add_argument('--resolution', choices=['production','reference'], default='production')
    p.add_argument('--backend', choices=['cpu','gpu_cupy'], default='cpu')
    p.add_argument('--reduced', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = run_cell(a.geometry, a.sigma, a.resolution, a.backend, query_ids=[17], reduced=a.reduced)
    write_fresh(a.output, result)
    print(json.dumps(plain(dict(metrics=result['metrics'],failures=result['failures'],seconds=result['seconds']))))


if __name__ == '__main__':
    main()
