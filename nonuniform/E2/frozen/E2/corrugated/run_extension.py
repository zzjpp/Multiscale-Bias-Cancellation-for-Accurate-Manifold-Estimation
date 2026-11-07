"""Unchanged numerical/support functions; no standalone experiment CLI."""
from dataclasses import asdict,is_dataclass
import importlib
import os
import platform
import sys
import numpy as np
import scipy
from scipy.spatial import cKDTree
from corrugated_geometry import sample
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')
def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def require_gpu_environment():
    import cupy as cp
    if sys.platform != 'linux' or platform.python_version() != '3.12.7':
        raise ValueError('Original Linux Python3.12.7 required')
    if (np.__version__, scipy.__version__, cp.__version__) != ('2.1.3', '1.18.1', '14.2.0'):
        raise ValueError('Original numerical environment required')
    if any(os.environ.get(name) != '1' for name in THREADS):
        raise ValueError('One CPU thread per worker required')
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    if name != 'NVIDIA L40' or cp.cuda.runtime.runtimeGetVersion() != 12090:
        raise ValueError('Original L40/CUDA12.9 required')
    cp.zeros(1, dtype=cp.float64)
    return cp, dict(backend='cupy', python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, cupy=cp.__version__, gpu_name=name,
        gpu_pci_bus_id=cp.cuda.Device().pci_bus_id, precision='float64',
        threads={name: os.environ[name] for name in THREADS},
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))


def fine_dr_rows(data, geometry, queries, config, gpu):
    """Independent fine fits, all queries including those whose Multi fails."""
    library = importlib.import_module('dryy_gpu' if gpu else 'dryy.dr')
    error_type = importlib.import_module('dryy.dr').DRFailure
    controls = dict(config['dr'])
    tree = cKDTree(data) if controls.get('gaussian_truncation_factor') is not None else None
    if gpu:
        operator = library.EmpiricalDRGPU(data, geometry, .1, controls, tree)
        estimate, solve = library.estimate_pca_normal_gpu, library.solve_fiber_gpu
    else:
        operator = library.EmpiricalDR(data, geometry, .1, controls, tree)
        estimate, solve = library.estimate_pca_normal, library.solve_fiber
    rows = []
    for truth in queries:
        try:
            basis = estimate(operator, truth)
            fit = solve(operator, truth, basis, .1, config['solver'])
        except error_type as error:
            rows.append(dict(status='failed', reason=error.reason, point=None,
                             diagnostics=dict(error.diagnostics)))
        else:
            rows.append(dict(status='success', reason=None, point=fit['point'],
                diagnostics={key: fit[key] for key in ('coordinate', 'fiber_basis', 'fiber_origin', 'trace', 'iterations', 'normal_space_source')}))
    return rows


def cell(method, data, geometry, queries, config, library, gpu):
    if method == 'STSP':
        cpu_operator = importlib.import_module('e2fs.operator').EmpiricalSTSP
        operator = importlib.import_module('e2fs.gpu_operator').GPUEmpiricalSTSP if gpu else cpu_operator
        library.EmpiricalSTSP = operator
        weights = importlib.import_module('e2fs.coefficients').two_scale_weights(2, 3, .1, .2, .005)
        if weights['status'] != 'converged':
            raise ValueError('Original STSP coefficient gate failed')
        result = library.run_cell(data, geometry, .2, queries, weights)
        result['weights'] = weights['weights']
        result['coefficient_evidence'] = weights
        return result
    backend = importlib.import_module('dryy_gpu') if gpu else library
    if method == 'DR':
        function = backend.dr_cell_gpu if gpu else library._dr_cell
        result = function(data, geometry, queries, .005, .1, .2, config)
        result['single_h1_rows'] = fine_dr_rows(data, geometry, queries, config, gpu)
        result['Single_h1'] = importlib.import_module('dryy.metrics').summarize_points(geometry, result['single_h1_rows'], 'point')
    else:
        function = backend.yy_cell_gpu if gpu else library._yy_cell
        result = function(data, cKDTree(data), geometry, queries, .005, 40., config)
    return result


def row_groups(method, result):
    if method == 'STSP':
        return {name: [row[key] for row in result[path]['queries']]
                for name, path, key in [('Single_h1','single','Single_h1'), ('Single_h2','single','Single_h2'), ('Multi','multi','ENF_Multi')]}
    groups = dict(Single_h2=result['single_rows'], Multi=result['multi_rows'])
    if method == 'DR':
        groups['Single_h1'] = result['single_h1_rows']
    else:
        groups['Single'] = groups.pop('Single_h2')
    return groups

