"""Original E3 numerical cells, extracted verbatim for the public entrypoint."""
import hashlib
import importlib
import importlib.util
import inspect
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from science_support import expose_outputs, group_names
ROOT = Path(__file__).resolve().parent
def install_dr_recorder(lib, gpu, records, rejections):
    spec = importlib.util.spec_from_file_location('explicit_original_rejection_adapter', ROOT / 'frozen/dr_rejected_basis_failure_adapter.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    if gpu:
        import cupy as xp
        module = importlib.import_module('dryy_gpu')
        name = 'solve_fiber_gpu'
    else:
        xp = np
        module = importlib.import_module('dryy.dr')
        name = 'solve_fiber'
    from dryy.dr import DRFailure

    def observe(value):
        rejections.append(value)
        return len(rejections) - 1
    original = getattr(module, name)
    wrapped = adapter.wrap(original, xp, DRFailure, observe)

    def recorded(operator, origin, basis, search_h, solver):
        caller = inspect.currentframe().f_back
        truth = caller.f_locals.get('truth')
        if truth is None:
            raise ValueError('Original DR query context not observable')
        value = wrapped(operator, origin, basis, search_h, solver)
        records.append(dict(query_truth=lib['plain'](truth), operator_h=float(operator.h), search_h=float(search_h), **{k: lib['plain'](value[k]) for k in ('point', 'coordinate', 'fiber_basis', 'fiber_origin', 'normal_space_source', 'trace')}))
        return value
    setattr(module, name, recorded)
    return (module, name, original)


def dr_cell(lib, data, g, queries, scale, gpu=True):
    records = []
    rejections = []
    module, name, original = install_dr_recorder(lib, gpu, records, rejections)
    try:
        function = module.dr_cell_gpu if gpu else importlib.import_module('dryy.runner')._dr_cell
        result = function(data, g, queries, 0.015, scale / 2, scale, lib['config'])
    finally:
        setattr(module, name, original)
    result['root_success_records'] = records
    result['rejected_bases'] = rejections
    groups = dict(Single_h1=result['single_h1_rows'], Single_h2=result['single_rows'], Multi=result['multi_rows'])
    return dict(scale=scale, groups=groups, summaries={k: result[k] for k in groups}, native=result)


def yy_cells(lib, data, g, queries, scales, gpu=True):
    config = lib['config']
    settings = dict(config['yy'], weight_residual_tolerance=config['validation']['weight_residual_tolerance'])
    from dryy.coefficients import yy_weights
    tree = cKDTree(data)
    if gpu:
        from dryy_gpu import prepare_common_start_gpu as prepare, make_candidate_cache_gpu as cache_fn, evaluate_common_start_gpu as evaluate
    else:
        prepare = lib['prepare_cpu']
        evaluate = lib['evaluate_cpu']
        cache_fn = None
    evaluate = expose_outputs(evaluate)
    names = group_names('YY')
    cells = [dict(scale=scale, groups={k: [] for k in names}, stages=[], weights=yy_weights(g.d, g.q, scale / 0.015 * np.asarray(settings['relative_radii']), settings['kernel_power'])) for scale in scales]
    starts = []
    for j, truth in enumerate(queries):
        try:
            start = prepare(data, tree, g, truth, 0.015, settings)
            cache = cache_fn(data, tree, start, 0.015, max(scales) / 0.015, settings) if cache_fn else None
        except lib['YYFailure'] as error:
            starts.append(dict(status='failed', reason=error.reason, diagnostics=error.diagnostics))
            for cell in cells:
                for rows in cell['groups'].values():
                    rows.append(dict(status='failed', reason=error.reason, point=None, diagnostics=error.diagnostics))
                cell['stages'].append(None)
            continue
        starts.append(dict(status='success', **start))
        for cell in cells:
            try:
                fitted = evaluate(data, tree, g, start, 0.015, cell['scale'] / 0.015, settings, **{'cached_vectors': cache} if gpu else {})
            except lib['YYFailure'] as error:
                for rows in cell['groups'].values():
                    rows.append(dict(status='failed', reason=error.reason, point=None, diagnostics=error.diagnostics))
                cell['stages'].append(None)
                continue
            cell['stages'].append(fitted['stages'])
            for name, point, stage in zip(names[:3], fitted['single_radius_points'], fitted['stages']):
                cell['groups'][name].append(dict(status=stage['status'], reason=stage['reason'], point=point, diagnostics=dict(common_start_digest=start['digest'], stage=stage)))
            cell['groups']['Multi'].append(dict(status=fitted['multi_status'], reason=None if fitted['multi'] is not None else 'second_stage_incomplete', point=fitted['multi'], diagnostics=dict(common_start_digest=start['digest'], weights=fitted['weights'])))
        if gpu and j % 64 == 0:
            print('YY_QUERY', g.key, j, len(queries), flush=True)
    for cell in cells:
        cell['summaries'] = {name: lib['summarize'](g, rows, 'point') for name, rows in cell['groups'].items()}
    return (cells, starts)


def original_forward_metrics(g, rows):
    from dryy.metrics import summarize_points
    return summarize_points(g, rows, 'point')

def stsp_cell(lib,data,g,queries,scale,gpu=True):
    from e3.coefficients import two_scale_weights
    from e3.operator import EmpiricalSTSP
    from scientific import original_forward_metrics
    if gpu:
        from e3.gpu_operator import GPUEmpiricalSTSP
    pipeline=lib['pipelines'];pipeline.EmpiricalSTSP=GPUEmpiricalSTSP if gpu else EmpiricalSTSP
    weights=two_scale_weights(g.d,g.D,scale/2,scale,.015)
    if weights['status']!='converged':raise ValueError('Original STSP coefficients rejected')
    native=pipeline.run_cell(data,g,scale,queries,weights);native['coefficient_evidence']=weights
    native['weights']=weights['weights'];native['coefficients']=weights['coefficients']
    groups={}
    for name,path,field in (('Single_h1','single','Single_h1'),('Single_h2','single','Single_h2'),('Multi','multi','ENF_Multi')):
        groups[name]=[dict(status=r[field]['status'],reason=r[field]['failure_reason'],
            point=r[field]['point'] if r[field]['status']=='success' else None,
            diagnostics=r[field]['diagnostics']) for r in native[path]['queries']]
    return dict(scale=scale,groups=groups,summaries={k:original_forward_metrics(g,r) for k,r in groups.items()},native=native)

