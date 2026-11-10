"""Presentation-only; no green background, all curves/failures retained."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
GRID = (0.06, 0.075, 0.09, 0.11, 0.135, 0.165, 0.2, 0.245, 0.3, 0.37, 0.45, 0.55, 0.67, 0.82, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 5.0)
GEOMETRIES = (('circle_r2', '$S^1\\subset\\mathbb{R}^2$'), ('sphere2_r3', '$S^2\\subset\\mathbb{R}^3$'), ('clifford_torus_r4', 'Clifford torus $\\subset\\mathbb{R}^4$'), ('sphere3_r4', '$S^3\\subset\\mathbb{R}^4$'), ('sphere2_r10', '$S^2\\subset\\mathbb{R}^{10}$'), ('corrugated_torus_r3', 'Corrugated torus $\\subset\\mathbb{R}^3$'))
GROUPS = {'STSP': ('Single_h1', 'Single_h2', 'Multi'), 'DR': ('Single_h1', 'Single_h2', 'Multi'), 'YY': ('Single_0p4r1', 'Single_r1', 'Single_1p6r1', 'Multi')}
METRICS = {'main': ('E_avg', 'E_infty'), 'rev': ('E_reverse_mean', 'E_reverse')}
COLORS = {'Single_h1': '#882255', 'Single_h2': '#004488', 'Single_0p4r1': '#882255', 'Single_r1': '#004488', 'Single_1p6r1': '#117733', 'Multi': '#D55E00'}
MARKERS = {'Single_h1': '^', 'Single_h2': 'o', 'Single_0p4r1': '^', 'Single_r1': 'o', 'Single_1p6r1': 'D', 'Multi': 's'}
STYLES = {'Single_h1': ':', 'Single_h2': '-', 'Single_0p4r1': ':', 'Single_r1': '-', 'Single_1p6r1': '-.', 'Multi': '--'}
LABELS = {'Single_h1': 'Single ($h_1$)', 'Single_h2': 'Single ($h_2$)', 'Single_0p4r1': 'Single ($0.4r_1$)', 'Single_r1': 'Single ($r_1$)', 'Single_1p6r1': 'Single ($1.6r_1$)', 'Multi': 'Multi'}
MAIN_COUNTS = {g: 512 if g == 'corrugated_torus_r3' else 256 for g, _ in GEOMETRIES}
REV_COUNTS = dict(zip(MAIN_COUNTS, (1024, 8192, 8192, 16384, 8192, 1024)))
REF_COUNTS = {g: 65536 if g == 'corrugated_torus_r3' else 262144 for g in MAIN_COUNTS}
Y_LABELS = {'E_avg': 'Average geometric error $E_{\\mathrm{avg}}$', 'E_infty': 'Maximum geometric error $E_\\infty$', 'E_reverse_mean': 'Mean reverse coverage error $\\overline{E}_{\\rm rev}$', 'E_reverse': 'Reverse coverage error $E_{\\rm rev}$'}

def require(flag, message):
    if not flag:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def js(path):
    return json.loads(Path(path).read_bytes())

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def boolean(value):
    if isinstance(value, bool):
        return value
    require(value in ('True', 'False'), 'Strict original Boolean required')
    return value == 'True'

def identity(row):
    return tuple((row[n] for n in ('method', 'phase', 'geometry', 'estimator', 'scale', 'metric')))

def parse_rows(raw):
    expected = {(m, p, g, e, s, t) for m in GROUPS for p in METRICS for g, _ in GEOMETRIES for e in GROUPS[m] for s in GRID for t in METRICS[p]}
    seen = set()
    rows = []
    for source in raw:
        r = dict(source)
        for name in ('scale', 'median', 'q1', 'q3', 'query_failure_rate', 'sigma'):
            r[name] = float(r[name])
        for name in ('complete_repeats', 'total_repeats', 'query_count', 'reference_count', 'n', 'query_failure_count'):
            r[name] = int(r[name])
        r['reliable'] = boolean(r['reliable'])
        key = identity(r)
        require(key in expected and key not in seen, 'Unexpected/duplicate prescribed cell')
        seen.add(key)
        g, p = (r['geometry'], r['phase'])
        repeats = 20 if p == 'main' else 1
        count = (MAIN_COUNTS if p == 'main' else REV_COUNTS)[g]
        require((r['n'], r['sigma'], r['query_count'], r['reference_count'], r['total_repeats']) == (N[g], 0.015, count, REF_COUNTS[g], repeats), 'Original authorized design changed')
        failures = r['query_failure_count']
        complete = r['complete_repeats']
        require(0 <= failures <= count * repeats and 0 <= complete <= repeats, 'Invalid counts')
        require(r['query_failure_rate'] == failures / (count * repeats), 'Failure denominator changed')
        require(repeats - complete <= failures <= count * (repeats - complete), 'Incomplete repeats/failures inconsistent')
        reliable = complete >= 18 and r['query_failure_rate'] <= 0.01 if p == 'main' else r['query_failure_rate'] <= 0.01
        require(r['reliable'] == reliable, 'Original reliability gate changed')
        taxonomy = json.loads(r['failure_taxonomy_json'])
        require(isinstance(taxonomy, dict) and all((isinstance(k, str) and k and isinstance(v, int) and (not isinstance(v, bool)) and (v > 0) for k, v in taxonomy.items())) and (sum(taxonomy.values()) == failures), 'Full failure taxonomy omitted')
        values = [r[name] for name in ('q1', 'median', 'q3')]
        missing = all((math.isnan(v) for v in values))
        require(missing or (all((math.isfinite(v) and v >= 0 for v in values)) and values == sorted(values)), 'Undefined metrics imputed or quartiles malformed')
        require(missing == (failures == count * repeats), 'Missing/nonfinite outcomes accounting changed')
        if p == 'rev':
            require(values[0] == values[1] == values[2] or missing, 'Invented reverse repeat bands')
        rows.append(r)
    require(seen == expected and len(rows) == 6720, 'Complete6720 three-method cells required')
    lookup = {identity(r): r for r in rows}
    for r in rows:
        if r['metric'] != METRICS[r['phase']][0]:
            continue
        partner = lookup[identity(r)[:-1] + (METRICS[r['phase']][1],)]
        for field in ('complete_repeats', 'query_failure_count', 'query_failure_rate', 'reliable', 'failure_taxonomy_json'):
            require(r[field] == partner[field], 'Mean/max outcome populations disagree')
        for field in ('q1', 'median', 'q3'):
            require(math.isnan(r[field]) or r[field] <= partner[field] + 1e-12, 'Mean exceeds maximum')
    return rows

def favorable_runs(rows, method, phase, geometry, metric):
    lookup = {(r['estimator'], r['scale']): r for r in rows if (r['method'], r['phase'], r['geometry'], r['metric']) == (method, phase, geometry, metric)}
    flags = []
    for scale in GRID:
        cells = [lookup[e, scale] for e in GROUPS[method]]
        flags.append(all((r['reliable'] and math.isfinite(r['median']) and (r['median'] > 0) for r in cells)) and lookup['Multi', scale]['median'] < min((lookup[e, scale]['median'] for e in GROUPS[method][:-1])))
    runs = []
    start = None
    for i, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= 2:
                runs.append((GRID[start], GRID[i - 1]))
            start = None
    return runs

def series_counts(series):
    finite = [math.isfinite(r['median']) and r['median'] > 0 for r in series]
    return dict(total=len(series), finite_positive=sum(finite), finite_hollow=sum((v and (not r['reliable']) for r, v in zip(series, finite))), undefined_nonpositive=sum((not v for v in finite)), reliable=sum((r['reliable'] for r in series)))

def stem(method, phase, metric):
    suffix = 'all_single_scales_' + ('average_error' if metric == 'E_avg' else 'max_error') if phase == 'main' else metric
    return f'figure_e3_{method.lower()}_{suffix}'

def draw(rows, output, anchor):
    import numpy as np
    os.environ['SOURCE_DATE_EPOCH'] = '0'
    os.environ['MPLBACKEND'] = 'Agg'
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    require(platform.python_version().startswith('3.12.') and np.__version__ == '2.4.4' and (mpl.__version__ == '3.11.1'), 'Pinned local figure-only environment required')
    require(len(rows) == 6720 and len({identity(r) for r in rows}) == 6720, 'Complete validated table required')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    products = []
    spans = {}
    counts = {}
    for method in GROUPS:
        for phase in METRICS:
            for metric in METRICS[phase]:
                mpl.rcdefaults()
                mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10.5, 'legend.fontsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': 0.9, 'pdf.fonttype': 42, 'ps.fonttype': 42})
                fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.0), sharex=True)
                for axis, (g, title) in zip(axes.flat, GEOMETRIES):
                    run = favorable_runs(rows, method, phase, g, metric)
                    spans[':'.join((method, phase, g, metric))] = run
                    for e in GROUPS[method]:
                        series = sorted((r for r in rows if (r['method'], r['phase'], r['geometry'], r['metric'], r['estimator']) == (method, phase, g, metric, e)), key=lambda r: r['scale'])
                        require(len(series) == 28, 'Every independent curve needs all 28 scales')
                        x = np.array([r['scale'] for r in series])
                        y = np.array([r['median'] for r in series])
                        valid = np.isfinite(y) & (y > 0)
                        bad = valid & ~np.array([r['reliable'] for r in series], bool)
                        z = 7 if e in ('Single_h1', 'Single_0p4r1') else 3
                        axis.plot(x, np.where(valid, y, np.nan), color=COLORS[e], marker=MARKERS[e], linestyle=STYLES[e], linewidth=1.65, markersize=4.0, markerfacecolor=COLORS[e], markeredgecolor=COLORS[e], markeredgewidth=0.8, zorder=z)
                        axis.scatter(x[bad], y[bad], s=29, marker=MARKERS[e], facecolors='white', edgecolors=COLORS[e], linewidths=1.1, zorder=z + 1)
                        counts[':'.join((method, phase, g, metric, e))] = series_counts(series)
                    axis.set_xscale('log')
                    axis.set_yscale('log')
                    axis.set_xlim(0.055, 5.35)
                    axis.set_xticks((0.06, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0), ('.06', '.1', '.2', '.5', '1', '2', '5'))
                    axis.minorticks_off()
                    axis.grid(False)
                    axis.tick_params(direction='out', length=3, width=0.8, pad=2.0)
                    axis.spines[['top', 'right']].set_visible(False)
                    axis.set_title(title, pad=6)
                axes[0, 0].set_ylabel(Y_LABELS[metric])
                axes[1, 0].set_ylabel(Y_LABELS[metric])
                fig.supxlabel('Parent bandwidth $h_2$ ($h_1=h_2/2$)' if method in ('STSP', 'DR') else 'Central transverse radius $r_1$', y=0.086 if phase == 'rev' else 0.045, fontsize=11)
                if phase == 'rev':
                    fig.text(0.5, 0.049, '$n$: 12000 / 24000 / 24000 / 60000 / 24000 / 36000; $N_q$: 1024 / 8192 / 8192 / 16384 / 8192 / 1024', ha='center', fontsize=8.0)
                    fig.text(0.5, 0.014, 'Reference: 262144 (first five), 65536 (E7); fixed repeat: 0', ha='center', fontsize=8.0)
                else:
                    fig.text(0.5, 0.009, '$n$: 12000 / 24000 / 24000 / 60000 / 24000 / 36000; 20 repeats; $N_q=256$ (first five), 512 (E7); finite-repeat median', ha='center', fontsize=8.0)
                handles = [Line2D([], [], color=COLORS[e], marker=MARKERS[e], linestyle=STYLES[e], linewidth=1.65, markersize=5, label='GBC' if e == 'Multi' else LABELS[e].replace('Single', method, 1)) for e in GROUPS[method]]
                handles.append(Line2D([], [], linestyle='none', marker='o', markerfacecolor='white', markeredgecolor='#333333', markersize=5, label='Unreliable'))
                fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=len(handles), frameon=False, handlelength=1.8, columnspacing=1.0)
                fig.subplots_adjust(left=0.095, right=0.985, bottom=0.21 if phase == 'rev' else 0.16, top=0.865, wspace=0.3, hspace=0.49)
                for ext in ('png', 'pdf'):
                    path = output / (stem(method, phase, metric) + '.' + ext)
                    fig.savefig(path, dpi=400, facecolor='white', **{'metadata': {'CreationDate': None, 'ModDate': None}} if ext == 'pdf' else {})
                    products.append(path)
                plt.close(fig)
    report = dict(schema='E3_THREE_METHODS_UNIFIED_N_TWELVE_NO_GREEN_FIGURES_V1', status='EXPORTS_CREATED_ACTUAL_VISUAL_REVIEW_PENDING', logical_figures=12, layout='3 columns x 2 rows', table_rows=6720, renderer_sha256=sha(__file__), n_by_geometry={g: N[g] for g, _ in GEOMETRIES}, latest_user_prescribed_n_by_geometry=True, outputs={p.name: sha(p) for p in products}, anchors=anchor, series_counts=counts, common_reliable_improvement_intervals_not_shaded=spans, all_28_scales_and_independent_baselines_retained=True, full_science_rerun=False, new_fits=False, actual_visual_review=False, formal_overwrite=False, median_over_finite_repeats_not_complete_only=True, reverse_from_full_merged_cloud_not_pooled=True, controlled_known_geometry_queries_initialization_and_evaluation=True, DR_matching_normal='compact centered exact h1-PCA', YY_axis='original data-derived YY axis', independent_YY_main_V2_reverse_V1_sources_preserved=True)
    with (output / 'FIGURE_MANIFEST.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    return report
from design import N
