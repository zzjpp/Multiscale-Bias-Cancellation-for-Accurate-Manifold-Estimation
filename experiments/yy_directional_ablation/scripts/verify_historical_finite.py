"""Verify all recovered E5 packets and the actual median/IQR values in Figure 8.

This reaggregates archived per-query geometric metrics. It does not rerun fits
or independently recompute truth distances from observation arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

if __package__:
    from .plot_historical_finite import METHODS, COLORS, read_packets, summarize, write_csv
else:
    from plot_historical_finite import METHODS, COLORS, read_packets, summarize, write_csv

ROOT = Path(__file__).resolve().parents[1]
NS = {'s': 'http://www.w3.org/2000/svg'}
SVG_COORDINATE_TOLERANCE = 2e-6  # SVG coordinates are serialized to six decimals (points).


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_points(value: str) -> np.ndarray:
    if re.sub(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|[MLZmlz]|\s|,', '', value):
        raise ValueError('Unsupported selected SVG path command')
    numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', value)
    if len(numbers) % 2:
        raise ValueError('Malformed selected SVG path')
    return np.asarray(numbers, dtype=float).reshape(-1, 2)


def check_selected_svg(summary: list[dict], svg: Path) -> dict:
    """Compare data against vector paths, using the original plotting transform.

    No transform is fitted to the median values. Axis rectangles are read from
    the selected SVG; the documented original log autoscale and 5% margins
    determine the transform independently from all q1/q3 data limits.
    """
    tree = ET.parse(svg).getroot()
    height = float(tree.attrib['viewBox'].split()[3])
    axes = {g.get('id'): g for g in tree.findall('.//s:g', NS)
            if g.get('id') in {'axes_1', 'axes_2'}}
    if set(axes) != {'axes_1', 'axes_2'}:
        raise ValueError('The selected finite E5 figure must have two axes')
    deviations = []
    for index, metric in enumerate(('E_inf', 'E_avg'), 1):
        axis = axes[f'axes_{index}']
        rectangles = [g for g in axis if g.get('id', '').startswith('patch_')]
        background = rectangles[0].find('s:path', NS)
        bounds = path_points(background.attrib['d'])
        left, top = bounds.min(axis=0)
        right, bottom = bounds.max(axis=0)
        block = [r for r in summary if r['metric'] == metric]
        minimum = min(r['q1'] for r in block)
        maximum = max(r['q3'] for r in block)
        log_lo, log_hi = np.log10([minimum, maximum])
        span = log_hi - log_lo
        log_lo -= .05 * span
        log_hi += .05 * span
        xs = left + (np.arange(3) + .1) / 2.2 * (right - left)

        def transform(values):
            ys = bottom - (np.log10(values) - log_lo) / (log_hi - log_lo) * (bottom - top)
            return np.column_stack((xs, ys))

        medians, bands = {}, {}
        for group in axis:
            name = group.get('id', '')
            if name.startswith('line2d_'):
                line = group.find('s:path', NS)
                if line is None:
                    continue
                for method in METHODS:
                    if 'stroke: ' + COLORS[method].lower() + ';' in line.get('style', '').lower():
                        medians[method] = path_points(line.attrib['d'])
            elif 'PolyCollection_' in name:
                use = group.find('.//s:use', NS)
                path = group.find('.//s:path', NS)
                if use is None or path is None or use.get('transform'):
                    raise ValueError('Unexpected IQR vector representation')
                for method in METHODS:
                    if 'fill: ' + COLORS[method].lower() + ';' in use.get('style', '').lower():
                        points = path_points(path.attrib['d'])
                        points += [float(use.get('x', '0')), float(use.get('y', str(height)))]
                        bands[method] = points
        if set(medians) != set(METHODS) or set(bands) != set(METHODS):
            raise ValueError('Missing or ambiguous median/IQR curves in selected SVG')
        for method in METHODS:
            cells = sorted((r for r in block if r['method'] == method), key=lambda r: r['sample_size'])
            median = transform([r['median'] for r in cells])
            lower = transform([r['q1'] for r in cells])
            upper = transform([r['q3'] for r in cells])
            polygon = np.vstack((upper[:1], lower, upper[-1:], upper[::-1]))
            for observed, expected, label in [(medians[method], median, 'median'),
                                              (bands[method], polygon, 'IQR')]:
                if observed.shape != expected.shape:
                    raise ValueError(f'Incorrect {label} path shape: {metric}/{method}')
                error = float(np.max(np.abs(observed - expected)))
                if error > SVG_COORDINATE_TOLERANCE:
                    raise ValueError(f'Selected SVG disagrees with raw {label}: {metric}/{method}: {error}')
                deviations.append(error)
    return dict(status='MATCH',median_values=24,quartile_boundary_values=48,
                maximum_vector_coordinate_difference=max(deviations),
                SVG_coordinate_tolerance=SVG_COORDINATE_TOLERANCE,
                SVG_sha256=sha(svg),
                interpretation='All medians and both IQR boundaries match the selected vector figure within SVG serialization rounding.')


def audit(data_root: Path | None = None, svg: Path | None = None) -> dict:
    data_root = (data_root or ROOT / 'data/finite').resolve()
    svg = svg or ROOT / 'paper_figures/figure_e5_finite_sample_dual_metrics.svg'
    availability = json.loads((ROOT / 'data/finite_packet_availability.json').read_bytes())
    config = json.loads((ROOT / 'config/finite_sample.json').read_bytes())
    expected = availability['expected_files']
    wanted = {f'packets/e5_directional_n{n:07d}_r{repeat:03d}.json'
              for n in config['sample_sizes'] for repeat in range(config['repeats'])}
    observed = {p.relative_to(data_root).as_posix() for p in (data_root / 'packets').glob('e5_directional_n*_r*.json')}
    if set(expected) != wanted or observed != wanted or len(wanted) != 60:
        raise ValueError('Incomplete or unexpected historical E5 packet grid')
    historical_config = sha(data_root / 'original_config.json')
    packet_hashes = {}
    source_hashes = set()
    failures = 0
    for name in sorted(wanted):
        path = data_root / name
        record = expected[name]
        if path.stat().st_size != record['size'] or sha(path) != record['sha256']:
            raise ValueError('Changed historical E5 packet: ' + name)
        packet = json.loads(path.read_bytes())
        metadata, result = packet['metadata'], packet['result']
        n, repeat = result['sample_size'], result['repeat']
        if name != f'packets/e5_directional_n{n:07d}_r{repeat:03d}.json':
            raise ValueError('Packet identity/filename mismatch')
        if metadata['config_sha256'] != historical_config:
            raise ValueError('Historical config binding mismatch')
        source_hashes.add(metadata['source_sha256'])
        if result['sigma'] != config['sigma'] or result['kappas'] != config['kappas']:
            raise ValueError('Historical experiment settings changed')
        queries = result['query_rows']
        if len(queries) != 256 or sorted(q['query_index'] for q in queries) != list(range(256)):
            raise ValueError('Incomplete or duplicated historical query grid')
        failures += result['failure_count']
        if result['failure_count'] or result['failures'] or result['retained_count'] != 256:
            raise ValueError('Historical failure accounting differs; do not drop failures')
        for method in METHODS:
            values = np.asarray([q['metrics'][method]['geometric'] for q in queries])
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Nonfinite or negative query error')
            summary = result['summaries'][method]
            if not np.isclose(values.max(), summary['geometric_max'], rtol=1e-13, atol=1e-15):
                raise ValueError('Stored repeat maximum disagrees with queries')
            if not np.isclose(values.mean(), summary['geometric_mean'], rtol=1e-13, atol=1e-15):
                raise ValueError('Stored repeat mean disagrees with queries')
        packet_hashes[name] = record['sha256']
    if len(source_hashes) != 1:
        raise ValueError('Mixed historical producer sources')
    repeats = read_packets(data_root / 'packets')
    summary = summarize(repeats)
    plot_check = check_selected_svg(summary, svg)
    return dict(schema='E5_RECOVERED_FINITE_STATISTICS_AUDIT_V1', status='PASS',
                packets=60, repeats_per_sample_size=20, sample_sizes=config['sample_sizes'],
                queries_per_packet=256, archived_estimator_query_outcomes=60*256*len(METHODS),
                total_query_failures=failures, per_repeat_method_rows=len(repeats),
                summary_rows=len(summary), selected_plot_check=plot_check,
                original_config_sha256=historical_config,
                historical_producer_source_sha256=next(iter(source_hashes)),
                packet_sha256=packet_hashes, summaries=summary,
                original_pilot_selection_and_conditioning_caveats_retained=True,
                new_fits=False, truth_distances_recomputed=False,
                evidence='Reaggregation of original recorded query metrics and comparison to the actual selected Figure 8 vector paths.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Fresh directory for audit and reaggregated CSVs')
    args = parser.parse_args()
    report = audit()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
        write_csv(args.output/'summary_dual_metrics.csv',report['summaries'])
        write_csv(args.output/'repeat_values_dual_metrics.csv',read_packets(ROOT/'data/finite/packets'))
    print(json.dumps({k:report[k] for k in ['status','packets','archived_estimator_query_outcomes','summary_rows','selected_plot_check']},indent=2))


if __name__=='__main__':main()
