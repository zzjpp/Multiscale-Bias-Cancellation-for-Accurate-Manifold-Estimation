"""Figure-only E3 replay: remove the three original footer text calls, nothing else."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import types

CORE_SHA = 'c55f7f1b5c534d7e393a36772cc1ee2bacd9e2012db181509407314a3803a5a9'
DESIGN_SHA = '9b48fc8b2f05c38d68429094ae81b4da3d6f4ff6af372b617557e4c648435b9c'
ROWS_SHA = '4587a5f034b03f18580409e5adda3eff969c3c465cb1c5d6bd380aca0c24b3d6'
ACCEPTANCE_SHA = 'c1245b893c99d51c24efaf5ceb7ce63915895fefbb81fe91f28691c1dca7ab38'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(flag, message):
    if not flag:
        raise ValueError(message)


def without_footers(tree):
    """Delete one exact drawing-only If; retain every other AST node verbatim."""
    result = copy.deepcopy(tree)
    draw = next(n for n in result.body if isinstance(n, ast.FunctionDef) and n.name == 'draw')
    matches = []
    for node in ast.walk(draw):
        if not isinstance(node, ast.If) or len(node.body) != 2 or len(node.orelse) != 1:
            continue
        calls = node.body + node.orelse
        if all(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
               and isinstance(n.value.func, ast.Attribute)
               and isinstance(n.value.func.value, ast.Name)
               and n.value.func.value.id == 'fig' and n.value.func.attr == 'text'
               for n in calls):
            matches.append(node)
    require(len(matches) == 1, 'Exactly one original three-call footer block required')
    footer = matches[0]
    texts = [n.value.args[2].value for n in footer.body + footer.orelse]
    require(texts == [
        '$n$: 12000 / 24000 / 24000 / 60000 / 24000 / 36000; $N_q$: 1024 / 8192 / 8192 / 16384 / 8192 / 1024',
        'Reference: 262144 (first five), 65536 (E7); fixed repeat: 0',
        '$n$: 12000 / 24000 / 24000 / 60000 / 24000 / 36000; 20 repeats; $N_q=256$ (first five), 512 (E7); finite-repeat median',
    ], 'Only the selected formal E3 footers may be removed')
    class Remove(ast.NodeTransformer):
        def visit_If(self, node):
            if node is footer:
                return None
            return self.generic_visit(node)
    result = Remove().visit(result)
    require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == 'text' for n in ast.walk(next(
                        x for x in result.body if isinstance(x, ast.FunctionDef) and x.name == 'draw'))),
            'Unexpected remaining figure text')
    return ast.fix_missing_locations(result), texts


def load_core(root):
    design_path = root/'original_design.py'
    core_path = root/'original_figure_core.py'
    require(sha(design_path) == DESIGN_SHA and sha(core_path) == CORE_SHA,
            'Frozen source changed')
    design = types.ModuleType('original_e3_design')
    design.__file__ = str(design_path)
    exec(compile(design_path.read_text(encoding='utf-8'), str(design_path), 'exec'), design.__dict__)
    original = ast.parse(core_path.read_text(encoding='utf-8'))
    modified, removed = without_footers(original)
    modified.body = [n for n in modified.body
                     if not (isinstance(n, ast.ImportFrom) and n.module == 'design')]
    core = types.ModuleType('e3_no_footer_presentation')
    core.__file__ = str(core_path)
    core.N = design.N
    exec(compile(modified, str(core_path)+'::three_footer_calls_removed', 'exec'), core.__dict__)
    return core, removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-sha', required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    require(sha(root/'REPLAY_MANIFEST.json') == args.manifest_sha, 'External manifest anchor required')
    manifest = json.loads((root/'REPLAY_MANIFEST.json').read_bytes())
    require(manifest['schema'] == 'E3_NO_FOOTER_FIGURE_ONLY_REPLAY_V1', 'Wrong presentation schema')
    actual = {p.name for p in root.iterdir() if p.is_file()}
    require(actual == set(manifest['members']) | {'REPLAY_MANIFEST.json'}, 'Replay inventory changed')
    for name, record in manifest['members'].items():
        require(Path(name).name == name and not (root/name).is_symlink()
                and sha(root/name) == record['sha256'], 'Replay member changed: '+name)
    require(sha(root/'unified_rows.csv') == ROWS_SHA
            and sha(root/'original_acceptance.json') == ACCEPTANCE_SHA, 'Actual accepted table required')
    acceptance = json.loads((root/'original_acceptance.json').read_bytes())
    receipt = json.loads((root/'original_numerical_receipt.json').read_bytes())
    prior = json.loads((root/'original_figure_manifest.json').read_bytes())
    require(acceptance['numerical_report_sha256'] == sha(root/'original_numerical_receipt.json')
            and acceptance['figure_manifest_sha256'] == sha(root/'original_figure_manifest.json')
            and receipt['outputs']['unified_rows.csv']['sha256'] == ROWS_SHA
            and acceptance['anchors'] == prior['anchors'] == receipt['anchors'], 'Original acceptance chain changed')
    core, removed = load_core(root)
    rows = core.parse_rows(core.read_csv(root/'unified_rows.csv'))
    if args.verify_only:
        print(json.dumps(dict(rows=len(rows), footer_calls_removed=len(removed), new_fits=False,
                              science_repeated=False, original_failures_preserved=True)))
        return
    if args.output is None:
        parser.error('--output required unless --verify-only')
    require(not args.output.resolve().is_relative_to(root), 'Output must be outside replay inputs')
    report = core.draw(rows, args.output, prior['anchors'])
    require(report['series_counts'] == prior['series_counts'], 'Curve/failure populations changed')
    require(json.dumps(report['common_reliable_improvement_intervals_not_shaded'], sort_keys=True)
            == json.dumps(prior['common_reliable_improvement_intervals_not_shaded'], sort_keys=True),
            'Improvement intervals changed')
    proof = dict(schema='E3_FOOTER_ONLY_PRESENTATION_CHANGE_V1', logical_figures=12, exports=24,
        old_core_sha256=CORE_SHA, overlay_sha256=sha(__file__), unchanged_table_sha256=ROWS_SHA,
        original_acceptance_sha256=ACCEPTANCE_SHA, deleted_text=removed,
        all_other_source_AST_unchanged=True, axes_positions_and_axis_labels_unchanged=True,
        original_series_counts_and_reliability_unchanged=True, new_fits=False,
        science_repeated=False, actual_visual_review=False,
        original_scientific_acceptance_retained_not_reissued=True)
    (args.output/'PRESENTATION_CHANGE.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
    print(json.dumps(dict(exports=24, logical_figures=12, removed_footer_lines=True,
                          new_fits=False, science_repeated=False)))


if __name__ == '__main__':
    main()
