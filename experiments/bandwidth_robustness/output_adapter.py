import ast
import copy
import inspect
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def expose_outputs(function):
    before = ast.parse(inspect.getsource(function))
    returns = [n for n in ast.walk(before) if isinstance(n, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Dict):
        raise ValueError('Unexpected original YY return contract')
    if any((isinstance(k, ast.Constant) and k.value == 'single_radius_points' for k in returns[0].value.keys)):
        return function
    after = copy.deepcopy(before)
    ret = next((n for n in ast.walk(after) if isinstance(n, ast.Return)))
    ret.value.keys.append(ast.Constant('single_radius_points'))
    ret.value.values.append(ast.Name('outputs', ast.Load()))
    check = copy.deepcopy(after)
    ret = next((n for n in ast.walk(check) if isinstance(n, ast.Return)))
    ret.value.keys.pop()
    ret.value.values.pop()
    if ast.dump(check, include_attributes=False) != ast.dump(before, include_attributes=False):
        raise ValueError('Original scientific AST altered')
    namespace = dict(function.__globals__)
    exec(compile(ast.fix_missing_locations(after), str(ROOT / 'common.py') + '::output_only', 'exec'), namespace)
    return namespace[function.__name__]

