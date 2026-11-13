# Contributing

This is a research implementation. Changes should make the numerical procedure easier to inspect without silently changing the experiment being reproduced.

## Local checks

Use Python 3.12.9 in a separate development environment:

```bash
python -m pip install -r requirements-dev.txt
python -m black --check manifold_experiments run_paper.py reproduce_all.py smoke_all.py verify_paper.py release_contract.py validate_release.py tools tests
python validate_release.py
python verify_paper.py
python -m pytest -q tests
python -m manifold_experiments --profile smoke --backend cpu --output ./smoke_run --workers 2
```

Use a fresh output directory. Smoke computes actual small fits, but is not evidence for full-study rates. `python tools/export_runtime.py --output NEW_EXTERNAL_DIRECTORY` exports the source/design-only runtime for checking that it has no hidden dependence on historical results.

## Scientific changes

Explain changes to estimators, coefficients, kernels, seed streams, query grids, solver tolerances, reliability rules, dtype, threads, reduction order, or failure handling. Preserve failed outputs and compare old and new results on identical inputs. A changed protocol belongs in an explicitly named new experiment.

The native `src/` and `frozen/` packages under each experiment retain the source versions used by that numerical path. Old internal names and receipt paths support provenance. Avoid mass formatting or namespace deduplication inside them: similar files may implement different population/finite or main/reverse variants.

The public layers are `manifold_experiments/catalog.py`, `protocol.py`, `cli.py`, `worker.py`, and `evaluate.py`. Fitters with overlapping native import namespaces run in separate interpreters. Keep that isolation and separate native result fields from paper display names.

## Manifests and output files

After intentional changes, run `python tools/build_manifests.py`, then repeat checks. Do not hand-edit hashes to conceal changed scientific files. Runs from another source revision must fail their resume check.

Commit source, configurations, tests, and documentation. Do not commit generated runs, observation clouds, credentials, or large historical archives. Paper figure filenames are compatibility identifiers; a changed label must not change a measured value.

Pull requests should explain the problem, resulting behavior, scientific impact, and validation. Do not describe smoke as reproduction of the entire study. Contributions use MIT; retain existing notices.
