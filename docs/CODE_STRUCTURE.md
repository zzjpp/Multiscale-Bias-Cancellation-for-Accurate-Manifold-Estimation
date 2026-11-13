# Code guide

## Public workflow

`python -m manifold_experiments` and the compatibility `run_paper.py` invoke the same CLI.

| Module | Responsibility |
|---|---|
| `manifold_experiments/catalog.py` | Paper headings, sections, figures, public names, stable experiment codes |
| `manifold_experiments/protocol.py` | Ordered scientific grid and canonical task IDs; paper/smoke budgets stay distinct |
| `manifold_experiments/cli.py` | Source-bound plans, run ownership, isolated processes, resume checks |
| `manifold_experiments/worker.py` | One task per interpreter; original version-specific fitters |
| `manifold_experiments/evaluate.py` | Complete-packet checks, native metrics, numerical gates, reliability, rendering |
| `release_contract.py` | Current source/design validation |
| `verify_paper.py` | Manuscript protocol and design-input checks; does not rerun the study |

Process isolation is intentional. Native libraries share package names (`dryy`, `e1_theory_path`, etc.) while implementing different scientific versions. One interpreter per task prevents importing another experiment's implementation.

## Experiment implementations

See [EXPERIMENTS.md](EXPERIMENTS.md) for paper headings and configurations. Each public experiment directory contains adapters/renderers and its version-specific `src/` or `frozen/` numerical packages. Historical deployment/replay CLIs are not the supported fresh-run workflow.

The population packages `theory_path_population_all_e1` and `theory_path_population_six_manifolds` retain their original internal names. The worker selects the former for the first five geometries and the latter for corrugated torus; YY packages are separate. These are provenance identifiers, not extra public experiments. `e56v2` is the internal directional-ablation library, not a separate E6 experiment.

Native `Multi` fields mean method-specific GBC. Keeping these fields avoids changing packet interfaces or confusing independent single-scale fits with matched scale outputs. The controlled comparison distinguishes Single, naive squared-bandwidth extrapolation, GBC-STSP, and original hard-ball MS-STSP.

## Outputs

```text
RUN/
  PLAN.json                 Selected tasks, profile, backend, source identity
  packets/                  Fresh fits, including failed queries
  logs/                     Per-task logs
  errors/                   Engineering failures; never silently retried
  evaluation/E1…E5/         Tables, diagnostics, numerical audits
  figures/                  Full-profile paper figures
  evaluation/E5/figures/    Directional-ablation exports
  counterexample/           Constructed normal-graph result
  smoke_figures/            Reduced diagnostics only
  EVALUATION.json           Declared completion/failure and scope
```

`RUNNING.json` records active ownership. Do not remove another process's lock or reuse its directory for another plan. Valid packets resume only under the identical source and plan. Reorganization changes the source identity even if its numerical comparison is identical.

`docs/FIGURE_MAP.json` separates manuscript include paths from fresh generated paths. Figure filenames stay unchanged. Optional `reference_values/` tables and historical receipts are not estimator inputs.

## Maintenance

`docs/SCIENTIFIC_PRESERVATION.json` retains original source identifiers/hashes and current locations. `docs/REFACTOR_VALIDATION.json` records actual naming/layout regression scope. Neither substitutes for full scientific validation on a new production run.

The formatter targets public orchestration, tests, and maintenance tools. Scientific constants, RNG streams, arithmetic order, tolerances, and failure semantics remain in the original numerical implementations. See [CONTRIBUTING.md](../CONTRIBUTING.md).
