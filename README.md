# Beyond Curvature Bias: Higher-Order Manifold Fitting via Multiscale Cancellation

Python implementation of the paper's experiments. The supplied **2026-11-12 manuscript** is identified by filename and SHA-256 in [PAPER_REFERENCE.json](PAPER_REFERENCE.json). Experiment names, sections, and figure numbers follow that version.

The supported pipeline generates observations, fits estimators, evaluates errors, and draws figures **from scratch**. Historical fits, private servers, and old figure exports are not required.

## Quick start

```bash
git clone --depth 1 https://github.com/zzjpp/Multiscale-Bias-Cancellation-for-Accurate-Manifold-Estimation.git
cd Multiscale-Bias-Cancellation-for-Accurate-Manifold-Estimation
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python validate_release.py
python verify_paper.py
python -m pytest -q tests
python -m manifold_experiments --list
```

Start with the reduced **real fitting → evaluation → plotting** check:

```bash
python -m manifold_experiments --profile smoke --backend cpu --output ./smoke_run --workers 2
```

It runs 24 tasks across all seven selections and produces seven diagnostic PNG/PDF pairs plus the CP counterexample. **Smoke changes sample, query, and integration budgets; it is not evidence for paper errors or rates.**

## Paper experiment mapping

| Selection / compatibility code | Paper heading | Sections | Implementation |
|---|---|---|---|
| `population` / `E1` | Population Verification | 5.1; E.1; E.7 | [population_verification](experiments/population_verification/) |
| `fixed-scales` / `E2` | Finite-Sample Behavior at Fixed Scales | 5.2; E.2 | [finite_sample_fixed_scales](experiments/finite_sample_fixed_scales/) |
| `bandwidth` / `E3` | Bandwidth Robustness and Reverse Coverage | 5.3; E.3 | [bandwidth_robustness](experiments/bandwidth_robustness/) |
| `correspondence` / `E4` | Correspondence Accuracy and Computational Stability | 5.4; E.3.1; E.6.3 | [correspondence_accuracy](experiments/correspondence_accuracy/) |
| `directional-ablation` / `E5` | Directional Ablation of the YY Estimator | 5.5; E.4; E.6.3 | [yy_directional_ablation](experiments/yy_directional_ablation/) |
| `nonuniform-population` / `nonuniform_E1` | Multiscale Bias Cancellation for STSP under Nonuniform Sampling — Population | E.5 | [nonuniform/population](experiments/nonuniform/population/) |
| `nonuniform-fixed-scales` / `nonuniform_E2` | Multiscale Bias Cancellation for STSP under Nonuniform Sampling — Finite Sample | E.5 | [nonuniform/fixed_scales](experiments/nonuniform/fixed_scales/) |

```bash
# Inspect the complete selected plan without fitting.
python -m manifold_experiments --experiment population --output ./population_run --plan-only
# Execute that source-bound plan.
python -m manifold_experiments --experiment population --output ./population_run --resume
# Other examples: use fresh output directories.
python -m manifold_experiments --experiment fixed-scales --output ./fixed_scale_run --workers 4
python -m manifold_experiments --experiment bandwidth --backend gpu --output ./bandwidth_run --workers 2
```

Select `all` for the complete **1,356-task** study. A task can contain multiple scales and hundreds of queries; full runs are expensive. `--workers` launches separate interpreters with one BLAS/OpenMP thread each. Original scientific libraries have overlapping namespaces, so task-level process isolation is intentional.

`run_paper.py` remains a compatibility entrypoint. New topic names and E1–E5 aliases resolve to the same ordered task IDs. Native `Multi` fields denote the paper's **GBC-STSP**, **GBC-DR**, or **GBC-YY**. Figure filenames are retained for existing LaTeX sources.

An existing directory requires `--resume` and an identical source-bound plan. Valid completed packets are reused; engineering failures are not silently retried. Scientific failures, NaNs, and unreliable markers remain. `--evaluate-only` evaluates a complete run without fitting again. A run from an earlier source revision cannot be silently resumed after reorganization.

## Code and results

```text
manifold_experiments/   CLI, paper catalog, job design, workers, evaluation
experiments/           Experiment-specific solvers, configurations, renderers
illustrations/         Editable schematic sources for Figures 1 and 3
docs/                  Figure map, scientific contracts, architecture, QA scope
tests/                 Protocol, naming, source-binding, failure-path regressions
tools/                 Source/design export and manifest maintenance
reference_values/      Optional small comparison tables; never estimator inputs
```

See [the experiment guide](docs/EXPERIMENTS.md), [code guide](docs/CODE_STRUCTURE.md), and [exact figure map](docs/FIGURE_MAP.json). Full-profile figures are written under `RUN/figures/`, `RUN/evaluation/E5/figures/`, and `RUN/counterexample/`; diagnostics and tables are under `RUN/evaluation/`. `EVALUATION.json` records completion or failure. Figures 1 and 3 have editable TikZ sources under `illustrations/`.

## Environments and scope

The tested CPU environment is **Python 3.12.9** with exact versions in [requirements.txt](requirements.txt). The original Linux GPU dependencies are separate in [requirements-gpu.txt](requirements-gpu.txt). GPU mode retains the original hardware/environment and same-input CPU/GPU checks, without silent CPU fallback. Correspondence and directional-ablation retain their CPU implementations. Numerical paths retain float64 arithmetic.

No external dataset is required. Synthetic observations use the original geometry/seed definitions. [Registered query/reference coordinates](experiments/bandwidth_robustness/assets/) are fixed design inputs, not fitted outputs. `RUN_SOURCE_MANIFEST.json` binds source and design inputs; `SOURCE_MANIFEST.json` binds the distribution. See [reproducibility notes](docs/REPRODUCIBILITY.md) for validation levels.

Release validation does not independently rerun the full production study. Hardware/library differences can affect floating-point results, threshold classifications, and timings. Finite-query maxima are not continuous suprema; finite-range slopes are not asymptotic proofs; refinement diagnostics are not certified bounds. YY's prescribed preliminary manifold and geometry-assisted common queries in the correspondence benchmark remain disclosed. Directional finite-sample results retain pilot-selection and historical conditioning caveats.

Large historical outputs are excluded from this revision. Earlier Git commits retain their history; use `--depth 1` to avoid downloading it. Code is under the [MIT License](LICENSE); existing notices are retained (see [LICENSES.md](LICENSES.md)). No artifact-evaluation badge is claimed.

See [CHANGELOG.md](CHANGELOG.md), [CONTRIBUTING.md](CONTRIBUTING.md), and [中文说明](README.zh-CN.md).
