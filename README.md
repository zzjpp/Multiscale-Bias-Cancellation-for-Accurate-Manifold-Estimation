# Beyond Curvature Bias: Higher-Order Manifold Fitting via Multiscale Cancellation

Source for the experiments in the final **arxiv_final.pdf** (128 pages; SHA-256 in `PAPER_REFERENCE.json`). The supported entry point performs **new input generation, fitting, evaluation and plotting**. It does not copy old figures or load historical fitted points.

## Install and check

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
```

The tested CPU environment is Python 3.12.9 with the exact versions in `requirements.txt`. Use a separate environment for the original Linux GPU dependencies in `requirements-gpu.txt`. GPU mode retains the original CPU/GPU equivalence gates and hardware checks; it never silently falls back to CPU. E4 and E5 retain their original CPU implementations even when other experiments use GPU.

## Run from scratch

Start with the reduced **actual fitting → packet → evaluation → figure** check:

```bash
python run_paper.py --profile smoke --backend cpu --output ./smoke_run --workers 2
```

It runs 24 tasks covering E1–E5 and both nonuniform studies, generates new observations/roots, preserves failed queries, and produces seven diagnostic PNG/PDF pairs plus the counterexample figure. **It changes sample/query/quadrature budgets and is not evidence for the paper's errors or convergence rates.**

Inspect the full plan without starting expensive computations:

```bash
python run_paper.py --experiment all --backend cpu --output ./paper_run --plan-only
```

Then execute that exact plan:

```bash
python run_paper.py --experiment all --backend cpu --output ./paper_run --resume --workers 1
```

Or select an experiment into a fresh directory:

```bash
python run_paper.py --experiment E2 --backend cpu --output ./e2_run --workers 4
python run_paper.py --experiment E3 --backend gpu --output ./e3_run --workers 2
```

`--workers` controls separate interpreters with one BLAS/OpenMP thread each. Use it conservatively. The GPU path requires the original visible device/environment; population GPU tasks include fresh same-input CPU equivalence checks, adding substantial CPU cost. CPU mode explicitly runs the original full-size CPU fitters. The full plan has **1,356 tasks**, many containing multiple scales and hundreds of queries; it is expensive and has no promised runtime.

Existing output directories require `--resume` and an identical source-bound plan. Valid completed tasks are reused; recorded engineering failures are **not automatically retried**. Scientific failures remain in packets and summaries. Add `--evaluate-only` to evaluate a complete new run separately. Sources and publication directories are never overwritten.

## Paper experiment mapping

| Selection | New calculations and selected figures |
|---|---|
| `E1` | STSP/DR/YY population, shared-fit naive/Proposed control, original MS-STSP; Figures 2, 4 and E.1–E.3 |
| `E2` | Three methods, six geometries, nested sizes, 20 repeats; Figure 5 and E.4–E.6 |
| `E3` | Three methods, six geometries, main/reverse, 28 scales; twelve plots, Figure 6 and E.7–E.12 |
| `E4` | Common-query ENF/local CP/Oracle/NN, constructed counterexample; Figure 7 and E.13 |
| `E5` | Population/finite directional ablation; Figure 8, E.14 and Table 2 |
| `nonuniform_E1` | Nonuniform population STSP; E.15–E.16 |
| `nonuniform_E2` | Nonuniform finite-sample STSP only; E.17–E.18. Unreported DR/YY runs are not required. |

Figures 1 and 3 are schematics with editable sources under `illustrations/`. `docs/FIGURE_MAP.json` records original manuscript filenames/pages, not a claim that historical exports are distributed. New figures are under `RUN/figures/`, `RUN/evaluation/E5/figures/`, and `RUN/counterexample/`. Summaries and precision diagnostics are under `RUN/evaluation/`; `EVALUATION.json` records completion or failure. Use `run_paper.py` as the public orchestration interface; original experiment modules are retained as versioned scientific libraries, and their historical deployment/replay CLIs are not the release workflow.

## Scientific contracts

- E1 keeps the seven sigma levels, original deterministic integration, smooth STSP kernel, noise-dependent coefficients and PCA-based ENF. Controlled Single/naive/Proposed share the **same** two fitted points. MS-STSP has its separate original hard-ball operator/iteration and is not the naive control. Integration refinement, root and weight checks precede paper rendering.
- E2 keeps sigma 0.005, bandwidths 0.1/0.2, 144 queries, eight nested prefixes of a 500000-point parent and 20 repeats. The withdrawn population overlay is absent.
- E3 order is circle, S2/R3, Clifford torus, S3/R4, S2/R10, corrugated torus. Sample sizes: `12000/24000/24000/60000/24000/36000`; reverse queries: `1024/8192/8192/16384/8192/1024`; **262144 references on all six geometries**. YY uses `(0.4,1,1.6) r1`. All scales, independent baselines, failures, NaN breaks and unreliable markers remain; green backgrounds are absent. Reverse distances use each complete fitted cloud, never pooled shard errors.
- STSP/DR matching uses compact centered reference-scale PCA; true normals are blocked. Original DR internal Gaussian fitting and failure adapters remain. YY's original geometry-assisted preliminary manifold and data-axis construction remain disclosed.
- E4 regenerates common queries on the empirical fine STSP branch with the original geometry-assisted initialization and seeds. CP searches a continuous **local empirical implicit sheet**, not discrete candidates or a certified global projection over all zero-set components. Fresh timings include common reference construction and target-curve/search costs; they are not a promise to reproduce the historical approximately 26-second timing on other hardware.
- E5 keeps actual `c0=3`, the pilot-selected finite scale triple, the historical rejected condition-number gate and original evidence class. Cleanup does not upgrade that evidence. Directional errors use the original base point and the projected actual first-stage axis.
- Nonuniform densities are relative to surface measure, amplitude 0.5 (max/min ratio 3). Clean area-uniform proposals are rejected **before** Gaussian noise with the original separate acceptance stream. Definitions are in `nonuniform/E1/config.json` and `nonuniform/E2/sampling.py`.

## Inputs, outputs and validation limits

No external training data or private server path is needed. Observations are generated from distributed geometry/seed definitions. E3's exact registered query/reference NPY coordinates are included as fixed design inputs shared across methods, not fitted outputs. `RUN_SOURCE_MANIFEST.json` binds only source/configuration/design inputs. The runtime was also tested in a separate copy containing **no historical numerical outputs**.

Large historical packets, duplicate observation arrays, queue logs and old figure exports are excluded from this revision. Small optional tables under `reference_values/` are **never read by the fresh estimator pipeline**. Older Git commits retain their history; use `--depth 1` to avoid downloading it. Private local scientific archives were preserved during cleanup.

Release checks cover clean installation, protocol consistency and reduced real end-to-end runs. They do **not** claim an independent rerun of the entire 1,356-task production study during packaging. Numerical diagnostics are not certified bounds; finite query maxima are not continuous suprema; finite-range slopes are not asymptotic proofs. Hardware/library differences may affect floating-point values, classifications near thresholds, and timing. The repository is distributed under the MIT License (see `LICENSE` and `LICENSES.md`); existing contributor notices are retained. E1 was relicensed to MIT with the repository owner's explicit authorization on 2026-11-12. No artifact badge is claimed.
