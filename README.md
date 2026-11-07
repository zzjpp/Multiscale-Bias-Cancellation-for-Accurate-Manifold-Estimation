# Beyond Curvature Bias: Higher-Order Manifold Fitting via Multiscale Cancellation

Code, selected numerical results, and figure replay for the **2026-11-07 main manuscript and separate Supplementary Material**. Their exact PDF SHA-256 values are recorded in `PAPER_REFERENCE.json`; the manuscript PDFs themselves are not uploaded.

## Reproduce the current paper figures

Use Python 3.12 in a separate environment:

```bash
python -m pip install -r requirements.txt
python validate_release.py
python verify_paper.py
python -m pytest -q tests --basetemp ./tmp/pytest
python reproduce_all.py --output ./replay_output
```

The output directory must not already exist. The command renders **33 logical experimental plots / 66 PNG and vector exports**, including all 12 E3 panels-of-six figures. Two schematic illustrations have separate editable sources under `illustrations/`. See [every main/supplement figure and its exact source](docs/FIGURE_MAP.md).

This is saved-result replay, not an independent rerun of the large numerical experiments. Numerical values, IQR bands, unreliable markers, undefined cells, all 28 E3 scales and the original failure rules are retained. Font or Matplotlib differences can change rendered bytes without changing numerical values. E1 noise axes decrease from 0.03 to 0.008; fitted slopes use the original numerical sigma values.

## Experiment directories

| Directory | Current paper content | New numerical calculations |
|---|---|---|
| `E1/` | Supplement S1-S3 and normalized circle Figure 4 | Original population packages and `E1/smoke.py` |
| `E1/weight_control/` | Figure 2: shared-fit Single, naive h-squared, coefficient-based Proposed and original MS-STSP | Shared-point postprocessing; independent coefficient/precision checks |
| `E1/ms_stsp/` | Hard-ball MS-STSP population operator, fixed-reference iteration and OLS intercept | `study.py cell --help` |
| `E2/` | Figure 5 and S4-S6: finite sample size | Original `run_fits.py` / `evaluate_fits.py` |
| `E3/` | Figure 6 and S7-S12: bandwidth and reverse coverage | Original `run_fits.py` / `evaluate_fits.py` |
| `E4/correspondence/` | Figure 7 and S13: local continuous CP benchmark and oscillatory counterexample | `finite_experiment.py`, `population_experiment.py`, `counterexample.py` |
| `E5/` | Figure 8, S14 and Table 2: directional ablation | Original E5 raw-run entrypoints |
| `nonuniform/E1/` | S15-S16: nonuniform population STSP | `run_population.py --help` |
| `nonuniform/E2/` | S17-S18: nonuniform finite-sample STSP | Original `parent.py` / `aggregate.py` |

`E7` and standalone obsolete studies are excluded. Corrugated torus remains the sixth geometry, not a separate E7 experiment. E2 has **no population reference-line overlay**. Larger raw archives, queue controllers, connection helpers, SSH material, logs, caches, and unrelated experiments are not part of this repository.

## Numerical and geometric scope

- STSP/DR matching uses compact centered reference-scale PCA; their original kernels and solvers are unchanged. No true normal is substituted into matching.
- Figure 2's three controlled curves share the same archived fits and ENF correspondence. Naive uses `(4/3,-1/3)`. The hard-ball MS-STSP curve is a separate algorithm and not the naive controlled estimator.
- E3 sample sizes in panel order are `12000/24000/24000/60000/24000/36000`; reverse query counts are `1024/8192/8192/16384/8192/1024`, with **262144 references for every geometry**. YY uses the selected `(0.4,1,1.6) r1` version; the 0.8 experiment is not the paper version.
- E4 CP continuously searches an empirical-PCA local implicit STSP sheet. It is not discrete NN and is not a certified global projection over every empirical zero-set component. Common query/branch initialization retains disclosed geometry assistance. Approximately 25.6 seconds is the measured batch median for this implementation, not a universal CP complexity bound.
- YY retains its controlled geometry-based preliminary manifold and original data-axis construction. Cross-method random streams are not claimed to be paired.
- Nonuniform densities are relative to surface measure, with amplitude 0.5 and max/min ratio 3. Finite samples use rejection of clean area-uniform proposals **before** Gaussian noise. Original query grids and failures are retained; default paper replay selects STSP only in this supplementary study.

## Data availability and precision

Selected tables, population packets and CP measurements are distributed. Historical E2/E3 and nonuniform-E2 full raw archives are too large for this repository; exact generation and aggregation code is included. **The 60 historical E5 held-out finite packets are missing from the local source artifact.** The selected Figure 8 exports are bound to their provenance, but raw-data validation of those medians remains incomplete. This limitation is not hidden by figure replay. See [data availability](docs/DATA_AVAILABILITY.md).

GPU production used its original Linux Python/CuPy environment; use `requirements-gpu.txt` separately from the plotting requirements. Full calculations are expensive. Reduced smoke tests check interfaces only and are not production quadrature/accuracy gates. No selected production result was refitted or tuned during cleanup; reduced CPU smoke checks are reported separately.

## Attribution and licensing

Existing notices are retained; see `LICENSES.md`. E1-derived components retain their original all-rights-reserved status. A public GitHub repository does not change those rights. No new blanket license is assigned. Historical source-binding records may contain original compute paths as provenance labels; execution resolves distributed inputs relative to the package, and no authentication material is shipped.
