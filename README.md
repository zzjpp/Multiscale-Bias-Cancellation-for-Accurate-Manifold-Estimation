# Simulation experiments (E1--E7)

This repository contains the seven simulation releases E1--E7 for the
manifold-fitting paper, plus separate query-density and reverse-coverage
diagnostics. Each E1--E7 release has a frozen configuration, pinned
dependencies, a documented entrypoint, and a functional test. The auxiliary
diagnostic directories are additional working artifacts and have different
provenance and release checks.

[`RELEASE_AUDIT.md`](RELEASE_AUDIT.md) records the earlier E1--E6 release audit;
it does not certify the subsequently added E7 and auxiliary directories.

## Scope and evidence

| Experiment | Question | Released output | Primary command |
|---|---|---|---|
| `E1` | Population error and empirical scale slopes for STSP, DR, and YY | Six main/appendix figures | `cd E1 && python reproduce.py` |
| `E2` | Finite-sample error versus sample size on five manifolds | Six main/appendix figures | `cd E2 && python reproduce.py` |
| `E3` | Robustness over bandwidth `h` or YY radius `r1` at `n=12000` | Six method/metric figures | `cd E3 && python reproduce.py --mode formal --workers 4 --gpu-workers 4` |
| `E4` | Oracle/ENF/NN finite-sample correspondence comparison | One two-panel figure | `cd E4 && python run.py` |
| `E5` | Scalar versus directional extrapolation | Two figures | `cd E5 && python reproduce.py --mode formal --workers 4` |
| `E6` | Sensitivity to noise-level misspecification | Three figures | see `E6/README.md` |
| `E7` | Corrugated-torus stress test | Three two-panel figures | `cd E7 && python reproduce.py` |

The supplementary reverse-coverage figures and query-density checks use
`E3_Erev_gpu_pilot/`, `E3_Erev_high_density/`, and
`E7_query_density_extension/`. Their READMEs specify which runs are repeat-0
diagnostics and which inputs come from audited production packets. They are not
additional 20-repeat E3 or E7 releases.

Important evidence qualifications are retained in the experiment READMEs:

- E1 and E2 redraw audited aggregate outputs; they do not rerun the expensive
  population/Monte-Carlo jobs.
- E3 formal DR/YY production requires a CUDA 13 environment. The CPU smoke
  path exercises all three method contracts without claiming formal evidence.
- E4 is the registered 32-query equal-density correspondence example.
- E5 now runs the paper's finite-sample protocol with explicit `c0=3` and a
  revised complementary-weight condition-number limit of 256. The original
  limit of 50 failed and remains in the provenance record. The revision was
  made after seeing that failure; a passing full audit is evidence for the
  disclosed revised protocol, not for the original gate.
- E6 ships audited population summaries for quick plotting and a separate full
  CPU recomputation path.
- E7 redraws audited aggregate outputs; its raw numerical production is not
  included in the E7 plotting release.

No failed or unreliable cells have been silently removed. Refer to each
experiment README before interpreting its figures.

## Environments

The pinned environments are not numerically identical across E1--E7. In
particular, E3 and E6 freeze different NumPy/SciPy/Matplotlib versions. Use one
virtual environment per experiment rather than forcing a single merged lock:

```bash
cd E2                         # replace with E1, ..., E7
python -m venv .venv
python -m pip install -r requirements.txt
python -m pytest -q
```

For E3 GPU production, additionally install
`E3/requirements-gpu-cu13.txt` in its E3 environment.

The top-level `requirements.txt` contains only the pinned test dependency for
the repository-contract check; it is not a substitute for the seven experiment
lock files.

## Fast repository check

`validate_release.py` is an older whole-tree static audit. It currently fails:
its directory contract predates E7, and the auxiliary directories contain
server scripts, path references, and generated logs that violate its original
E1--E6 public-release rules. Do not interpret its failure as a numerical
failure of an E1--E7 experiment or its earlier E1--E6 PASS record as a clean
audit of this enlarged directory. To inspect the current findings, run:

```bash
python validate_release.py
```

The top-level test wraps the same older check and currently has the same
scope limitation:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests
```

The included per-experiment tests remain useful for their documented scopes.
Full E3 GPU production and the expensive E5 numerical run are manual jobs; a
smoke result alone does not establish the values reported in the paper.

## Directory map

```text
simulation_experiments/
├── E1/ ... E7/              experiment releases
├── E3_Erev_gpu_pilot/       1024-query repeat-0 diagnostic
├── E3_Erev_high_density/    higher-density repeat-0 diagnostic
├── E7_query_density_extension/  E7 repeat-0 query-density diagnostic
├── .github/workflows/       earlier E1--E6 CI checks
├── tests/                   top-level repository contract
├── validate_release.py      standard-library public-release audit
├── requirements.txt         top-level test-only lock
├── LICENSE                  collection licensing notice
└── LICENSES.md              per-experiment license summary
```

## Reproducibility boundary

Per-experiment checks cover only their stated numerical and plotting contracts.
E1, E2, and E7 primarily regenerate figures from bundled summaries. E3 and E5
require substantial computation for a complete rerun. The auxiliary reverse-
coverage outputs are repeat-0 diagnostics. Numerical gates, data hashes,
formal-versus-smoke distinctions, and tolerances are specified inside each
experiment; cross-platform raster/PDF byte identity is not a scientific
requirement.

