# E4: estimated-normal-fiber versus nearest-neighbour correspondence

This self-contained folder reproduces the two-panel E4 figure for the
nonconstant-curvature closed space curve

```text
gamma(theta) = (1.5 cos(theta), sin(theta), 0.3 sin(2 theta)) in R^3.
```

It contains only the finite-sample `N_q = M_match = 32` experiment used to
produce the reference figure. It does not contain the historical E4 sweeps,
the later 256/2000-query diagnostics, server launchers, or unrelated results.

## Scientific protocol

- `h1 = 0.12`, `h2 = 0.24`, `sigma = 0.06`;
- `n = 200,000` synthetic observations;
- `N_q = M_match = 32`;
- ten held-out repeats, indexed 10--19;
- fixed `PCG64DXSM` seed lineage with master seed `20260911`;
- methods: true-normal Oracle, estimated-normal-fiber (ENF), and discrete
  nearest neighbour (NN);
- bars: median across repeats; error bars: interquartile range.

All data are generated synthetically. No external dataset or network access is
required.

## Environment

The reference output was generated with Python 3.12.9. Install the pinned
dependencies in a fresh environment:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

## One-command reproduction

From this directory, run:

```bash
python run.py
```

The command creates `outputs/` containing:

- `e4_enf_vs_nn_equal_density.png`;
- `e4_enf_vs_nn_equal_density.svg`;
- `summary.csv`;
- `AUDIT.json`;
- the ten compact repeat packets.

Expected runtime is about one minute on a modern desktop CPU. The program
forces no external data download and records the deterministic seed lineage in
every packet.

## Smoke test

```bash
python -m pytest -q
```

The smoke test runs the real estimator on a small synthetic instance and
checks that the figure, summary, and audit are produced. The smoke result is a
functionality check, not evidence for the paper.

## Reference result

The checked-in target is `reference/e4_enf_vs_nn_equal_density.png`. The
reference median errors are:

| Method | E_avg | E_infty |
|---|---:|---:|
| Oracle | 0.0100548 | 0.0227381 |
| ENF | 0.0100516 | 0.0227160 |
| NN | 0.0209652 | 0.0418692 |

Floating-point results should agree within numerical tolerance on the pinned
environment; image files are not expected to be byte-identical across
operating systems because font rendering can differ.

## Layout

```text
E4/
  config.json             frozen experiment and solver parameters
  run.py                  single end-to-end entrypoint
  requirements.txt        pinned Python dependencies
  src/e4/                 numerical estimator implementation
  tests/test_smoke.py     small real-pipeline smoke test
  reference/              target figure and reference summary
```

The 32-point experiment is a sparse equal-density correspondence example. It
must not be interpreted as uniform dominance over arbitrarily dense NN
candidate sets.
