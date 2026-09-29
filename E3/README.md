# E3: bandwidth robustness at fixed sample size

This directory is the self-contained release for simulation experiment E3. It
reproduces the bandwidth/radius sweeps for three estimators:

- smooth-kernel STSP, plotted against bandwidth `h`;
- directional regression (DR), plotted against bandwidth `h`;
- Yao--Yau (YY), plotted against transverse radius `r1`.

The release has exactly one sample-size design: `n=12000`. The shared noise
level is `sigma=0.015`; all methods use the same five manifolds, 20 repeats,
256 evaluation points, seed lineage, and 28-point plotting range from `0.06`
through `5.0`. STSP and DR use `h1=h/2`. YY fixes `c0=24`, `C2=12`, and varies
`r1=kappa*sigma`. Failed query-scale combinations are retained; the plotting
code displays points failing the preregistered reliability rule as open markers.

No external data is needed. Samples, Gaussian noise, and query grids are all
generated deterministically by the checked-in code. See
`data/GENERATED_DATA.md`.

## Layout

```text
config/e3_n12000.json       human-readable common contract
data/GENERATED_DATA.md      generated-data statement
methods/stsp/               STSP implementation and tests
methods/dr_yy/              DR/YY CPU code and CuPy-equivalent GPU backend
scripts/plot_results.py     matched five-manifold figures
reproduce.py                single public entry point
```

No raw formal packets are committed because they are generated artifacts. They
are written below `results/`, which is ignored by Git.

## Environment

Python 3.12 was used for the audited run. Create an isolated environment and
install the pinned CPU dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

For the formal DR/YY run on a CUDA 13 host, install the GPU lock file instead:

```bash
python -m pip install -r requirements-gpu-cu13.txt
```

## Reproduce the reported results

Run a fast CPU smoke test for all three methods:

```bash
python reproduce.py --mode smoke
```

To reproduce the reported E3 results, run the complete experiment (STSP on
CPU; DR and YY on the first visible CUDA device), audit every parent packet,
and render all six publication figures with this exact command:

```bash
python reproduce.py --mode formal --workers 4 --gpu-workers 4
```

The formal run is resumable: completed immutable parent packets are skipped.
To rebuild figures after a completed run:

```bash
python reproduce.py --mode plot
```

The output figures are separated by method and metric:
`e3_{stsp,dr,yy}_{average_error,max_error}.{png,pdf}`. No runtime panel is
created.

## Reproducibility contract

- Randomness uses the checked-in PCG64DXSM seed lineage.
- The same noisy sample is reused across all scales within a parent repeat.
- YY uses a common first-stage start across its entire radius grid.
- CPU/GPU numerical and success/failure equivalence is checked before formal
  DR/YY production.
- Reliability requires query failure rate at most `0.01` and at least 18
  complete repeats; failed values are never removed from the declared grid.
- Method implementations under `methods/` are preserved from the audited
  sources; release work changes packaging and the explicit E3 grid only.

## License

MIT; see `LICENSE`.
