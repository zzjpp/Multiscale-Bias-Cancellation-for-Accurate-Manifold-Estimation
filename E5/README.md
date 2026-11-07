# E5 — directional YY ablation

```bash
python E5/reproduce.py --mode paper
python E5/reproduce.py --mode smoke
```

`paper` validates and copies the four **selected, existing** exports into
`E5/results/figures`, without refitting. They correspond to paper Figure7
(finite PNG/SVG) and Figure20 (population PNG/PDF). Original configuration
labels 6/22 are historical metadata, not the current paper's figure numbers.

**Data limitation:** the seven original population packets are in
`data/population`; all Table2 slopes can be checked with `python verify_paper.py`
from the repository root. The 60 historical finite held-out packets are
missing locally. Selected Figure7 exports are preserved, but raw-data
verification of its medians is incomplete; see `docs/DATA_AVAILABILITY.md`.

Full expensive new CPU computations and raw-result plotting:

```bash
python E5/reproduce.py --mode raw --workers 4
python E5/reproduce.py --mode raw-plot
```

The original source/configuration, per-cell seed rule, gates, common three
fitted points and failure retention are unchanged. No rejected query is silently
replaced. Population seven-noise design uses 64 queries and radii(1.25,5,8)sigma;
finite sigma=.05 uses radii(5,6,12)sigma with n240000/480000/960000,20repeats,
256queries. Scalar/C/directional ablations change weights on the same fits.
The recorded pilot ill-conditioning/rejected setup and subsequent selected
configuration are preserved in provenance, not represented as predeclared.
Known curve initialization/evaluation is controlled. Smoke uses two queries
and reduced population quadrature, plus the original smallest finite parent;
it is not a full result reproduction.
