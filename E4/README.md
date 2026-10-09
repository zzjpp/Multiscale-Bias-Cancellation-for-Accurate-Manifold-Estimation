# E4 — fixed-query correspondence vs candidate density

```bash
python E4/reproduce.py --output ./new_figures/E4
python E4/reproduce.py --mode verify
python E4/smoke.py
```

The figure uses the actual rerun with the exact archived ENF h1 fitted
coordinates shared by Oracle, ENF and every NN density. Ten original n200000
parents (repeats10–19), 32 query arrays and source-bound frozen inputs are
included. sigma=.06,h1=.12,h2=.24 on
gamma(theta)=(1.5cos(theta),sin(theta),.3sin(2theta)).

M_match=32/36/40/44/48/52/56/60/64/80/96/112/128/256/512. NN uses the
original nested balanced arclength candidate refinement and Euclidean matching.
ENF uses PCA at the fixed h1 coordinate; Oracle uses the true geometric chart.
The h1 reference construction/branch selection also uses the controlled true
chart. These controlled geometric choices are explicitly retained.

An explicit fresh GPU rerun is available:

```bash
python E4/reproduce.py --mode recompute
```

It verifies original source/input hashes and the archived actual fixed-query
CPU/GPU gate, refuses an existing `fitting/outputs/packets`, and writes new
full-density packets/tables/figures under `fitting/outputs`. To independently
run a new gate, use a fresh copy of `fitting` without the old gate file and
run `fixed_queries.py gate` before `fixed_queries.py run`; never relabel the
bundled gate as newly tested. Use the original Linux GPU environment, not
figure-only requirements. Actual four-query CPU smoke is not a new GPU gate.

Reported complete matching cost includes original fine reference construction
and matching/coarse candidate fitting. The additional frozen-coordinate
residual consistency check was added **after** the original reference
stopwatch and is excluded from recorded times. Timing values remain unchanged.
Discrete NN approaching ENF/Oracle as candidates become dense is not a
claim that continuous nearest projection is inherently inaccurate.
