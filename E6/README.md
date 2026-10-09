# E6 — noise misspecification and weights

```bash
python E6/reproduce.py --mode figures
python E6/smoke.py
```

Three original-template figures use source-bound fixed-fit results and
complete failure/audit evidence; the saved contract contains historical
`formal_publication_permitted=False` preparation metadata. It is not changed
to manufacture a new scientific/publication approval. Root release validation
and rendering QA are separate evidence.

Optional raw audit (not automatically repeated during figure replay) and a
fresh full scientific CPU run:

```bash
python E6/reproduce.py --mode audit
python E6/reproduce.py --mode recompute --run-name independent_run
```

Full fitting requires Linux Python3.12.7/NumPy2.1.3/SciPy1.18.1/
Matplotlib3.11.2, one CPU thread, separate `requirements-fitting.txt`.
No RNG is used: quadrature,256queries and nine deltas are deterministic.
Correct-sigma fitted points remain fixed while only weights vary. The metric
is **displacement from the correct-sigma output**, not manifold distance.
STSP/DR h1=.15,h2=.30,sigma=.05; matching is compact centered h1-PCA.
YY geometric inputs remain their unchanged controlled population source.
Residual magnitudes from different coefficient systems are not interchangeable.
Smoke verifies actual reduced STSP/DR fits remain identical across all weights
and zero deviation at delta0; it does not replace full resolution checks.
