# E3 reverse-coverage GPU pilot

This isolated pilot evaluates reverse coverage for the released E3 DR and YY
implementations with 1024 deterministic query points and a 65536-point
reference grid. It **does not rerun the original 256 queries**: the GPU only
evaluates 768 newly added points, and the summarizer merges those outputs with
the immutable formal 256-query packets. It does not modify or replace the
audited E3 release.

The scientific sample size, noise, five geometries, scale grids, kernels,
weights, solvers, failure taxonomy, and seed lineage are inherited unchanged
from `../E3`.  Only the query count is overlaid from 256 to 1024.  Repeat 0 is
predeclared as a diagnostic pilot; it is not formal 20-repeat evidence.

DR and YY use the existing CuPy backend.  STSP is deliberately excluded from
this GPU pilot because the released STSP implementation is CPU-only.

Before production, run the original E3 CPU/GPU equivalence gate.  Then launch
one parent per method and geometry with `run_gpu_pilot.py`.  After all ten
packets finish, run `python summarize_erev.py` to produce the CSV.

## Completed diagnostic

The 1024-query diagnostic is complete for DR and YY on all five geometries.
All 10 incremental packets completed, the inherited CPU/GPU equivalence gate
passed 902/902 comparisons, and the union of the immutable 256-point grid and
the new 768-point complement agrees with the frozen 1024-point grid to
`1.57e-16`.  The final audit status is `PASS`.

Release-ready outputs are under `formal_results_1024/`:

- `e3_erev_1024_complete.csv`: all 560 method/geometry/estimator/scale rows;
- `e3_erev_1024_dr.{png,pdf}` and `e3_erev_1024_yy.{png,pdf}`;
- `E3_EREV_1024_AUDIT.json`: completeness and scientific-provenance audit;
- `OUTPUT_SHA256.txt`: hashes of every released output.

The figures and audit can be rebuilt locally from the merged CSV with:

```powershell
python finalize_erev.py `
  --input e3_erev_1024_complete.csv `
  --output-dir formal_results_1024
```

The evidence class remains a predeclared repeat-0 query-density diagnostic.
It complements, but does not replace, the previously audited 20-repeat E3
DR/YY experiment.
