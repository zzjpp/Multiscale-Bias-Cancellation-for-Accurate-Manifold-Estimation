# E3 — bandwidth and reverse coverage

```bash
python E3/reproduce.py --output ./new_figures/E3
python E3/reproduce.py --verify-only
python E3/run_fits.py --method YY --smoke
```

Default is the author-selected **0.4/1/1.6 YY radius** version, not the 0.8
diagnostic. It draws all 12 original-format 3×2 PNG/PDF pairs from the selected
6720-row table: three methods × forward/reverse × average/maximum. No green
background or small footer. Green YY1.6 curve, all 28 scales, NaN gaps,
independent baselines, failures and hollow unreliable points are retained.

Panel keys: circle_r2/sphere2_r3/clifford_torus_r4/sphere3_r4/sphere2_r10/
corrugated_torus_r3. n=12000/24000/24000/60000/24000/36000; sigma=.015.
Main20 repeats, 256 queries (corrugated512). Reverse repetition0 with
1024/8192/8192/16384/8192/1024 queries; **all262144 reference points**.
The complete fitted cloud is evaluated against all references; no shard-error
pooling. Original query and reference NPY arrays are bundled and checksummed.

Full new-run commands in the strict original Linux GPU environment:

```bash
python E3/run_fits.py --method DR --phase main --geometry sphere3_r4 --repeat 0 --output ./new_raw/DR_main_sphere3_r4_00.json.gz
python E3/run_fits.py --method DR --phase rev --geometry sphere3_r4 --output ./new_raw/DR_rev_sphere3_r4_00.json.gz
```

Repeat all methods/geometries and main repeats0–19 (360 main +18 full-cloud
reverse tasks). A same-parent CPU/GPU gate uses original scales .06/.82/5
and selected queries, original status/reason/point tolerances, plus YY
common-start center/axis checks. There is no CPU fallback. Optional
`--data-file` accepts explicitly supplied observations and records their SHA;
it does **not** pretend those are the historical platform-specific parent.

```bash
python E3/evaluate_fits.py --input ./new_raw --output ./new_summary.csv
python E3/reproduce.py --table ./new_summary.csv --output ./new_figures/E3_rerun
```

STSP/DR inter-manifold matching uses original compact centered h1-PCA; DR's
internal Gaussian fit and exact nonorthogonal-basis failure are unchanged.
YY original data axis/known-geometry controlled initialization remains, with
distinct main-V2/reverse-V1/corrugated libraries. Sampling streams/platform
differences and adaptive n selection are disclosed rather than made artificially
paired. Small-n support, spectral, geometric and numerical failures can remain;
the selected DR S3 results at n60000 are not repaired or cropped.

Main reliability: >=18/20 complete AND total failure<=1%; reverse<=1%.
Partial finite repeat quantiles remain displayed even if unreliable. Private
historical raw/queue histories are not shipped; new-run files never overwrite
the selected paper table. Legacy `e7_config.json` inside the geometry-only
package is retained byte-identically as a historical seed/config source;
there is **no independent E7 experiment or launcher**.
