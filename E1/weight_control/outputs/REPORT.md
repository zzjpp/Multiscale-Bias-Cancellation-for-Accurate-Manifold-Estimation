# Shared-fit STSP weight experiment

All 42 geometry/noise settings are complete. Main evaluation reuses 84 existing production/refined STSP packets. No main fits or MS-STSP integration were rerun. One extra unit-circle CPU expectation solve at sigma=0.008 used 32768 latent nodes, radial order320 and stricter solver tolerances.

## Controlled quantities

Single, Naive and Proposed use the exact same Z_h1 and ENF-matched Z_h2 at each query. Naive uses (4/3,-1/3); Proposed uses the original archived weights after an independent one-dimensional coefficient check. Affine combinations are evaluated in the stable form Z_h1 + w2 B t2. The previous independently fitted Single baseline is retained in VALUES.csv as legacy_single; it is not silently used as the controlled Single.

## Actual full-grid slopes

| Geometry | Single | Naive h2 | Proposed | Original MS-STSP |
|---|---:|---:|---:|---:|
| Circle | 2.001415 | 1.999849 | 4.004123 | 1.941152 |
| S2 in R3 | 2.001826 | 1.999887 | 4.005450 | 1.978797 |
| Clifford torus | 2.002643 | 1.999890 | 4.007815 | 1.964466 |
| S3 in R4 | 2.002205 | 1.999957 | 4.006686 | 1.986863 |
| S2 in R10 | 2.001666 | 1.999955 | 4.004999 | 2.077790 |
| Corrugated torus | 2.006754 | 1.999590 | 3.817250 | 2.403150 |

The proposed full-grid corrugated slope is retained as 3.81725, not forced to4. The four-smallest-level diagnostic is3.91030. The other five proposed slopes are4.0041--4.0078. All six Naive slopes remain approximately2. These are finite-range data, not a proof of an asymptotic bound.

## Circle leading constant

Independent70-digit integral gives Gamma_1,2 = 0.4763768349582569652885717393231773337279215452490163825087128014734349. At sigma=.008 the observed Naive/sigma2 is0.4763693528826024 (relative difference0.00157062%). At sigma=.03 it is0.47626913326468157; the difference decreases as sigma decreases. Proposed/sigma4 stays between1.7470724303 and1.7571266003 on this grid.

## Numerical checks

The actual production/refined maximum point difference is 4.88222089373e-12; the largest metric difference is 1.01937958807e-12. Both are below the frozen budget 7.15600867451e-12 (0.001 times the smallest Proposed error, including average). The largest saved final root-correction diagnostic is 9.889539324e-12, below 7.15600867451e-10. These are diagnostics, not certified error bounds.

Independent60-digit geometric distance checks: 252 selected cases, maximum difference 4.14219439998e-16. For the five homogeneous examples this checks all original queries; for the corrugated surface it checks the maximum-error query of every controlled method/noise/resolution, not every query. The extra strict-circle solve matches the archived shared output points to the recorded double precision.

## Interpretation and limits

Along h=a sigma, the exact STSP bias is beta(a,d,D) sigma2. Naive h2 weights do not cancel this coefficient; the independent unit-circle coefficient is nonzero. Proposed cancels it. Original MS-STSP uses a different hard-ball kernel and fixed-reference iteration, so its comparison also involves operator and correspondence differences. It is not the Naive control and is reused as a separate population-adapted algorithm.

The first five homogeneous geometries each use one original canonical query; corrugated maximum uses128 fixed equal-area meridional queries. Thus E_infinity is a maximum over preserved queries, not a certified continuous supremum. All seven sigma levels and all saved successes are retained. No density, radius, solver or failure-rule tuning was used to force the predicted slopes. Original publication directories and organized code are not overwritten.

## Files and replay

- Figure PNG/PDF and figure_caption.tex in this directory.
- supplementary_table.tex/.pdf: one table with slope and constant panels.
- section_5_1_1_discussion.tex: manuscript-ready discussion.
- ../analysis/VALUES.csv, SLOPES.csv, WEIGHTS.csv, CIRCLE_CONSTANT.csv and PRECISION.csv.
- ../analysis/SHARED_POINTS.json.gz: all shared coordinates and outputs, both resolutions.
- ../SOURCE_BINDING.json and ../PLAN.json: source hashes and frozen design.

From the study directory, run `python -X utf8 -B render_and_report.py --output fresh_plot_directory --figure-only` with a nonexistent output directory to replay the figure. No fits or integrations occur.
