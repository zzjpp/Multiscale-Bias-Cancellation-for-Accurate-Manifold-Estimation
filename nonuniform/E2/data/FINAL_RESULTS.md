# Nonuniform-density E2: checked final results

All360 parents,360 exact-input CPU/GPU gates and the original once-only stored-output audit completed. The actual3352-member archive was downloaded and byte-bound on D. All6 PNGs and6 full PDF pages have now been visually inspected; all12 exports are identical to independent cold-process replay. Original89 organized E2 files remain byte-identical at this final check. No GPU rerun, numerical audit replay or figure redraw was performed for this report.

## Protocol

The six smooth positive surface-area densities are exactly those of the preceding nonuniform E1 study, with amplitude0.5 and max/min3. Original area-uniform clean proposals are rejected with probability complementary to (1+0.5phase)/1.5 before original sigma0.005 Gaussian noise. Independent acceptance stream2 is declared; original method-specific proposal/noise seed lineages remain. All eight nested n,20 repeats,144 queries, original kernels/weights/PCA/matching/solvers/tolerances and reliability rules are preserved. STSP/DR display the coarse Single(h2); their independent fine Single remains in the full table. YY retains original E2 radii(0.8,1,1.6), controlled known-geometry preliminary initialization and data axis. See ../../README.md and PLAN.json for exact definitions and seeds.

## Largest-n comparison

Values are medians over20 repeat-level errors at n=500000. Ratio=Multi/Single; below1 means lower observed error. All listed largest-n Single and Multi cells satisfy the original reliability rule.

| Method | Geometry | Average ratio | Maximum ratio |
|---|---|---:|---:|
| STSP | circle_r2 | 0.040337 | 0.152902 |
| STSP | sphere2_r3 | 0.110186 | 0.360870 |
| STSP | clifford_torus_r4 | 0.220544 | 0.533322 |
| STSP | sphere3_r4 | 0.357199 | 1.104959 |
| STSP | sphere2_r10 | 0.376980 | 0.619171 |
| STSP | corrugated_torus_r3 | 0.269617 | 0.527830 |
| DR | circle_r2 | 0.006536 | 0.020822 |
| DR | sphere2_r3 | 0.009435 | 0.024489 |
| DR | clifford_torus_r4 | 0.016726 | 0.037049 |
| DR | sphere3_r4 | 0.012784 | 0.032901 |
| DR | sphere2_r10 | 0.011309 | 0.025734 |
| DR | corrugated_torus_r3 | 0.037758 | 0.095785 |
| YY | circle_r2 | 0.033364 | 0.118835 |
| YY | sphere2_r3 | 0.069079 | 0.239450 |
| YY | clifford_torus_r4 | 0.345579 | 1.465032 |
| YY | sphere3_r4 | 0.183448 | 0.462636 |
| YY | sphere2_r10 | 0.275053 | 1.226935 |
| YY | corrugated_torus_r3 | 0.243982 | 0.833257 |

At n=500000 average error improves in all18 method/geometry combinations. Maximum error improves in15/18; exceptions are STSP S3(+10.50%), YY Clifford torus(+46.50%) and YY S2/R10(+22.69%). These are actual reliable-grid counterexamples to a universal maximum-error improvement claim, not omitted failures. No algorithm/radius/sample tuning was performed after observing them.

## Reliability and undefined points

Each figure has96 source cells(6geometries x8sample sizes x2curves).

| Method | Metric | Reliable | Finite unreliable/hollow | Undefined |
|---|---|---:|---:|---:|
| STSP | E_avg | 60/96 | 36 | 0 |
| STSP | E_infty | 60/96 | 36 | 0 |
| DR | E_avg | 70/96 | 26 | 0 |
| DR | E_infty | 70/96 | 26 | 0 |
| YY | E_avg | 54/96 | 34 | 8 |
| YY | E_infty | 54/96 | 34 | 8 |

Main reliability remains at least18/20 fully complete repeats and at most1% query failures. Partial finite-repeat medians/IQR are preserved and flagged. YY S3 has no defined plotted median at the first four n=2000/5000/10000/20000 for both displayed curves; no missing values were filled. STSP/DR Multi on S3 first becomes reliable at n=200000; the same is true for YY Single/Multi on S3.

## Full failure classification

Counts cover the complete independent-estimator/query/prefix grid. Average/maximum copies of the same failure counts are counted once, not twice.

| Native failure reason | Outcomes |
|---|---:|
| first_stage_mass_failure | 74564 |
| reference_branch_failure | 35808 |
| low_support | 32293 |
| eigengap_failure | 24288 |
| normal_transversality_failure | 2232 |
| first_stage_low_support | 1808 |
| target_branch_failure | 535 |
| newton_no_descent | 157 |
| second_stage_incomplete | 6 |

Total failed outcomes: 171691/1105920. Every failure is preserved in the imported raw/gate/claim/log evidence and full768-row table. Completeness/equivalence audit PASS does not mean zero scientific failures or all-grid reliability.

## Files and scope

The six PNG/PDF pairs are in this directory. NONUNIFORM_SUMMARY.csv is a byte-identical convenience copy of the sealed full768-row table. SINGLE_MULTI_COMPARISON.csv provides288 comparisons including all n and both metrics; RELIABILITY_AND_FAILURE_SUMMARY.json and VISUAL_REVIEW.json contain actual diagnostics and visual custody. Cold replay/source checks are in COLD_REPLAY_AND_SOURCE_CHECK.json.

Each maximum is over the original144 fixed queries, and the displayed statistic is the median of repeat maxima, not a certified continuous supremum. An average-error gain does not imply a maximum-error gain. These results alone do not isolate the causal effect of density from finite-sample stochastic differences; no paired uniform-versus-nonuniform causal claim is made. No original E1/E2/official figure or GitHub files were edited.
