# Figure 2: controlled STSP extrapolation weights

The current figure is `outputs/e1_stsp_population_infinity_with_ms_stsp.png/.pdf`. It uses six geometries, seven original noise levels and four curves, with the noise axis descending. All controlled estimators use exactly the same archived Z_h1 and ENF-matched Z_h2. Naive uses (4/3,-1/3); Proposed uses the noise-dependent coefficients. Original MS-STSP is a separate hard-ball algorithm, not the Naive control.

From the repository root: `python E1/weight_control/render_and_report.py --output NEW --figure-only`. This reads only sealed local analysis tables and does not fit or integrate. Output must be fresh. `analysis/VALUES.csv`, `SLOPES.csv`, `WEIGHTS.csv` and `CIRCLE_CONSTANT.csv` contain the actual values. `SOURCE_BINDING.json` binds all 114 frozen source/input members by byte checksum; historical external paths are provenance labels, while execution reads local `copy` paths.

Postprocessing reuses all 84 production/refined STSP packets; no Monte Carlo is introduced. Naive full-grid slopes are about 2 and the proposed slopes are about 4 on five examples (3.81725 on the corrugated torus). The circle constant is Gamma_1,2=0.476376834958...; Naive/sigma2 at .008 is 0.476369352883. These are finite-grid measurements, not asymptotic proofs.

To perform fresh postprocessing and the stricter independent circle calculation, copy the root Python files, PLAN.json, SOURCE_BINDING.json and frozen/ into a fresh study directory, then run analyze.py --circle-only, strict_circle.py, analyze.py, validate_precision.py and render_and_report.py --output outputs. Do not remove the selected results to force a rerun. Current release checks are in the repository root PUBLICATION_QA.json.
