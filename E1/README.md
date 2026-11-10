# E1 — population verification (Section 5.1 and Appendix E.1)

```bash
python E1/reproduce.py --output NEW
python E1/weight_control/render_and_report.py --output NEW_CONTROL --figure-only
python E1/smoke.py --method STSP
python E1/smoke.py --method DR
python E1/smoke.py --method YY
```

Default replay draws the five error plots in Figures E.1–E.3 and the circle normalized triptych in Figure 4. Figure 2 is the separate shared-fit weight control, including the original hard-ball MS-STSP. The numerical tables and their historical Single/Multi names are unchanged; presentation uses the baseline/GBC labels of the final paper. Noise decreases from 0.03 to 0.008.

Original numerical packages are kept separate to avoid same-name module collisions. Fresh full population calculations are expensive:

```bash
python E1/theory_path_population_all_e1/run_experiment.py --workers 1
python E1/theory_path_population_six_manifolds/run_experiment.py --workers 1
python E1/theory_path_population_yy_all_e1/run_experiment.py --workers 1
python E1/theory_path_population_yy_six_manifolds/run_experiment.py --workers 1
```

The first and third packages supply the original five geometries; the six-manifold extensions supply corrugated-torus calculations and their supporting audits. They are source dependencies, not separate paper experiments. Use fresh results directories and retain failed/partial calculations. Native `--help` documents CPU/GPU commands and original accuracy gates; reduced smoke is not a population convergence validation.

Seven noise levels: .008/.01/.0125/.016/.02/.025/.03. STSP/DR h1=3sigma,h2=6sigma. DR retains its Gaussian fitting operator; correspondence uses compact centered h1-PCA. YY uses transverse radii (1.25,5,8)sigma, first stage 3sigma and longitudinal scale 12sigma sqrt(log(1/sigma)). Controlled geometry initialization, original symmetry-reduced queries and all scientific settings remain unchanged. E1's original all-rights-reserved license remains in effect.
