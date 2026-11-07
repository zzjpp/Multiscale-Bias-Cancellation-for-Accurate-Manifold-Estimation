# E1 — population order

From the repository root:

```bash
python E1/reproduce.py --output ./new_figures/E1
python E1/smoke.py --method STSP
python E1/smoke.py --method DR
python E1/smoke.py --method YY
```

Default replay validates the original saved tables/provenance and draws the five raw-error panels selected for S1-S3 plus the normalized circle triptych (Figure 4). Figure 2 is drawn separately by `weight_control/render_and_report.py --output NEW --figure-only`. `--all-diagnostics` additionally draws the unselected legacy two-curve STSP maximum figure. No fit occurs. Noise axes decrease from 0.03 to 0.008.

Original numerical packages remain separate to avoid mixing similarly named
modules. Full CPU population runs (expensive), writing their own fresh results:

```bash
python E1/theory_path_population_all_e1/run_experiment.py --workers 1
python E1/theory_path_population_six_manifolds/run_experiment.py --workers 1
python E1/theory_path_population_yy_all_e1/run_experiment.py --workers 1
python E1/theory_path_population_yy_six_manifolds/run_experiment.py --workers 1
```

Use fresh package result directories; inspect any previous failed/partial
results rather than deleting or overwriting them. GPU alternatives and their
original gates are in the same four packages (`--help` describes their native
arguments). Package outputs are **new calculations**, not replacements for the
provenance-bound selected tables. Reduced-quadrature smoke is not the full
population convergence/resolution gate.

Noise grid: .008/.01/.0125/.016/.02/.025/.03. STSP/DR h1=3sigma,
h2=6sigma; DR internal Gaussian fitting remains unchanged and correspondence
uses compact centered h1-PCA. YY transverse radii=(1.25,5,8)sigma,
first stage=3sigma and longitudinal scale=12sigma sqrt(log(1/sigma)).
Six manifolds are retained; known latent quadrature/controlled geometry and
original symmetry-reduced query designs are not blind-data estimation.
Original population GPU environment differs from figure replay; no GPU
equivalence is inferred from Windows CPU smoke.

E1's original all-rights-reserved license is retained.
