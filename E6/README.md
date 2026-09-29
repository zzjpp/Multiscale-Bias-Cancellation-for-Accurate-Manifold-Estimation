# E6: Sensitivity to Noise-Level Misspecification

This folder is the minimal public release for the three E6 figures. It contains
no earlier E6 variants, finite-sample boxplots, exploratory parameter searches,
runtime plots, or unrelated experiment code.

## Scientific question

All scale-specific population fits are computed once using the true noise level
sigma = 0.05 and then frozen. Only the noise level inserted into the
multiscale weights is replaced by sigma_hat = (1 + delta) sigma, with

delta in {-0.30, -0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20, 0.30}.

The left panel reports the residual cancellation coefficient. The right panel
reports the population maximum displacement from the correctly weighted
estimator. There are 256 deterministic, approximately arc-length query points
on each geometry. No external data and no random sampling are used.

## One-click figure reproduction

Windows:

    reproduce.bat

Linux/macOS:

    sh reproduce.sh

These commands install the exact formal-run package versions, run the tests,
and regenerate exactly these three figures from the included formally audited
population results:

- figures/e6_dr_noise_misspecification_main.pdf
- figures/e6_stsp_noise_misspecification_supplement.pdf
- figures/e6_yy_noise_misspecification_supplement.pdf

PNG copies are generated alongside the PDFs. After the environment is
installed, this path normally takes less than one minute.

## Full numerical recomputation

To recompute all population fits, audit them, and regenerate the same figures:

Windows:

    recompute_all.bat

Linux/macOS:

    sh recompute_all.sh

The full run is deterministic and uses eight CPU threads only for independent
YY query integrations. BLAS threads are fixed to one. It does not use a GPU.
Runtime depends on the CPU and is typically tens of minutes. Peak memory and
runtime can vary across SciPy builds.

## Frozen setup

- True noise: sigma = 0.05.
- DR and STSP: h1 = 0.15, h2 = 0.30.
- YY circle radii: 0.20, 0.50.
- YY variable-curve radii: 0.20, 0.50, 0.80.
- Geometries: S^1 in R^2 and
  gamma(t) = (1.5 cos t, sin t, 0.3 sin 2t) in R^3.
- Relative noise-error grid: nine prespecified values from -30% to +30%.
- Query count: 256 per geometry.
- Every grid point is retained; there is no post-hoc trimming.

For DR, the left-panel residual depends only on d, h, sigma, and sigma_hat.
Both geometries have d = 1, so their residual curves coincide exactly. A single
blue curve is drawn intentionally; the geometry-dependent right panel retains
both curves.

## Audit status

The included formal result passed all registered gates:

- 256/256 queries resolved for every method and geometry;
- cancellation and geometric displacement are numerically zero at delta = 0;
- frozen fit digests are invariant across delta;
- maximum weight-constraint residual: 5.77e-15;
- maximum fine/coarse absolute difference: 1.80e-10;
- maximum fine/coarse relative difference: 2.01e-7.

The machine-readable record is results/AUDIT.json. The theoretical mapping is
documented in THEORY_CODE_CONTRACT.md.

## Repository layout

- config.json: frozen scientific and numerical parameters.
- plot_figures.py: fast entrypoint for exactly the three figures.
- run_experiment.py: complete recomputation, audit, and plotting entrypoint.
- src/e6_misspec/: estimator coefficients, geometry, population integration,
  auditing, and plotting.
- results/: formally audited numerical values bundled for direct plotting.
- tests/: coefficient, protocol, import, and end-to-end plotting checks.
- SHA256SUMS.txt: checksums for the release files.

The source uses only synthetic analytic manifolds and requires no download.
