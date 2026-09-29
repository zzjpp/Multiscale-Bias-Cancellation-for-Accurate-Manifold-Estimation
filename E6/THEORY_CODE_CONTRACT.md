# E6 theory-code contract

The theoretical source is Proposition 2.22 and Sections 3.1--3.3 of
`manifold_fitting (37).pdf`.  This experiment measures coefficient-error
propagation, not monotonicity of total finite-sample fitting error.

- The true noise level is fixed at `sigma=0.05`.
- The population data distribution, scale-specific fitting operators,
  fitted branches, reference normal spaces, and cross-scale correspondences
  are evaluated once at the true sigma and frozen.
- Only the cancellation weights receive
  `sigma_hat=(1+delta)*sigma`.
- DR uses `h1=0.15`, `h2=0.30` without tuning and
  `c_DR(h,sigma)=d(h^2+sigma^2)/2`.
- STSP uses the exact smooth-kernel coefficient from Theorem 3.4, not generic
  Richardson weights.
- YY uses physical radii held fixed.  Only the ratios
  `kappa_k=r_k/sigma_hat` used to form the axial and complementary weights
  change.  Residuals are evaluated against the true-sigma directional
  coefficients.
- The query grids contain 256 fixed, deterministic, approximately arc-length
  points for both geometries.
- `Delta_infinity` compares an incorrectly weighted estimator with the
  correctly weighted estimator formed from the exact same frozen fits.
- Every prespecified delta is retained.  There is no post-hoc trimming.
- A run is formal only if all constraints, the exact zero at delta=0, all
  256-query fits, frozen-fit digest invariance, and the coarse/fine numerical
  resolution gate pass.
