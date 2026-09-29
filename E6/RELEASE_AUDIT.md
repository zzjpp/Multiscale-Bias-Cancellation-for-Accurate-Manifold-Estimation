# Release audit

Date: 2026-09-27

## Scope

This directory is an identity-neutral GitHub release containing only the code,
frozen configuration, audited numerical results, tests, and outputs required
for the three E6 figures.

## Deterministic checks

- Static reproducibility checker: README, environment, seed/hash stability,
  entrypoint, data instructions, and result commands all OK.
- Tests in the locked formal environment: 5 passed.
- The fast entrypoint generated exactly three PNG/PDF pairs.
- All three regenerated PNG files matched the formal E6 v3 figures byte for
  byte in the locked environment.
- No external data or random samples are used. PYTHONHASHSEED is fixed to zero;
  BLAS thread counts are fixed to one. Eight threads are used only across
  independent YY query integrations in the full computation.

## Static-audit dispositions

SAFE findings: none remain.

The seven reported identity findings are verified false positives: every
flagged character is Python's matrix-multiplication operator (@), not an email
address. No author name, email address, home directory, server address, or
credential is present.

The six reported behavior-risk findings are deliberate assignments/reads of
OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, and MKL_NUM_THREADS. Their values are
frozen by both one-click wrappers and match the formal run. They are retained
because removing them would weaken numerical reproducibility.

## Numerical evidence

The authoritative formal gate record is results/AUDIT.json. It reports PASS,
256/256 resolved queries, invariant frozen-fit digests, numerical zero at the
correct noise level, maximum weight residual 5.77e-15, and maximum fine/coarse
absolute difference 1.80e-10.
