# E7 release refactor

## Applied safe changes

- Built a fresh minimal release instead of deleting from the audited working
  directory.
- Retained only the three audited aggregate tables and their audit records.
- Added a single plotting entrypoint, pinned dependencies, README, license,
  data provenance, tests, and ignore rules.
- Excluded caches, logs, packets, diagnostics, exploratory runs, runtime plots,
  and reverse-error figures.

## Behavior-risk decisions

- Scientific configurations, seeds, grids, reliability classifications, and
  aggregate values were not changed.
- The new code validates the complete frozen scale grid before plotting.
- Plot constants were copied unchanged into `config.json`; no estimator is
  recomputed by this release.

## Identity

This release is prepared as a public, non-blind GitHub artifact. It contains no
personal paths, server addresses, usernames, or credentials.
