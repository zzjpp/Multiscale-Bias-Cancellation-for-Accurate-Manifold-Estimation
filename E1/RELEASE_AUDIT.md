# Release audit

Audit date: 2026-09-26

## Applied safe changes

- Added a standard `src/`, `tests/`, `data/`, `figures/`, and GitHub Actions
  layout.
- Added one documented entrypoint and exact dependency pins.
- Removed runtime caches and excluded them in `.gitignore`.
- Removed machine-specific paths, credentials, server references, and identity
  information from the release.
- Added machine-readable frozen configuration, data provenance, checksums, and
  a restrictive placeholder license.

## Numerical-behavior decisions

- The 115 audited error rows and all prespecified scale grids are unchanged.
- No failed cell was removed and no adaptive slope window was introduced.
- Seed `20260926` is recorded defensively, although figure reproduction uses no
  random sampling.
- The release audit flags two occurrences of `alpha=0.78` as possible
  hardcoded hyperparameters. They are the deliberately frozen opacity of the
  STSP/DR bars, not scientific parameters. Moving them would not improve
  scientific reproducibility and could change the intended visual match, so
  they are consciously retained.

## Final automated checks

- Identity findings: 0.
- Safe cleanup findings: 0.
- Reproducibility essentials: README, environment, seed, entrypoint, data, and
  results command all `OK`.
- Tests: 3 passed.
- End-to-end output: six PDFs, six PNGs, and `figures/FIGURE_MANIFEST.json`.

