# Changelog

## 2026-11-19 — CI and fresh standalone entrypoints

- Fixed the confirmed Windows pytest temporary-directory parent error.
- Connected all seven public `reproduce.py` scripts to the fresh source-bound pipeline; no-argument invocation displays safe help.
- Renamed plotting-only interfaces to `render_figures.py` with explicit tables, and historical wrappers to `legacy_replay.py`.
- Added Windows/Ubuntu CI and targeted entrypoint regressions; retained estimators, seeds, numerical gates, task plans and figure filenames.
- Documented AoS's official code/supplement policy and the distinction between engineering checks and full numerical replication.

## 2026-11-13 — Paper-aligned public interface

- Bound the repository to the supplied 2026-11-12 manuscript.
- Organized experiment directories by paper topic and added a section/figure catalog.
- Added `python -m manifold_experiments`, readable selections, and `--list`.
- Retained `run_paper.py`, E1–E5 task codes, native metric fields, and figure filenames.
- Formatted public orchestration/validation and documented contribution checks.
- Preserved numerical protocols, job order, seeds, solvers, and failure handling. Actual before/after validation is recorded in `docs/REFACTOR_VALIDATION.json`.

Earlier revisions remain in Git history. Large historical outputs are not required by the fresh-run pipeline.
