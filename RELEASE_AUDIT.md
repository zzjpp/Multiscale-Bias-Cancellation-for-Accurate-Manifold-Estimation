# Simulation release audit

Date: 2026-09-27

## Scope

`simulation_experiments/` contains only the six isolated simulation releases
E1--E6 plus repository-level documentation, CI, and a structural validation
test. It does not contain the real-data experiment, server credentials,
historical working directories, quarantined results, or generated E2/E3/E4/E5
run directories.

The experiment sources were copied from the already reviewed local releases:

- E1 from `E1_simulation/`;
- E2--E6 from their same-named release directories.

A per-file SHA-256 comparison passed for every copied code and data file. The
only deliberate differences are the E1 directory-name documentation, the E2
MIT copyright-holder label correction, removal of the nested E1 workflow, and
the new repository-level files.

## Executed checks

| Scope | Check | Result |
|---|---|---|
| Repository | `python validate_release.py` | PASS; E1--E6 present, no forbidden generated directories or sensitive-path/credential matches |
| Repository | `python -m pytest -q tests` | 1 passed |
| E1 | `python -m pytest -q` | 3 passed |
| E2 | `python -m pytest -q` | 2 passed |
| E3 | `python reproduce.py --mode smoke` | 4 + 5 + 5 tests passed; STSP/DR/YY smoke cells completed |
| E4 | `python -m pytest -q` | 1 passed |
| E5 | `python reproduce.py --mode smoke` | 5 tests passed; population and finite-sample smoke cells passed |
| E6 | `python -m pytest -q` | 5 passed |

Smoke outputs and Python/pytest caches were removed after testing.

## Static audits

The reproducibility audit reports all six essentials as OK: README, pinned
environment, seeds/determinism declaration, entrypoints, data instructions, and
exact result commands. It detects eleven test files across the collection.

The release-refactor audit reports:

- identity findings: 0;
- SAFE cleanup findings: 0;
- behavior-sensitive findings: 44.

The remaining behavior-sensitive findings are intentionally not auto-fixed:
seven are frozen numerical or plot constants, and the remainder concern
environment reads or `set(...)` uses in the already audited E3--E6 numerical,
test, and orchestration code. Changing any of these solely to silence a static
heuristic could change evaluation order, thread behavior, solver behavior, or
reported graphics. They therefore remain exactly as in the source releases;
the experiment-level tests and formal audit contracts are the controlling
evidence.

## Size and hosting

The collection has 170 files totaling approximately 2.36 MB. The largest file
is below 0.25 MB, so Git LFS is not required. Generated run directories are
ignored. Each experiment has its own authoritative license; see `LICENSES.md`,
especially the all-rights-reserved E1 status.

This audit establishes release cleanliness and functional execution on the
local checked environment. It is necessary but not sufficient for an external
artifact badge or an independent numerical reproduction claim.

