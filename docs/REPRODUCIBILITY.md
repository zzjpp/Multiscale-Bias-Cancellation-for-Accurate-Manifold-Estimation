# Reproduction scope

The supported command is `python -m manifold_experiments --experiment all --output NEW`. It computes new results. See the root README for environments, smoke versus paper profiles, selected experiments, outputs and cost cautions.

The source-only runtime is exportable with `python tools/export_runtime.py --output NEW_SOURCE_DIRECTORY`. It includes the exact runtime manifest and all required source/configuration/E3 design coordinates, and excludes historical fitted outputs. Testing this export detects accidental dependence on archives in a parent workspace. The export must be placed in a fresh directory outside this repository.

The runtime and distribution inventories have distinct purposes. `RUN_SOURCE_MANIFEST.json` binds only required source/design files and supports source-bound new tasks. `SOURCE_MANIFEST.json` inventories the entire distribution, including optional small references and documentation. `validate_release.py` checks bytes, while `verify_paper.py` checks paper protocol definitions. Neither is numerical reproduction or independent validation.

Release QA includes a clean Python 3.12.9 virtual environment, actual 24-task reduced runs and source-only runs, plus protocol/failure/custody regressions. Full production computations were not rerun during release preparation; the full plan is expensive. Timing is machine-dependent. CPU mode is explicitly the original reference implementation, while GPU mode retains its original environment and same-input equivalence requirements.

The current repository is tied to the supplied 2026-11-12 manuscript recorded in `PAPER_REFERENCE.json`. AoS code/supplement requirements are documented in `docs/AOS_REPRODUCIBILITY.md`; no formal artifact-evaluation badge or journal certification is claimed. The repository uses the MIT License; existing contributor notices are retained. E1 was relicensed to MIT with the repository owner's explicit authorization on 2026-11-12.

Standalone `reproduce.py` entrypoints now share the fresh orchestrator. Plot-only programs require explicit input tables; legacy archive replay is not a fresh reproduction path. See `docs/CI_AND_ENTRYPOINT_AUDIT.md` for current checks and limitations.
