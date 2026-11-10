# Reproduction scope

The supported command is `python run_paper.py --experiment all --output NEW`. It computes new results. See the root README for environments, smoke versus paper profiles, selected experiments, outputs and cost cautions.

The source-only runtime is exportable with `python tools/export_runtime.py --output NEW_SOURCE_DIRECTORY`. It includes the exact runtime manifest and all required source/configuration/E3 design coordinates, and excludes historical fitted outputs. Testing this export detects accidental dependence on archives in a parent workspace. The export must be placed in a fresh directory outside this repository.

The runtime and distribution inventories have distinct purposes. `RUN_SOURCE_MANIFEST.json` binds only required source/design files and supports source-bound new tasks. `SOURCE_MANIFEST.json` inventories the entire distribution, including optional small references and documentation. `validate_release.py` checks bytes, while `verify_paper.py` checks paper protocol definitions. Neither is numerical reproduction or independent validation.

Release QA includes a clean Python 3.12.9 virtual environment, actual 24-task reduced runs and source-only runs, plus protocol/failure/custody regressions. Full production computations were not rerun during release preparation; the full plan is expensive. Timing is machine-dependent. CPU mode is explicitly the original reference implementation, while GPU mode retains its original environment and same-input equivalence requirements.

The current repository is tied to the final `arxiv_final.pdf` recorded in `PAPER_REFERENCE.json`. No specific journal's artifact track or double-blind policy was supplied, and no formal artifact-evaluation badge is claimed. Existing licensing boundaries remain unchanged.
