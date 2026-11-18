# CI and reproduction-entrypoint audit — 2026-11-19

## Confirmed problem and repair

The [previous main-branch Actions run](https://github.com/zzjpp/Multiscale-Bias-Cancellation-for-Accurate-Manifold-Estimation/actions/runs/43384965165), at commit `3f1a28a2f5d4b6cf1c811236376ba12cf080709c`, completed with `24 passed, 3 errors`. All three errors were pytest fixture setup failures creating `./tmp/pytest` without its parent directory. Earlier format/manifest/protocol steps passed; the real smoke step was skipped. That run did not establish a numerical algorithm failure, nor did it validate the later smoke step.

CI now uses `--basetemp ./pytest_tmp` and checks Windows and Ubuntu. Test outputs are excluded from source/runtime manifests. The workflow retains test XML and smoke summary/log/error artifacts for inspection; it does not put fitted-output archives into the repository. Read the [actual Actions conclusions](https://github.com/zzjpp/Multiscale-Bias-Cancellation-for-Accurate-Manifold-Estimation/actions) for the commit being assessed. This document does not presume a not-yet-executed CI run has passed.

## Public interfaces

All seven experiment-level `reproduce.py` scripts now delegate to the same fresh, source-bound orchestrator. No arguments displays help; `--output NEW` performs the selected computation, evaluation and plotting. All original paper tasks, seeds, kernels, solvers, precision gates, failure handling, and figure filenames remain unchanged.

The fixed-scale, bandwidth and nonuniform fixed-scale plotting programs are explicitly named `render_figures.py`; `--table` and `--output` are required. The orchestrator supplies tables from the current run. Original archive-dependent wrappers are retained as `legacy_replay.py`. Version-specific frozen scientific modules retain their original names and bytes, and are not advertised as standalone public reproduction commands.

## Actual local checks

- **46 tests passed** using the corrected pytest command. These cover all seven local entrypoints from a different working directory, exact plans, safe no-argument help, explicit renderer inputs, missing-packet/source-mismatch refusals, scientific-source byte preservation, and exclusion of test outputs.
- A **24-task actual CPU fitting/evaluation/plotting run passed** in a separate source/design-only export with no historical fitted outputs. It used the pinned Python 3.12.9 environment.
- Against the previously checked reduced run, **15,199 scientific numeric values matched at zero tolerance**, including fitted points, diagnostics, inputs/hashes, statuses and failures. Only observed timings and random invocation IDs were excluded. The full 1,356-task design and job order remain identical; the protocol module's AST is unchanged.
- All three renamed full-format renderers consumed actual existing tables and produced **20 PNGs identical byte-for-byte** to the prior checked exports, plus 20 PDFs. This is interface regression, not new full-size scientific fitting.
- The six static reproducibility essentials are present. A generic ML artifact heuristic flags the TikZ READMEs because it does not recognize `pdflatex`, and flags absent training/checkpoints/archival badges. The schematics have actual LaTeX commands; this estimator study requires neither neural training nor pretrained checkpoints. Figure-to-command mappings are centralized in `docs/FIGURE_MAP.json`. These heuristic warnings are not relabeled as scientific validation or journal badge approval.

Private test logs, earlier erroneous export attempts, generated plots and original evidence were retained outside the Git distribution. No original results were replaced, and no scientific failure was suppressed.

## Unverified scope

The complete fresh **1,356-task production study was not rerun** during this repair, and the GPU production environment was not reinstalled/revalidated. Reduced CPU success and archived-renderer equality do not establish that every production numerical value has been independently reproduced. Full calculations must pass their own precision/completeness audits; hardware-sensitive threshold decisions and timings may differ.

The [AoS checklist](AOS_REPRODUCIBILITY.md) distinguishes the journal's actual public-code/supplement policy from engineering checks. Prepare a fixed code/configuration/design-input snapshot with the paper as instructed by the journal. Neither this repository audit nor a green CI run constitutes AoS certification.
