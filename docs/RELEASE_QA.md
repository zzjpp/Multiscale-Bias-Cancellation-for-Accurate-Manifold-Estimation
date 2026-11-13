# Release validation scope

## Current paper-aligned organization, 2026-11-13

The current manuscript is the supplied 2026-11-12 PDF, SHA-256 `d0dcc8f6dc30acd72b0be9bd5429049750852448655ff9e3deb8a3b27aa5a275`. Exact public headings and current source directories are listed in `manifold_experiments/catalog.py` and `docs/EXPERIMENTS.md`.

- New and compatibility names generate the same ordered 1,356-task paper plan and 24-task smoke plan. The protocol module's AST is unchanged by formatting.
- Two actual 24-task CPU smoke runs were executed before/after the reorganization. The latter ran in an exported source/design-only directory containing no historical outputs.
- All fitted coordinates, inputs/seeds/hashes, diagnostics, statuses, failures and scientific numeric values matched at zero tolerance: 15,199 numeric leaves. Observed timings and random per-invocation pipeline IDs were excluded explicitly; they are not deterministic scientific outputs.
- 27 protocol/naming/source/failure-path regressions passed, including exact current-location checks of all 329 previously bound core scientific sources.
- Full-format rendering is checked separately using actual existing reference values. This tests the renderer interface; it is not a new production fit or new scientific evidence.
- The source/design export excludes tests and optional historical references by design. The complete repository includes the tests. Release, runtime byte binding, and full-study numerical validation remain different claims.

The compact machine-readable scope is in `docs/REFACTOR_VALIDATION.json`. Earlier QA below is retained as historical evidence and does not make a newly modified file inherit a past PASS. Full production computations and GPU experiments were not rerun during this organization change.

## Historical source-focused release, 2026-11-10

- Historical authority: `arxiv_final.pdf`, SHA-256 `bdc7b87ba20118aaf323bbe25d6b6921235e9f18c1c1d0453604c92f60d561be`, 128 pages. Current manuscript identity is recorded separately in `PAPER_REFERENCE.json`.
- New public entrypoint: `run_paper.py` (new inputs → fitting → saved packets → evaluation → figures).
- Clean Python 3.12.9 CPU environment was actually installed from pinned requirements.
- Nine protocol/failure/custody regressions passed. The full manuscript plan contains 1,356 tasks.
- The actual source-only copy completed 24 reduced tasks across all seven experiment selections. It contained no historical numerical outputs. Executable/configuration/design files in the submitted revision match that tested copy byte-for-byte; ten module README files subsequently clarified dataset instructions.
- The new full-format aggregation/rendering interfaces were also exercised on actual historical values: 32 PNG plots and 32 corresponding rendered PDF pages were viewed. The constructed counterexample was computed afresh in the smoke pipeline. This interface check is not a new production fit or a claim of repeated whole-study science validation.
- 329 core scientific source files retain exactly their prior released bytes. E5 config changes only update figure-number metadata, retaining original numbers under legacy fields.
- Large historical packets/duplicate observation arrays/old exports were preserved privately and removed from the current Git tree. E2–E5 contributor notices remain; E1 and the root use MIT following the repository owner's explicit 2026-11-12 authorization. E3 exact query/reference coordinates remain as required design inputs; small reference tables are optional and excluded from new fitting inputs.
- All six static reproducibility essentials are OK in the source-only export. This is an engineering checklist, not a formal artifact badge.
- GPU wheel names were confirmed through official PyPI metadata. A combined download probe timed out; neither a fresh GPU installation nor a full production GPU rerun is claimed. Original same-input CPU/GPU gates and environment requirements remain enforced.
- Full production numerical reproduction was not rerun during release cleanup. Population accuracy checks, failure accounting, geometry-assisted initialization, local-CP scope, finite-query maxima and E5 pilot provenance retain their disclosed limitations.

## License update, 2026-11-12

E1 and the root now use MIT, preserving the existing E1 copyright holder and E2–E5 notices. Only license/documentation/checksum records changed; no fitting code, scientific configuration or numerical data changed. The 2026-11-10 numerical validation scope above remains historical and is not enlarged by this update.
