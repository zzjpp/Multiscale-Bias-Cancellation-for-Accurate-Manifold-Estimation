# Final-manuscript release checks, 2026-11-10

- Authority: `arxiv_final.pdf`, 128 pages, SHA-256 in PAPER_REFERENCE.json.
- 61 tests passed: 48 root interface/evaluation/provenance checks, eight E5 tests and five original nonuniform sampling tests. The initial test collection command had a wrong test-file path and ran zero tests; corrected suites passed. Logs are retained in the local task archive.
- Eleven actual reduced CPU experiment commands passed, including E5 population and finite smoke. Smoke does not validate production accuracy or GPU equivalence.
- 41 saved numerical/protocol checks matched. The known 60 missing E5 finite raw packets remain disclosed.
- Full saved-result replay emitted exactly the 33 selected experimental plots / 66 exports. All 66 export bytes match the selected source files. E5 finite exports are checked copies, not independently recalculated medians/IQRs.
- All 33 experimental plots and both schematics match inclusions in the final PDF: exact decoded vector content streams or exact embedded raster pixels. Figure 1 now uses the final native TikZ schematic instead of the previous Matplotlib schematic.
- Actual overview review covered 33 PNGs and all 32 complete PDF pages rendered in contact sheets. No clipping, missing panels, unintended green background, or omitted reliability markers was observed.
- Retained scientific input arrays, frozen implementations, raw packets, tables and failure records are byte-identical to the pre-cleanup snapshot. No production fits or parameter tuning were performed.
- Removed independent experiments/plots are reversibly archived outside the repository. Immutable historical plans and required frozen code remain as evidence/dependencies, not as additional selected experiments.

Environment: Windows Python 3.12, versions pinned in requirements.txt; original GPU environment remains separate. A passing release check is not independent experimental replication or an artifact badge.
