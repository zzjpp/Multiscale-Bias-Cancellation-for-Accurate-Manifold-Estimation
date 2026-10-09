# Provenance and reproduction limits

The release was staged from the existing local research tree using an explicit
allowlist. Frozen numerical libraries, saved numerical tables and registered
coordinates were copied byte for byte. Original numerical configurations are
unchanged; **only E2 STSP's unused default storage path** is replaced by the
relative `.reruns/E2_STSP` in both identical config copies. This changes their
config-file hashes, not any scientific value; new-run gates bind the new hashes.
No old config/source PASS hash is pretended to apply to those new bytes. Sources are
bound by `SOURCE_MANIFEST.json`; local private copy/extraction receipts retain
the original-to-release path mapping outside this repository.
The new E2/E3 production entrypoints validate these distributed bytes before
fitting and record the manifest digest; their evaluators reject another or
missing source contract. No historical packet is relabeled as a new run.

Changes are limited to relative-path public entry points, documentation,
tested saved-table plotting bridges and declared extraction of existing
functions. The E3 numerical functions `install_dr_recorder`, `dr_cell`,
`yy_cells`, `original_forward_metrics`, `stsp_cell` and YY output adapter were
extracted without changing their original function text. E2's support
functions were similarly extracted; the public STSP wrapper uses the original
per-geometry `(d,D)` coefficient call rather than incorrectly applying a
corrugated-only wrapper's fixed `(2,3)` to all geometries.
Some unchanged geometry symbols/configuration provenance still mention the
historical name E7. They identify the original corrugated source/seed lineage;
there is no standalone E7 directory, experiment entrypoint or import dependency.

The E1 default selects the paper's raw-error figures plus the normalized circle;
additional diagnostics remain available. E2 plots the paper's coarse Single
and Multi but retains independent fine Single in the table. E3 uses the
selected full table, an explicit **reverse-only** reference-count bridge,
and the existing no-footer renderer. No failure is relabeled successful and
no metric is changed to improve appearance.

The latest E3 table combines valid reused S1 n12,000 data and DR/YY n24,000
components with newly computed missing components, plus the subsequent actual
corrugated reverse evaluation using 262,144 references. This adaptive history,
platform-specific sampling and YY source versions are not erased. Only the
selected 0.4/1/1.6 YY version is the paper default; 0.8 was a separate diagnostic.

E4 retains the actual rerun with all methods using archived ENF h1 coordinates,
its ten frozen n200,000 parents, query arrays and source-bound original gate.
E5 retains selected exports and the original numerical ablation code. E6
retains the fixed-fit inputs and unchanged source/audit package.

Figure replay, new CPU smoke, archival scientific audit and an independent
full estimator rerun are different evidence classes. Local release testing
does not imply all-query reliability, absence of failures, a new GPU gate,
asymptotic proof or artifact-evaluation badge. GPU runners refuse an untested
environment; no installation, network access or remote scheduler is implicit.
