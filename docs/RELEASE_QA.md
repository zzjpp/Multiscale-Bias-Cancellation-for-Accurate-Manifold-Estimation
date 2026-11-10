# Final-manuscript release checks — E5 recovery update, 2026-11-10

- All 60 historical E5 finite packets are now distributed. Their exact sizes/SHA-256 values match the October 7 migration record and the byte-verified server recovery. The prior local-data gap is resolved.
- Reaggregation uses all 61,440 recorded estimator/query outcomes: 60 packets × 256 queries × four methods, with zero recorded query failures. All 24 medians and 48 quartile-boundary values match the actual selected Figure 8 SVG within its six-decimal coordinate serialization.
- An actual fresh PNG/SVG redraw from the original metrics was run with the unchanged original postprocessor. Its PNG exactly matches the selected paper PNG. SVG metadata/IDs need not be byte-identical; vector data paths are checked independently. The reconstructed PNG was visually inspected.
- 61 current tests passed: 48 root tests and 13 E5 tests, including rejection of missing packets, changed bytes and incorrect median/IQR values. Prior nonuniform sampling QA and the prior 11-command reduced CPU smoke remain applicable to unchanged numerical sources.
- 44 saved numerical/protocol checks pass with no unresolved E5 raw-packet gap. This is archived-result verification, not independent refitting or recomputation of truth distances from fitted coordinates.
- Full saved-result replay was rerun and all 66 selected exports remain byte-identical. The final paper still contains Figures 1–8 and E.1–E.18, using 33 experimental plots and two schematics. Their previous inclusion/visual checks remain applicable because the selected graphics did not change.
- Kernels, initialization, seed rules, sample/query/reference counts, tolerances, weights and failure policies were not changed. The original E5 pilot-selection history, declared/applied-c0 distinction and historical conditioning-gate caveat remain disclosed.

No full production experiment was rerun. The recovered packets store original query metrics and input digests; they do not provide every noisy observation or fitted coordinate. No artifact-badge claim is made.
