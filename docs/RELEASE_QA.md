# Actual release checks, 2026-11-07

- 41 archived-result/protocol checks match the supplied manuscript.
- All 48 root test cases reached PASS: 39 in the initial run; the remaining 9 passed after fixing the local temporary-directory environment and explicitly versioning the public documentation contract.
- Five original nonuniform sampling tests passed. One actual reduced nonuniform E1 CPU query succeeded, with zero failures; this is not a production precision gate.
- Full saved-result replay generated 33 logical paper plots. All 66 paper PNG/vector exports match the current selected source files byte-for-byte in the recorded environment. Two additional counterexample branch-switch exports are diagnostic, not paper figures.
- The complete original CPU smoke entrypoint passed all 11 method/experiment commands, including E5 population and finite smoke.
- The six static reproducibility essentials are OK. Source/data checksum validation and credential-literal/size scans passed.

The original engineering failure and temporary-directory errors were retained locally; they are not relabeled as numerical failures or hidden behind a full-grid scientific PASS. No new production GPU run or artifact badge is claimed. Missing E5 finite raw packets remain explicitly disclosed.
