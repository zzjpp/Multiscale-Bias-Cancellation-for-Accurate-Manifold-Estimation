# E7 artifact reproducibility checklist

| Essential | Status | Evidence |
|---|---|---|
| README | OK | Exact commands and output files are documented. |
| Environment | OK | `requirements.txt` pins all plotting/test packages. |
| Seed | N/A | Plotting audited deterministic aggregates uses no RNG. |
| Entrypoint | OK | `reproduce.py`, `reproduce.bat`, and `reproduce.sh`. |
| Data | OK | Synthetic-data provenance is documented in `data/DATA.md`. |
| Results command | OK | `python reproduce.py` creates exactly three figures. |
| Smoke test | OK | Full plotting path is exercised in `tests/test_release.py`. |

This checklist establishes a testable artifact path; it does not replace a
separate numerical code-versus-paper verification.
