# Reproducibility checklist

| Essential | Status | Evidence |
|---|---|---|
| README | OK | Setup, commands, outputs, protocols, and scope are documented. |
| Environment | OK | Python version guidance and exact dependency pins are provided. |
| Seed | OK | No RNG is used; seed `20260926` is still set and recorded defensively. |
| Entrypoint | OK | `python reproduce.py` and `reproduce.bat`. |
| Data | OK | All 115 formal cells are included with provenance and checksums. |
| Results command | OK | One command regenerates all six PDF/PNG figure pairs. |
| Tests | OK | Data-contract, registered-slope, and end-to-end figure tests. |

## Expected resource use

- Hardware: CPU only; no GPU is used.
- External services: none.
- Network access after dependency installation: none.
- Expected runtime: typically below one minute on a contemporary desktop CPU.

## Verification boundary

The package reproduces the six figures from audited population-error values.
It does not claim byte-identical PDF/PNG files across operating systems, and it
does not rerun the upstream population integration and nonlinear solvers.

