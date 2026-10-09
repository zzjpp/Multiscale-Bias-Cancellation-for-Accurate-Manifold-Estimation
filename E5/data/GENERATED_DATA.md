# Generated data and paper artifacts

E5 uses no external dataset. Raw observations and population quadrature nodes
are generated from the frozen JSON configurations.

The default paper reproduction does not rerun the expensive finite-sample
simulation. It verifies and publishes the exact manuscript figure sources in
`paper_figures/`. Their SHA-256 values and evidence classes are frozen in
`PROVENANCE.json`.

Use `python reproduce.py --mode raw --workers 4` only when a complete numerical
replay is required. Raw packets and diagnostics are written under
`results/raw/` and never overwrite the paper sources.