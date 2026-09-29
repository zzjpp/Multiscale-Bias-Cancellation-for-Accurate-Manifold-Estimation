# E2: finite-sample manifold-fitting figures

To reproduce the results, run `python reproduce.py` from this directory.

This minimal release generates exactly the six E2 figures used to compare the
finite-sample behavior of smooth-kernel STSP, density ridge (DR), and Yao--Yau
(YY). Each method has one main-text figure and one appendix figure. No E1, E3,
runtime, CPU-time, or parameter-search code is included.

The plots use the formally audited aggregate summaries supplied in `data/`.
They do **not** rerun the expensive Monte Carlo experiment. The release keeps
every predeclared sample size and every reliability label: unreliable cells are
shown with hollow markers and are never removed from the plots.

## Environment

Reference environment: Windows 11, Python 3.12.9, NumPy 2.4.4, Matplotlib
3.11.1. The plotting path is CPU-only and requires no GPU.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## One-command reproduction

From this directory, run:

```bash
python reproduce.py
```

The command verifies both input SHA-256 checksums, validates the complete
`5 geometries × 3 methods × 2 estimators × 2 metrics × 8 sample sizes` grid,
and writes exactly six logical figures under `results/`. Each is exported as
both vector PDF and 400-dpi PNG, so the directory contains 12 graphics plus an
output hash manifest.

| Result | Methods/data shown | Exact command | Output stem |
|---|---|---|---|
| Main STSP | `S^1⊂R^2`, Clifford torus `⊂R^4`, `S^2⊂R^10` | `python reproduce.py` | `figure_e2_stsp_main` |
| Appendix STSP | `S^2⊂R^3`, `S^3⊂R^4` | `python reproduce.py` | `appendix_e2_stsp_remaining` |
| Main DR | main-text geometries | `python reproduce.py` | `figure_e2_dr_main` |
| Appendix DR | appendix geometries | `python reproduce.py` | `appendix_e2_dr_remaining` |
| Main YY | main-text geometries | `python reproduce.py` | `figure_e2_yy_main` |
| Appendix YY | appendix geometries | `python reproduce.py` | `appendix_e2_yy_remaining` |

The two rows in every figure report median average error `E_avg` and median
maximum error `E_infty`; the translucent bands span the first and third
quartiles. Single-scale estimates are blue circles, multiscale estimates are
orange squares, and hollow markers denote cells that failed the frozen
reliability gate.

Expected runtime is approximately 10--20 seconds on a typical laptop CPU. No
network access or external data are required.

## Verification

Run the release-contract and end-to-end plotting tests:

```bash
python -m pytest -q
```

The tests check the input hashes and full plotting grid, invoke the public
entrypoint in a temporary directory, and require exactly six PNG/PDF figure
pairs. This is a functional release check; it is not a claim that the original
Monte Carlo computation is bit-for-bit reproducible from this plotting-only
package.

## Layout

```text
E2/
├── config/config.json            frozen grid, labels, colors, input hashes
├── data/                         formally audited aggregate summaries
├── src/e2plots/                  data validation and plotting implementation
├── tests/test_release.py         end-to-end release contract
├── reproduce.py                 single public entrypoint
├── requirements.txt             pinned plotting/test environment
└── LICENSE
```

## License

MIT; see `LICENSE`.

