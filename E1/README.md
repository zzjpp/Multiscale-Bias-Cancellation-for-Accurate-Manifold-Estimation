# E1 Population Simulation

This repository reproduces the six population maximum-error figures used in
Experiment E1 for three manifold-fitting methods:

- smooth-kernel STSP;
- Gaussian density-ridge fitting (DR); and
- the smooth-kernel Yao--Yau (YY) estimator.

The included data are the final audited population outputs. Historical,
quarantined, fixture, and quick-run results are not used.

## Quick start

Python 3.12 is recommended.

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install -r requirements.txt
python reproduce.py
```

On Windows, after installing the requirements, the same reproduction can be
started with:

```bat
reproduce.bat
```

The command validates the registered data checksum and complete prespecified
grids before writing all PDF and PNG outputs to `figures/`.

## Outputs

| Method | Main-text figure | Supplementary figure |
|---|---|---|
| STSP | `figures/e1_stsp_population_main.{pdf,png}` | `figures/e1_stsp_population_appendix.{pdf,png}` |
| DR | `figures/e1_dr_population_main.{pdf,png}` | `figures/e1_dr_population_appendix.{pdf,png}` |
| Yao--Yau | `figures/yy_population_main_bar_line.{pdf,png}` | `figures/yy_population_appendix_bar_line.{pdf,png}` |

The main-text figures contain
`S^1 subset R^2`, the Clifford torus in `R^4`, and `S^2 subset R^10`.
The supplementary figures contain `S^2 subset R^3` and `S^3 subset R^4`.

All panels report the population maximum error `E_infinity`. STSP and DR use
`h_2` on the horizontal axis; YY uses `r_1`. The displayed slopes are ordinary
least-squares slopes of `log(E_infinity)` against the logarithm of the complete
prespecified scale grid. No adaptive fitting window is used.

## Registered protocols

- **STSP:** `sigma=0.05`,
  `h_2={0.5,0.6,0.7,0.8,0.9,1.0,1.125,1.25}`, and `h_1=h_2/2`.
- **DR:** `sigma=0.05`,
  `h_2={0.15,0.20,0.25,0.30,0.35,0.40,0.45}`, and `h_1=h_2/2`.
- **Yao--Yau:** `sigma=0.05`,
  `r_1={0.4,0.425,0.45,0.475,0.5,0.55,0.6,0.65}`; each directional
  extrapolation uses radii `(0.4,1.0,1.6) r_1` and a common first-stage start.

## Data and provenance

No download is required. `data/e1_population_errors.csv` contains 115 formal
cells: five geometries for each method over its complete registered scale
grid. `data/provenance.json` records:

- the data checksum;
- the source-audit and YY packet checksums;
- the registered protocols and geometry split; and
- the expected Single and Multi slopes.

`config.json` records the same frozen protocol in machine-readable form.
`reproduce.py` refuses to draw figures when the data checksum, grid, resolution
flags, or positivity checks fail.

## Tests

```bash
python -m pytest -q
```

The tests validate all 115 cells, reproduce every registered slope to absolute
tolerance `1e-12`, and execute the complete six-figure pipeline in a temporary
directory. GitHub Actions runs both the tests and the reproduction command.

## Repository layout

```text
E1/
|-- data/                      # audited errors and provenance
|-- figures/                   # six PDF/PNG figure pairs
|-- src/e1_simulation/         # validation and plotting implementation
|-- tests/                     # data-contract and end-to-end tests
|-- reproduce.py               # cross-platform one-command entrypoint
|-- reproduce.bat              # Windows convenience entrypoint
`-- requirements.txt           # exact tested dependency versions
```

In the combined simulation release, continuous checks are defined once at the
repository root under `.github/workflows/`.

## Determinism and scope

The release performs no random sampling: it redraws the figures from frozen,
audited population errors. A fixed seed (`20260926`) is nevertheless recorded
for defensive determinism. Numerical values and fitted slopes are therefore
deterministic for the included data. Rasterization and PDF byte-level hashes
can still vary across operating systems or font backends; scientific values
are checked numerically rather than by requiring cross-platform byte identity.

This repository reproduces the reported figures from formal outputs. It does
not rerun the computationally expensive population quadrature and root-solving
jobs from scratch.

## License

The included `LICENSE` reserves all rights and grants no reuse permission.
Replace it with the authors' chosen open-source license if public reuse is
intended.

