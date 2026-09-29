# E7: Complex-Geometry Stress Test

This is the minimal public release for the three E7 figures. Each figure has
two panels: maximum error on the left and average error on the right. The three
figures report STSP, DR, and Yao--Yau, respectively. No runtime plots,
reverse-error plots, exploratory parameter searches, raw packets, or server
logs are included.

## Frozen experiment

- Geometry: corrugated torus embedded in R^3.
- Sample size: n = 12,000.
- Noise standard deviation: sigma = 0.015.
- Query points: 512.
- Independent repetitions: 20.
- Scale grid: 28 prespecified values from 0.06 to 5.0.
- STSP and DR use h2 on the horizontal axis with h1 = h2 / 2.
- Yao--Yau uses r1 on the horizontal axis.
- A point is reliable only when the aggregate query-failure rate is at most
  1% and at least 18 of 20 repetitions are complete. Unreliable points are
  retained and drawn with hollow markers.

The included CSV files are the immutable, formally audited aggregates. No
external data download and no GPU are required to regenerate the figures.

## One-click reproduction

Windows:

    reproduce.bat

Linux/macOS:

    sh reproduce.sh

After dependencies are installed, the direct command is:

    python reproduce.py

This command generates exactly these three figures, with PNG and PDF versions:

- `figures/e7_stsp_maximum_average_errors.*`
- `figures/e7_dr_maximum_average_errors.*`
- `figures/e7_yy_maximum_average_errors.*`

To write into another directory:

    python reproduce.py --output-dir path/to/output

## Repository layout

- `config.json`: frozen setup, plotting style, schemas, and complete scale grid.
- `data/`: three audited aggregate CSV files and their audit records.
- `src/e7_release/`: schema validation and publication plotting code.
- `tests/`: frozen-data and end-to-end plotting smoke tests.
- `figures/`: the three released figures in PNG and PDF formats.
- `SHA256SUMS.txt`: checksums for release inputs and outputs.

## Provenance

STSP passed its original E7 audit. DR and Yao--Yau passed a separate 40-packet
formal audit (20 repetitions per method). The aggregate CSV and audit hashes
were verified after transfer from the compute server. See `data/DATA.md` and
the JSON audit records for the exact provenance boundary.
