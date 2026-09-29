# Included data

This directory contains only aggregated, formally audited E2 summaries. No
external dataset is required.

- `stsp_formal_summary.csv`: smooth-kernel STSP summary. The plotting code uses
  only `E_avg` and `E_infty`; the additional audited diagnostic rows are kept
  unchanged for provenance.
- `dr_yy_formal_summary.csv`: DR and tuned Yao--Yau summary for the same two
  metrics.

The exact SHA-256 values are frozen in `config/config.json` and checked
before plotting. Each plotted point is a median with its first-to-third
quartile band. Reliability labels are preserved from the formal audits; the
release does not recompute, remove, or post-select failed cells.

