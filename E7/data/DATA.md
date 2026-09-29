# Data provenance

No external dataset is used. E7 is a synthetic experiment on a corrugated
torus. This plotting release bundles only the formally audited aggregate
tables needed for the three publication figures:

- `e7_stsp_aggregated.csv`: STSP, copied byte-for-byte from the audited E7
  aggregate.
- `e7_dr_aggregated.csv`: DR, copied byte-for-byte from the formally audited
  GPU production aggregate.
- `e7_yy_aggregated.csv`: Yao--Yau, copied byte-for-byte from the formally
  audited GPU production aggregate.

The numerical experiment is intentionally outside this minimal plotting
release. Failed and unreliable cells remain present in the CSV files and are
never interpolated or removed.
