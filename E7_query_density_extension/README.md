# E7 query-density extension

This isolated diagnostic checks whether the nearly flat reverse coverage error
is controlled by the fixed query-set fill distance.  It does not modify the
frozen 20-repeat E7 production configuration or its code hash.

It reuses formal repeat 0 for the 512-query baseline and evaluates the same
sample at 1024 and 2048 equal-area query points on a predeclared 13-point
bandwidth grid.  Horizontal lines in the figure are the corresponding ideal
query coverage radii, `delta_query`.

Run:

```powershell
python run_density_curve.py
```

The outputs are `e7_query_density_curve.csv`, `e7_query_density_Erev.png`, and
`e7_query_density_Erev.pdf`.  This is a diagnostic, not a replacement for the
formal E7 evidence.
