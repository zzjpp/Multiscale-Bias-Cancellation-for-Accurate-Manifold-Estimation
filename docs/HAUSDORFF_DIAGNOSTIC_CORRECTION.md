# YY summary diagnostic correction — 2026-11-19

## Confirmed error

The old `finite_cloud_hausdorff_upper = max(fill_distance, geometric_max)` field in `e56v2.experiment.summarize()` was mislabeled. It is not a valid bound on the two-sided Hausdorff distance between the fitted point cloud and the continuous true curve. Geometric distance to the curve ignores displacement of the correspondence along the curve.

For example, take four equally spaced queries on the unit circle, and four distinct outputs on the arc with angles between −0.2 and 0.2. All output-to-circle distances are zero, and the query fill radius is `2 sin(π/8) ≈ 0.7654`, but the output cloud's Hausdorff distance to the circle is `2 cos(0.1) ≈ 1.9900`. Both the old maximum and the sum of fill radius with `geometric_max` fail to bound it.

If retained base points `x_j` genuinely cover the manifold with radius `δ`, and `y_j` are their paired outputs, the triangle inequality gives

```text
d_H({y_j}, M) ≤ δ + max_j ||y_j − x_j||.
```

This requires the full paired displacement and a valid continuous covering radius. The implementation's `retained_fill_distance` instead takes a maximum over 8192 numerical reference points. That is a sampled coverage estimate, not a certified continuous covering radius. No replacement Hausdorff certificate is introduced by this correction.

## API correction and compatibility

New summaries use `sampled_fill_geometric_max_diagnostic` for the same numeric maximum. The new name explicitly describes a diagnostic, not a Hausdorff distance or upper bound. The misleading field is absent from new summaries; downstream users must not continue treating old values as bounds.

Native result schemas advance to `E5_RELEASE_POPULATION_CELL_V2` and `E5_RELEASE_FINITE_PAPER_PROTOCOL_CELL_V3`. Historical packets retain their original schemas, field names and source provenance; they are not rewritten. New runs have a different source identity and cannot silently resume packets from the previous revision.

## Impact on the paper figures

Figure 8's finite-sample renderer computes `E_inf` and `E_avg` directly from each retained query's `metrics[method]['geometric']`, then takes the original repeat quantiles. It does not read the mislabeled field. Population Figure E.14 and Table 2 use geometric/normal error summaries, which are unchanged. The native fitters, geometry, coefficients, weights, seeds, queries, failure handling and metric arithmetic are unchanged.

The original scientific-source hashes remain recorded in `SCIENTIFIC_PRESERVATION.json`. One subsequent amendment explicitly records this file's changed hash and a normalized AST check allowing only the diagnostic label, result-schema identifiers and docstrings to differ. The remaining 328 bound scientific files retain their exact original bytes.

Regression checks include a tangential-drift counterexample, empty-query/undefined cases, unchanged valid metrics, and the distinction between sampled and continuous coverage. The accompanying audit compares summaries and E5 exports on actual existing packets; it is not a new full-size fit or an independent reproduction of all E5 results.

## Checks performed

Local checks passed 53 tests and two actual reduced YY fitting/evaluation tasks. After normalizing only the declared diagnostic name and native schema identifiers, both tasks' entire scientific results equal their prior results. Reaggregation of the actual 7 population and 60 finite packets preserved 2412 valid summary entries, all Figure 8 geometric rows and repeat quantiles. The resulting Figure 8 and population-ablation PNGs are byte-identical to the previous checked exports. These were reduced fits and stored-metric/plot checks; no full-size refitting was performed and no Hausdorff certificate is asserted.
