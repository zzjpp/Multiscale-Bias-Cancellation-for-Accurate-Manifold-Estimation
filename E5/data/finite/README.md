# Original E5 finite-sample results

The 60 `packets/e5_directional_n*_r*.json` files are the historical held-out results for n=240000/480000/960000, repeats 0–19, 256 queries, and four methods. They were recovered from the post-move BeeGFS project directory; all sizes and SHA-256 values match the October 7 migration receipt. They are not newly simulated data.

`original_config.json`, `ORIGINAL_MIGRATION_RECEIPT.json`, `RECOVERY_RECEIPT.json` and the immutable `audit/` files preserve the historical experiment and recovery history. `REAGGREGATION_AUDIT.json` and the two dual-metric CSVs are new postprocessing of the original query metrics.

Run `python E5/scripts/verify_historical_finite.py` from the repository root. It checks the entire packet grid and every query, recomputes per-repeat maxima/means and medians/IQRs, and compares all 24 medians and 48 quartile boundaries to the actual selected Figure 8 SVG paths. The SVG tolerance is in rounded vector coordinates, not a relaxed scientific fitting tolerance.

These packets store per-query metrics and input/axis digests; they do not distribute every noisy observation or fitted coordinate. This audit reaggregates recorded geometric metrics and does not independently recompute truth distances or rerun fitting. Original pilot selection, declared/applied c0 distinction and the historical rejected conditioning criterion remain explicitly recorded; recovery does not upgrade them to predeclared formal evidence.
