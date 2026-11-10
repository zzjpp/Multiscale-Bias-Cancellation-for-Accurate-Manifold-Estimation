# E5 — directional YY ablation

```bash
python E5/reproduce.py --mode paper
python E5/scripts/verify_historical_finite.py
python E5/scripts/replay_historical_finite.py --output NEW
python E5/reproduce.py --mode smoke
```

`paper` first checks all 60 recovered finite packets and reaggregates their query metrics against the actual Figure 8 median/IQR vector paths, then validates/copies the four selected exports to E5/results/figures. Figure 8 uses finite PNG/SVG; Figure E.14 uses population PNG/PDF. Filenames and selected export bytes remain unchanged.

`replay_historical_finite.py` actually redraws Figure 8 from the original per-query metrics, writing PNG/SVG and the per-repeat/summary CSVs to a fresh directory. The reconstructed PNG matches the selected paper PNG byte-for-byte in the recorded environment; SVG metadata and element IDs may differ across runs, while all vector data paths are checked numerically. Plotting-library changes may affect rendering bytes.

Seven population packets are in data/population; all 60 historical finite packets, their original audits/configuration, migration/recovery receipts and new reaggregation tables are in data/finite. No E5 finite packet is missing from this release. The packets store recorded query metrics, not all original observation arrays or fitted coordinates. Verification is not independent refitting or truth-distance recomputation.

Full expensive new CPU calculations:

```bash
python E5/reproduce.py --mode raw --workers 4
python E5/reproduce.py --mode raw-plot
```

The original algorithm/configuration, seed rules, common three fitted points and failure retention are unchanged. Population uses 64 queries, seven sigma levels and radii (1.25,5,8)sigma. Finite uses sigma=.05, radii (5,6,12)sigma, n=240000/480000/960000, 20 repeats and 256 queries. Scalar controls and directional cancellation reuse the same fits. The old 6/22 figure labels in configuration are historical metadata; current numbering is Figure 8 and Figure E.14.

The finite configuration was selected after pilot screening. Its historical complementary-moment condition number 244.94 exceeded the original threshold 50; the retained residual-based gate and selection history are not silently upgraded. Actual c0=3 is distinguished from its legacy declared value. Geometry-based initialization is controlled. Reduced smoke is not production accuracy validation.
