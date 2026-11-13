# Finite-Sample Behavior at Fixed Scales

Paper sections 5.2; E.2; Figures 5, E.4–E.6. Compatibility code: `E2`.

From the repository root:

```bash
python -m manifold_experiments --experiment fixed-scales --backend cpu --output NEW
```

σ=0.005; h1=0.1, h2=0.2 for STSP/DR; YY uses (0.8,1,1.6)×40σ and r0=32σ. Eight nested sample sizes from 2,000 to 500,000, 144 fixed queries, 20 repetitions. Original medians, IQRs, partial-query metrics and reliability rules remain. No population-reference overlay is added.

Configuration/source locations: `frozen/{STSP,DR,YY}/` configs and `corrugated/frozen/` variants. Original version-specific `src/` and `frozen/` packages retain numerical behavior and provenance names. Use the public pipeline, not historical deployment/replay scripts, for a complete fresh run.

No external dataset is required; observations are generated from the original geometry and seed definitions. Add `--profile smoke` for a reduced real fit that is not paper evidence. See the root README, `docs/EXPERIMENTS.md`, and `docs/FIGURE_MAP.json` for environments, output paths and validation limits.
