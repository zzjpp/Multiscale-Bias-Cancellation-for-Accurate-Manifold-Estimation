# Multiscale Bias Cancellation for STSP under Nonuniform Sampling — Finite Sample

Paper sections E.5; Figures E.17–E.18. Compatibility code: `nonuniform_E2`.

From the repository root:

```bash
python -m manifold_experiments --experiment nonuniform-fixed-scales --backend cpu --output NEW
```

STSP only; same nonuniform densities as population, clean area-uniform rejection before Gaussian noise, with the original separate acceptance stream. The fixed-scale finite-sample protocol otherwise follows §5.2. Unreported nonuniform DR/YY studies are not public experiment selections.

Configuration/source locations: `sampling.py` and the fixed-scale fitter configurations. Original version-specific `src/` and `frozen/` packages retain numerical behavior and provenance names. Use the public pipeline, not historical deployment/replay scripts, for a complete fresh run.

No external dataset is required; observations are generated from the original geometry and seed definitions. Add `--profile smoke` for a reduced real fit that is not paper evidence. See the root README, `docs/EXPERIMENTS.md`, and `docs/FIGURE_MAP.json` for environments, output paths and validation limits.
